import torch
import huggingface_hub
import pandas as pd
import re
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import gc
import time

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

def remove_paranthesis(text):
    return re.sub(r'\(.*?\)', '', text)

class CharacterChatBot():
    def __init__(self, model_path, data_path="/content/Movie-Analysis-NLP/dataset/business_proposal.csv", huggingface_token=None):
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_history = 20

        if huggingface_token:
            huggingface_hub.login(huggingface_token)

        if huggingface_hub.repo_exists(model_path):
            self.model, self.tokenizer = self.load_model(model_path)
        else:
            print("Model not found. Training will start...")
            train_dataset = self.load_data()
            self.train(self.base_model_path, train_dataset)
            self.model, self.tokenizer = self.load_model(model_path)

    def chat(self, message, history):
      if len(history) > self.max_history:
          history = history[-self.max_history:]

      messages = []
      messages.append({"role": "system", "content": "You are Kang Tae-moo from the movie 'Business Proposal'. Your response should reflect his personality and speech pattern.\n"})

      for past in history:
          messages.append({"role": "user", "content": past[0]})
          messages.append({"role": "assistant", "content": past[1]})

      prompt = ""
      for msg in messages:
          if msg["role"] == "system":
              prompt += f"[System]: {msg['content']}\n"
          elif msg["role"] == "user":
              prompt += f"[User]: {msg['content']}\n"
          elif msg["role"] == "assistant":
              prompt += f"[Kang Tae-moo]: {msg['content']}\n"

      # Câu hỏi hiện tại
      prompt += f"[User]: {message}\n[Kang Tae-moo]:"

      inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

      with torch.no_grad():
          output_ids = self.model.generate(
              **inputs,
              max_new_tokens=64,
              temperature=0.6,
              top_p=0.9,
              do_sample=True,
              pad_token_id=self.tokenizer.eos_token_id
          )

      generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
      response = generated_text[len(prompt):].strip()

      response = re.split(r"[.?!]\s|\n", response)[0].strip() + "."

      del inputs, output_ids
      gc.collect()
      torch.cuda.empty_cache()

      if NVML_AVAILABLE:
          handle = nvmlDeviceGetHandleByIndex(0)
          info = nvmlDeviceGetMemoryInfo(handle)
          used_gb = info.used / (1024 ** 3)
          print(f"[INFO] GPU Memory Usage: {used_gb:.2f} GB")

          if used_gb > 35:
              print("[WARNING] GPU RAM vượt 35GB, nên reset Colab Kernel để tránh OOM!")

      return response

    def load_model(self, model_path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def train(self, base_model_name_or_path, dataset):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
        
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )

        training_arguments = SFTConfig(
            output_dir="./results",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=200,
            logging_steps=10,
            learning_rate=2e-4,
            fp16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.3,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none"
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_arguments,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=512
        )

        trainer.train()
        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, "final_ckpt")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        del model, base_model
        gc.collect()
        torch.cuda.empty_cache()

    def load_data(self):
        df = pd.read_csv(self.data_path).dropna()
        df['line'] = df['line'].apply(remove_paranthesis)
        df['word_count'] = df['line'].str.strip().str.split().apply(len)

        df["KangTaeMoo_flag"] = (df['name'] == 'Kang Tae-moo') & (df['word_count'] > 4)
        indices = df[df["KangTaeMoo_flag"] & (df.index > 0)].index

        system_prompt = "You are Kang Tae-moo from the movie 'Business Proposal'. Your response should reflect his personality and speech pattern.\n"
        prompts = [
            f"{system_prompt}{df.iloc[i - 1]['line']}\n{df.iloc[i]['line']}"
            for i in indices
        ]

        return Dataset.from_pandas(pd.DataFrame({"prompt": prompts}))