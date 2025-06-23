import gradio as gr
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer,CharacterNetworkGenerator
from character_chatbot import CharacterChatBot
from dotenv import load_dotenv
load_dotenv()
import os

def get_themes(theme_list, subtitles_path, save_path):
    theme_list = theme_list.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    output_df = output_df[theme_list]

    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme', 'Score']

    output_chart = gr.BarPlot(
        output_df,
        x='Theme',
        y='Score',
        title="Series Theme",
        tooltip=["Theme","Score"],
        vertical=False,
        width=500,
        height=260
    )

    return output_chart

def get_character_network(subtitles_path, ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html

def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatBot("longbui125/Chatbot_Llama-3-8B",
                                         huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
                                         )

    output = character_chatbot.chat(message, history)
    output = output.strip()
    return output

def main():
    with gr.Blocks() as iface:
        #Character chatbot
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Kang Tae-moo Chat</h1>")
                gr.ChatInterface(chat_with_character_chatbot)




        #Theme classification
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifier)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Theme")
                        subtitles_path = gr.Textbox(label="Subtitles or script path")
                        save_path = gr.Textbox(label="Save path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])

        #Character network
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network(NERS and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or script path")
                        ner_path = gr.Textbox(label="NERs save path")
                        get_network_graph_buttion = gr.Button("Get character network")
                        get_network_graph_buttion.click(get_character_network, inputs=[subtitles_path, ner_path], outputs=[network_html])


    iface.launch(share=True)
            

if __name__ == "__main__":
    main()