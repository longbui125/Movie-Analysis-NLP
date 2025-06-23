from glob import glob
import pandas as pd

def load_subtitles_dataset(dataset_path):
    subtitles_paths = glob(dataset_path+'/*.ass')

    scripts = []
    episode_num = []

    for path in subtitles_paths:
        #Read lines
        with open(path,'r') as file:
            lines = file.readlines()
            lines = lines[15:]
            lines = [",".join(line.split(',')[9:]) for line in lines]

        script = " ".join(lines)

        episode = int(path.split('_ep')[-1].split('.')[0].strip())

        scripts.append(script)
        episode_num.append(episode)

    df = pd.DataFrame.from_dict({"episode": episode_num, "scripts":scripts})

    return df