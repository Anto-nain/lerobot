import lerobot
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

repo_id = "lirislab/bimanual_scene1_take4"
dataset = LeRobotDataset(repo_id)
print(dataset.features.keys())

keep = {#keep frames : [from, to excluded]
    0:[0,98],
    2:[2,119],
    3:[0,113],
    4:[3,128],
    5:[0,127],
    6:[6,113],
    9:[0,119],
    10:[2,114],
    11:[0,119],
    12:[2,102],
    15:[3,135],
    16:[0,122],
    21:[0,126],
    22:[2,118],
    26:[0,100],
    27:[0,110],
    31:[0,123],
    32:[0,124],
    35:[0,116],
    37:[0,118],
    41:[0,123],
    45:[0,100],    
}
def process_episode(episode_index):
    episode_frames = []
    start_idx = dataset.episode_data_index['from'][episode_index].item()
    end_idx = dataset.episode_data_index['to'][episode_index].item()

    for frame_index in range(start_idx, end_idx):
        frame = dataset[frame_index]
        if (episode_index not in keep.keys()):
            episode_frames.append(frame)
        elif (frame_index - start_idx >= keep[episode_index][0] and frame_index - start_idx < keep[episode_index][1]):
            episode_frames.append(frame)
        else:
            print(f"Deleted Episode [{episode_index}] frame {frame_index - start_idx}/{end_idx - start_idx - 1}")

    return episode_frames

for i in range(dataset.meta.total_episodes):
    print(f"Processing episode {i}")
    episode_frames = process_episode(i)
    dataset.episode_buffer = dataset.create_episode_buffer(episode_index=i)
    print(dataset.episode_buffer)
    
    for frame in episode_frames:
        #print(frame)
        try:
            for key in ["task", "action", "observation.state", "observation.images.top", "observation.images.front","timestamp","frame_index","index","task_index"]:
                dataset.episode_buffer[key].append(frame[key])
            dataset.episode_buffer["size"] += 1
        except Exception as e:
            print(e)
            print(frame)
    dataset.save_episode()

# Consolidate the dataset
dataset.consolidate()

# Push the cleaned dataset to the Hugging Face Hub
#dataset.push_to_hub(tags=['cleaned'], private=False)