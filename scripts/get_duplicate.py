import lerobot
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

def get_motor_positions(frame):
    leader_positions = frame['action']
    follower_positions = frame['observation.state']
    return leader_positions, follower_positions

repo_id = "lirislab/bimanual_scene1_take4"
dataset = LeRobotDataset(repo_id)
camera_keys = dataset.meta.camera_keys
print(dataset)

for episode_index in range(dataset.meta.total_episodes):
    print()
    previous_leader_positions = None
    previous_follower_positions = None
    start_idx = dataset.episode_data_index['from'][episode_index].item()
    end_idx = dataset.episode_data_index['to'][episode_index].item()

    for frame_index in range(start_idx, end_idx):
        frame = dataset[frame_index]
        print(frame)
        break
        current_leader_positions, current_follower_positions = get_motor_positions(frame)

        if not (previous_leader_positions is None or previous_follower_positions is None or
            not torch.equal(previous_leader_positions, current_leader_positions)):# or
            #not torch.equal(previous_follower_positions, current_follower_positions)):
            print(f"Episode [{episode_index}] frame {frame_index-start_idx}/{end_idx-start_idx-1}")
        else:
            previous_leader_positions = current_leader_positions
            previous_follower_positions = current_follower_positions
    break
        



