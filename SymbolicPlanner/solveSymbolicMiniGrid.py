from gym_minigrid.wrappers import *
import torch
import utils
from SymbolicMCTSWithUncertainty import MCTS
#from SymbolicMCTS import MCTS
import cv2
import numpy as np
from gym_minigrid.window import Window
from DNDDeltaSymbolicPlanner import SymbolicPlanner
import pickle
import collections

# 'BC' : behavioural cloning model
# 'DQFD' : Deep Q-learning from demonstrations
# 'Ablation': 1-step ahead only solution (ablation study: doesn't plan 5 steps ahead)
# 'Conviction': our proposed solution
MODEL_NAME = 'Conviction'
GRID_DIFFICULTY = 'Intermediate'

if GRID_DIFFICULTY == 'Easy':
    W = 192
    H = 192
    TILE_SIZE = 32

elif GRID_DIFFICULTY == 'Intermediate':
    W = 228
    H = 228
    TILE_SIZE = 12

device = 'cuda'
ACTION_SPACE = 3
Z_DIM = 2048
NC = 3
DET_SEED = 555

np.random.seed(DET_SEED)

TEST_EPISODE = 701
NUM_EPISODES = 800
SKIP_LEVELS = 700

model_path = 'saved_models/best_full_modelV3'
data_dir = 'data/'

print("Initializing environment...")

if GRID_DIFFICULTY == 'Easy':
    env = gym.make('MiniGrid-Empty-Random-6x6-v0')
elif GRID_DIFFICULTY == 'Intermediate':
    env = gym.make('MiniGrid-FourRooms-v0')
# elif GRID_DIFFICULTY == 'Hard':
#     env = gym.make('MiniGrid-MultiRoom-N6-v0')

env = FullyObsWrapper(env)

window = Window('gym_minigrid')

obs = env.reset()                   # This now produces an RGB tensor only

print("Loading pre-trained model...")

#  load model
model = SymbolicPlanner()
model.load_state_dict(torch.load(model_path))

model.to(device)
model.eval()

print("Loading DND...")

model.memory = pickle.load(open('saved_models/dnd.pkl', 'rb'))

MCTS_planner = MCTS(model)

def get_actions(obs):

    def extract_direction(obs):
        direction = 0
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i, j, 0] == 10:
                    direction = obs[i, j, 2]

        return direction

    dir = extract_direction(obs['image'])
    data = obs['image'][:, :, 0]
    data_row = np.reshape(data, [-1])

    # concatenate direction to data vector
    data_row = np.concatenate([[dir], data_row])

    # print("data_row.shape = ", data_row.shape)
    # utils.display(data_row)

    action_sequence = MCTS_planner.plan(data_row)
    return np.array(action_sequence)

print("Solving mini-grid...")

log_done_counter = 0
out = None

# Performance stats
test_successes = 0
total_test_episodes = 0
is_test = False
frame_counter = 0
frames_per_episode = 0

while log_done_counter < NUM_EPISODES:

    if log_done_counter >= SKIP_LEVELS:

        if log_done_counter == TEST_EPISODE:
            is_test = True

        tmp_obs = obs
        actions = get_actions(obs)

        for current_a in actions:
            env.render()
            obs, rewards, done, _ = env.step(current_a)

            frame_counter += 1

            if done:
                break

    else:

        obs, rewards, done, _ = env.step(6)

    if done:

        if is_test:
            if frame_counter <= 90:
                test_successes += 1
                print("==> Success!")
            frames_per_episode += frame_counter
            total_test_episodes += 1

        log_done_counter += 1
        print("Level %s Done!" % log_done_counter)

        obs = env.reset()

        frame_counter = 0

frames_per_episode /= float(total_test_episodes)
R = float(test_successes) / float(total_test_episodes)

print("==> Frames per episode = ", frames_per_episode)
print("==> Success rate = %s (%s failures)" % (R, total_test_episodes - test_successes))

if out is not None:
    out.release()

