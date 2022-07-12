from gym_minigrid.wrappers import *
import torch
import utils
from DeltaSymbolicPlanner import SymbolicPlanner
import numpy as np
import pickle

device = 'cuda'
ACTION_SPACE = 3
Z_DIM = 512
NC = 3
LOAD_DND = True

model_path = 'saved_models/best_full_modelV3_backup'
data_dir = './data/'

GRID_DIFFICULTY = 'Intermediate'

if GRID_DIFFICULTY == 'Easy':
    W = 192
    H = 192
    TILE_SIZE = 32

elif GRID_DIFFICULTY == 'Intermediate':
    W = 228
    H = 228
    TILE_SIZE = 12

# load MotionPlanner model
model = SymbolicPlanner(action_space=ACTION_SPACE, device=device, dict_len=500)

# Load neural network module
model.load_state_dict(torch.load(model_path))

model.to(device)
model.eval()

# Prime DND
X, a, v = utils.loadMinigridSymbolicDemos(data_dir)

print("len(X) = %i, len(x) = %i" % (len(X), len(a)))

T = 700
training_X = X[:T]
training_v = v[:T]
training_a = a[:T]

test_X = X[T:]
test_a = []
for i in range(len(a)-T):
  test_a.append(np.array(a[T+i]))

test_v = []
for i in range(len(v)-T):
  test_v.append(np.array(v[T+i]))

step_counter = 0
for tmp_a in training_a:
    step_counter += len(tmp_a)

print("step_counter = ", step_counter)

def prime_DND(x, a):
    for a_idx in range(ACTION_SPACE):
        model.memory[a_idx].reset_memory()

    print("len of x = ", len(x))
    for seq_idx in range(len(x)):
        progress = seq_idx / float(len(x))
        print("Progress = %.4f %%" % (progress * 100.))
        tmp_X = torch.from_numpy(np.array(x[seq_idx]))
        embeddings = torch.reshape(model.encoder(tmp_X.to(device)), [len(x[seq_idx]), Z_DIM])

        actions = np.reshape(a[seq_idx], [-1]).astype(int)
        for i in range(embeddings.shape[0] - 1):
            model.memory[actions[i + 1]].save_memory(embeddings[i], embeddings[i + 1])

def display_raw(x):
    direction = x[0]
    print("direction: ", direction)
    for i in range(19):
        str_row = ""
        for j in range(19):
            actual_idx = 1 + (i * 19 + j)
            str_row += "|%.2f|" % x[actual_idx]

        print(str_row)
        print("------------------------------------------------------------------------------------------")

def display(x):
    direction = x[0]
    print("direction: ", direction)
    for i in range(19):
        str_row = ""
        for j in range(19):
            actual_idx = 1 + (j * 19 + i)

            if round(x[actual_idx]) == 1.0:
                # empty cell
                str_row += "|   |"
            elif round(x[actual_idx]) == 2.0:
                # wall
                str_row += "|||||"
            elif round(x[actual_idx]) == 10.0:
                # cursor
                if direction == 0:
                    str_row += "| > |"
                elif direction == 1:
                    str_row += "| V |"
                elif direction == 2:
                    str_row += "| < |"
                else:
                    str_row += "| A |"
            elif round(x[actual_idx]) == 8.0:
                # goal
                str_row += "| # |"
            else:
                str_row += "|%.2f|" % x[actual_idx]

        print(str_row)
        print("------------------------------------------------------------------------------------------")

input_str = ''

with torch.no_grad():
    while input_str != 'q':
        # select a random time step in a random episode. Display its current estimated value, as well as the image.
        episode_idx = np.random.choice(np.arange(len(test_X)))
        step_idx = np.random.choice(np.arange(len(test_X[episode_idx]) - 1))

        x_input = test_X[episode_idx][step_idx]
        display(x_input)

        # input a sequence of 5 actions from the user
        print("Action will be: ", test_a[episode_idx][step_idx])
        input_str = input("Enter action: ")

        x_input = test_X[episode_idx][step_idx+1]
        display(x_input)

        print("================================================================================")

