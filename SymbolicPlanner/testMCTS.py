from gym_minigrid.wrappers import *
import torch
import utils
from DNDDeltaSymbolicPlanner import SymbolicPlanner
from SymbolicMCTSWithUncertainty import MCTS
import numpy as np
import pickle

device = 'cuda'
ACTION_SPACE = 3
Z_DIM = 512
NC = 3
LOAD_DND = True

model_path = 'saved_models/best_full_modelV3'
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

print("Loading DND...")

model.memory = pickle.load(open('saved_models/dnd.pkl', 'rb'))

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
                if round(direction) == 0:
                    str_row += "| > |"
                elif round(direction) == 1:
                    str_row += "| V |"
                elif round(direction) == 2:
                    str_row += "| < |"
                elif round(direction) == 3:
                    str_row += "| A |"
                else:
                    str_row += "| ? |"
            elif round(x[actual_idx]) == 8.0:
                # goal
                str_row += "| # |"
            else:
                str_row += "|%.2f|" % x[actual_idx]

        print(str_row)
        print("------------------------------------------------------------------------------------------")

input_str = ''
np.set_printoptions(suppress=True)


MCTS_planner = MCTS(model)

with torch.no_grad():
    # select a random time step in a random episode. Display its current estimated value, as well as the image.
    episode_idx = np.random.choice(np.arange(len(test_X)))
    step_idx = np.random.choice(np.arange(len(test_X[episode_idx]) - 1))
    x_input = np.reshape(test_X[episode_idx][step_idx], [1, -1])

    initial_value = model.valueModel(torch.from_numpy(x_input).to(device))
    print("Initial estimated value: %.2f" % initial_value.cpu().data.numpy()[0])
    display(x_input[0])

    action_sequence = MCTS_planner.plan(x_input)

    print("best action sequence: ", action_sequence)

