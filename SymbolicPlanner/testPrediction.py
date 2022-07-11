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

model_path = './saved_models/best_full_modelV3'
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

with torch.no_grad():
    # select a random time step in a random episode. Display its current estimated value, as well as the image.

    # while input_str != 'q':
    episode_idx = np.random.choice(np.arange(len(test_X)))
    step_idx = np.random.choice(np.arange(len(test_X[episode_idx]) - 1))

    x_input = test_X[episode_idx][step_idx]
    display(x_input)
    #display_raw(x_input)

    # input a sequence of 5 actions from the user
    print("Actual action is: ", test_a[episode_idx][step_idx+1])
    input_str = input("Enter action: ")

    action = int(input_str)

    # call the transition model 5 times to predict the impact of the action sequence, then call the predictor module
    #  to estimate the predicted action sequence value. Display.

    def attention(data, indices):
        attended_cells = []
        attended_indices = []

        for batch_idx in range(data.shape[0]):
            tmp_idx = indices[batch_idx]

            tmp_idx_above = tmp_idx - 1
            tmp_idx_below = tmp_idx + 1
            tmp_idx_right = tmp_idx + 19
            tmp_idx_left = tmp_idx - 19

            cell_list = [0, tmp_idx, tmp_idx_left, tmp_idx_right, tmp_idx_above, tmp_idx_below]
            attended_indices.append(cell_list)
            attended_cells.append(torch.unsqueeze(data[batch_idx][cell_list], dim=0))

        return torch.cat(attended_cells, dim=0).to(device), np.array(attended_indices)

    def inverse_transform(data, indices):
        output = np.zeros([1, 362])
        for idx in range(len(indices)):
            target_idx = indices[idx]

            output[0, target_idx] = data[idx]

        return output

    pointer_idx = np.where(x_input == 10.)[0][0]

    # 1) attention mechanism
    batch_x = torch.from_numpy(np.array([x_input]))

    attended_view, attended_indices = attention(batch_x, [pointer_idx])

    print("attended_view: ", attended_view[0].cpu().data.numpy())

    pred_delta = model.dynamics_models[action](attended_view)

    print("pred_delta: ", pred_delta.cpu().data.numpy()[0])

    total_delta = inverse_transform(pred_delta.cpu().data.numpy()[0], attended_indices[0])
    pred_next = x_input + total_delta

    display(pred_next[0])

    print("================================================================================")

