from gym_minigrid.wrappers import *
import torch
import utils
from model.ConvictionPlannerV2 import ConvictionPlanner
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F

device = 'cuda'
ACTION_SPACE = 3
Z_DIM = 512
NC = 3
LOAD_DND = True
CALC_MAE = False

model_path = './saved_models/best_full_modelV3'
data_dir = 'BC_baseline/data/'

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
model = ConvictionPlanner(action_space=ACTION_SPACE, device=device, dict_len=25000)

# Load neural network module
model.load_state_dict(torch.load(model_path))

model.to(device)
model.eval()

# Prime DND
X, a, v = utils.loadMinigridDemonstrationsV2(data_dir, width=W, height=H)

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

# TODO: instead of priming DND, try pickling DND content at end of training and loading that instead?
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


if LOAD_DND:
    print("Loading DND...")

    model.memory = pickle.load(open('./saved_models/dnd.pkl', 'rb'))
else:
    print("Priming DND...")

    with torch.no_grad():
        prime_DND(training_X, training_a)
        print("Num dict keys (0) = ", len(model.memory[0].keys))
        print("Num dict keys (1) = ", len(model.memory[1].keys))
        print("Num dict keys (2) = ", len(model.memory[2].keys))

        print("Pickling the DND...")
        with open('./saved_models/dnd.pkl', 'wb') as pkl_file:
            pickle.dump(model.memory, pkl_file)

input_str = ''

with torch.no_grad():
    # select a random time step in a random episode. Display its current estimated value, as well as the image.
    episode_idx = np.random.choice(np.arange(len(test_X)))
    step_idx = np.random.choice(np.arange(len(test_X[episode_idx]) - 5))

    img = torch.unsqueeze(torch.from_numpy(test_X[episode_idx][step_idx]).to(device), axis=0)
    img_embedding = torch.reshape(model.encoder(img), [img.shape[0], Z_DIM])
    pred_value, pred_done, pred_actions = model.predict(img_embedding)
    pred_actions = F.softmax(pred_actions)
    print("==> Estimated current value: ", pred_value.cpu().data.numpy())
    print("==> Estimated done: ", pred_done.cpu().data.numpy())
    print("==> Estimated action probs: ", pred_actions.cpu().data.numpy())

    plt_img = np.reshape(test_X[episode_idx][step_idx], [W, H, NC])
    plt.imshow(plt_img)
    plt.title("Estimated value: %.4f" % (pred_value.cpu().data.numpy()))
    plt.show()

    while input_str != 'q':

        for idx in range(3):
            img_embedding, similarity = model.memory[idx].get_memory(img_embedding)
            uncertainty = -similarity.cpu().data.numpy()
            print("Action %i, uncertainty: %s" % (
                idx,
                uncertainty
            ))

        # input a sequence of 5 actions from the user
        input_str = input("Enter action: ")

        # call the transition model 5 times to predict the impact of the action sequence, then call the predictor module
        #  to estimate the predicted action sequence value. Display.
        uncertainty = 0.
        uncertainties = []

        idx = int(input_str)
        img_embedding, similarity = model.memory[idx].get_memory(img_embedding)
        uncertainty = -similarity.cpu().data.numpy()

        pred_value, pred_done, pred_actions = model.predict(img_embedding)
        conviction = pred_value * np.exp(uncertainty)
        print("==> Prediction: value: %.4f (uncertainty: %.4f, conviction: %.4f), done: %.4f, actions: %s" % (
            pred_value.cpu().data.numpy(),
            uncertainty,
            conviction,
            pred_done.cpu().data.numpy(),
            pred_actions.cpu().data.numpy()
        ))
        
        print("================================================================================")

