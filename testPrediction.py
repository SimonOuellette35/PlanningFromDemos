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

if CALC_MAE:
    NUM_TESTS = 100
    BATCH_SIZE = 10
    total_MAE = 0.
    for _ in range(NUM_TESTS):

        batch_X = []
        batch_y = []
        batch_actions = []

        for _ in range(BATCH_SIZE):
            episode_ok = False
            while not episode_ok:
                episode_idx = np.random.choice(np.arange(len(test_X)))
                if len(test_X[episode_idx]) > 6:
                    episode_ok = True

            step_idx = np.random.choice(np.arange(len(test_X[episode_idx]) - 5))

            batch_X.append(test_X[episode_idx][step_idx])
            batch_y.append(test_v[episode_idx][step_idx+5])
            batch_actions.append(test_a[episode_idx][step_idx+1:step_idx+6])

        batch_img = torch.from_numpy(np.array(batch_X)).to(device)
        batch_embedding = torch.reshape(model.encoder(batch_img), [batch_img.shape[0], Z_DIM])
        batch_actions = np.reshape(np.array(batch_actions), [len(batch_actions), 5])
        batch_y = np.array(batch_y)

        # TODO: get_memory doesn't work in batches, need to fix this. In part because get_memory doesn't
        #  return batches, but also because different actions in the batch imply interrogating different dictionaries...
        tmp_embedding = batch_embedding
        for i in range(batch_actions.shape[0]):
            for a in batch_actions[i]:
                tmp_embedding[i], _ = model.memory[a].get_memory(tmp_embedding[i])

        pred_values = model.valuePredictor(tmp_embedding)
        MAE = np.mean(np.abs(batch_y - pred_values.cpu().data.numpy()))
        print("MAE = ", MAE)

        total_MAE += MAE

    total_MAE /= float(NUM_TESTS)
    print("Total MAE: ", total_MAE)

with torch.no_grad():
    # select a random time step in a random episode. Display its current estimated value, as well as the image.
    episode_idx = np.random.choice(np.arange(len(test_X)))
    step_idx = np.random.choice(np.arange(len(test_X[episode_idx]) - 5))

    while input_str != 'q':

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

        # input a sequence of 5 actions from the user
        input_str = input("Enter action sequence: ")

        action_sequence = list(map(int, input_str.split(',')))
        print("Captured action sequence: ", action_sequence)

        # call the transition model 5 times to predict the impact of the action sequence, then call the predictor module
        #  to estimate the predicted action sequence value. Display.
        uncertainty = 0.
        uncertainties = []
        for a in action_sequence:
            img_embedding, similarity = model.memory[a].get_memory(img_embedding)
            uncertainty -= similarity.cpu().data.numpy()
            uncertainties.append(-similarity.cpu().data.numpy())

        pred_5step_value = model.valuePredictor(img_embedding)
        conviction = pred_5step_value * np.exp(uncertainty)
        print("==> Estimated 5-step ahead value: %.4f (uncertainty: %.4f, conviction: %.4f)" % (
              pred_5step_value.cpu().data.numpy(),
              uncertainty,
              conviction
              ))
        print("Uncertainties: ", uncertainties)
        print("================================================================================")

