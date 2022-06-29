from gym_minigrid.wrappers import *
import torch
import utils
from Ablation_baseline.AblationStudy import AblationModel
from model.ConvictionPlanner import ConvictionPlanner
from model.MCTS import MCTS
import cv2
import numpy as np
from gym_minigrid.window import Window
from DQfD_baseline.models import DQN
from BC_baseline.BCModelV2 import BCModel
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
Z_DIM = 512
NC = 3
DET_SEED = 555

np.random.seed(DET_SEED)

TEST_EPISODE = 701
NUM_EPISODES = 800
SKIP_LEVELS = 700
LOAD_DND = True

model_path = './saved_models/best_full_modelV3'
data_dir = 'BC_baseline/data/'

print("Initializing environment...")

if GRID_DIFFICULTY == 'Easy':
    env = gym.make('MiniGrid-Empty-Random-6x6-v0')
elif GRID_DIFFICULTY == 'Intermediate':
    env = gym.make('MiniGrid-FourRooms-v0')
# elif GRID_DIFFICULTY == 'Hard':
#     env = gym.make('MiniGrid-MultiRoom-N6-v0')

env = RGBImgObsWrapper(env)  # Get pixel observations
env = ImgObsWrapper(env)            # Get rid of the 'mission' field

window = Window('gym_minigrid')

obs = env.reset()                   # This now produces an RGB tensor only
obs = env.render('rgb_array', tile_size=TILE_SIZE)

print("Loading pre-trained model...")

#  load model
if MODEL_NAME=='DQFD':
    print("============= Using Deep Q-Learning from Demonstrations baseline model ===================")
    dtype = torch.cuda.DoubleTensor if device == 'cuda' else torch.DoubleTensor
    model = DQN(dtype, (NC, W, H), ACTION_SPACE)

    model.load_state_dict(pickle.load(open(model_path, 'rb')))
elif MODEL_NAME=='BC':
    print("============= Using Behavioral cloning baseline model ===================")
    # Using a pre-trained Behavioral cloning model
    model = BCModel(action_space=2, device=device)

    model.load_state_dict(torch.load(model_path))
elif MODEL_NAME=='Ablation':
    print("============= Using ablation study (1-step ahead motion planner) ===================")
    model = AblationModel(action_space=ACTION_SPACE, device=device, dict_len=25000)

    # Load neural network module
    model.load_state_dict(torch.load(model_path))
elif MODEL_NAME=='Conviction':
    print("============= Using Conviction planner ===================")
    model = ConvictionPlanner(action_space=ACTION_SPACE, device=device, dict_len=25000)

    model.load_state_dict(torch.load(model_path))

model.to(device)
model.eval()

if MODEL_NAME != 'DQFD' and MODEL_NAME != 'BC':
    print("Priming DND...")

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
    flat_a = []
    for tmp_seq_a in training_a:
        for tmp_a in tmp_seq_a:
            flat_a.append(tmp_a[0])

        step_counter += len(tmp_seq_a)

    print("step_counter = ", step_counter)

    def prime_DND(x, a):
        for a_idx in range(ACTION_SPACE):
            model.memory[a_idx].reset_memory()

        for seq_idx in range(len(x)):
            progress = seq_idx / float(len(x))
            print("Progress = %.4f %%" % (progress * 100.))

            img = torch.from_numpy(np.array(x[seq_idx])).to(device)
            embeddings = torch.reshape(model.encoder(img), [img.shape[0], Z_DIM])

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

    if MODEL_NAME == 'Conviction':
        counter = collections.Counter(flat_a)
        freq_0 = counter[0] / float(len(flat_a))
        freq_1 = counter[1] / float(len(flat_a))
        freq_2 = counter[2] / float(len(flat_a))

        action_priors = [freq_0, freq_1, freq_2]
        print("==> Using MCTS with baseline priors: ", action_priors)
        MCTS_planner = MCTS(model, action_priors)

prev_obs = None
prev_a = 0
def get_actions(obs):
    global prev_obs
    global prev_a

    def deltasToActionSequence(deltas):
        # transform into N steps forward, N steps (right or left): i.e. an L shape movement. 2 components.
        forward_steps = int(round(deltas[0][0]))
        sideways_steps = int(round(deltas[0][1])) # if negative, it means steps to the right, if positive, steps to the left.

        print("forward_steps = ", forward_steps)
        print("sideways_steps = ", sideways_steps)

        # convert to sequence of actions
        action_sequence = []

        if forward_steps == 0 and sideways_steps == 0:
            action_sequence.append(np.random.choice(np.arange(ACTION_SPACE)))
            return action_sequence

        if forward_steps < 0:
            action_sequence.append(1.)
            action_sequence.append(1.)
            for _ in range(abs(forward_steps)):
                action_sequence.append(2.)

        else:
            for _ in range(forward_steps):
                action_sequence.append(2.)

        if sideways_steps > 0:
            action_sequence.append(0)
            for _ in range(abs(sideways_steps)):
                action_sequence.append(2.)
        elif sideways_steps < 0:
            action_sequence.append(1)
            for _ in range(abs(sideways_steps)):
                action_sequence.append(2.)

        return action_sequence

    def actionPlanning(img, a):
        if model.memory[a] is None:
            return torch.zeros((img.shape[0], Z_DIM)).to(device).double(), torch.from_numpy(
                np.ones(img.shape[0]) * np.inf)

        embeddings = model.encoder(img)
        embeddings = torch.reshape(embeddings, [embeddings.shape[0], -1])

        pred_values = []
        uncertainties = []
        for i in range(img.shape[0]):
            emb_i = torch.unsqueeze(embeddings[i], dim=0)

            pred_embedding, similarity = model.memory[a].get_memory(emb_i)

            pred_value = model.valuePredictor(pred_embedding)

            uncertainties.append(-similarity.cpu().data.numpy())
            pred_values.append(pred_value.cpu().data.numpy()[0][0])

        return np.array(pred_values), np.array(uncertainties)

    def uncertaintyCalculation(img):
        # Action 0: left
        value1, incertitude1 = actionPlanning(img, 0)

        # Action 1: right
        value2, incertitude2 = actionPlanning(img, 1)

        # Action 2: forward
        value3, incertitude3 = actionPlanning(img, 2)

        return np.array([value1, value2, value3]), np.array([incertitude1, incertitude2, incertitude3])

    obs = obs / 255.

    img = torch.reshape(torch.unsqueeze(torch.from_numpy(obs), dim=0), [1, 3, W, H]).to(device).double()

    if MODEL_NAME == 'DQFD':
        state_batch = img
        q_vals = model(state_batch)
        print("q_vals = ", q_vals.cpu().data.numpy())
        return np.argmax(q_vals.cpu().data.numpy())
    elif MODEL_NAME == 'BC':
        state_batch = img
        deltas = model(state_batch)

        action_sequence = deltasToActionSequence(deltas.cpu().data.numpy())
        return np.array(action_sequence)
    elif MODEL_NAME == 'Conviction':
        action_sequence = MCTS_planner.plan(img)
        return np.array(action_sequence)

    # TODO: the following code is for the ablation study planning. Re-organize.

    values, incertitudes = uncertaintyCalculation(img)
    incertitudes = np.reshape(incertitudes, [-1])
    values = np.reshape(values, [-1])

    print("==> Values = ", values)
    print("\tIncertitudes = ", incertitudes)

    convictions = values * np.exp(incertitudes)

    print("\tConvictions = ", convictions)

    a = np.argmax(convictions)
    #a = np.argmax(values)
    #a = np.argmin(incertitudes)

    best_conviction = convictions[a]

    # Remap "Toogle" action to 5?
    if a == 3:
        a = 5

    # Temporary workaround for bug where the last frame was not included in training (would take too much time to
    # fix and re-train...)
    if (values > -3).any() and (incertitudes > 1.).all():
        return 2

    # Infinite loop prevention mechanism, part of planning logic
    if prev_obs is not None:
        if (prev_obs == obs).all() and prev_a == a:
            print("Infinite loop prevention kicked in: ")
            other_choices = np.arange(ACTION_SPACE)
            print("\taction space = ", other_choices)
            other_choices = np.delete(other_choices, a)
            print("\taction space (with offending action removed) = ", other_choices)
            a = np.random.choice(other_choices)
            print("\tnew random action choice = ", a)

    prev_obs = obs
    prev_a = a
    return a

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
            fourcc = cv2.VideoWriter_fourcc('p', 'n', 'g', ' ')
            out = cv2.VideoWriter("%s-%s.avi" % ('test_examples', log_done_counter - (TEST_EPISODE-1)), fourcc, 60, (W, H))

        tmp_obs = obs
        actions = get_actions(obs)

        if MODEL_NAME == 'BC' or MODEL_NAME == 'Conviction':  # special case: this returns a sequence of actions, not just 1 action
            for current_a in actions:
                env.render()
                obs, rewards, done, _ = env.step(current_a)
                obs = env.render('rgb_array', tile_size=TILE_SIZE)
                if out is not None:
                    out.write(obs)

                frame_counter += 1

                if done:
                    break
        else:
            env.render()
            obs, rewards, done, _ = env.step(actions)
            obs = env.render('rgb_array', tile_size=TILE_SIZE)
            if out is not None:
                out.write(obs)

            frame_counter += 1

    else:

        obs, rewards, done, _ = env.step(6)

    if done:

        if is_test:
            if frame_counter <= 90:
                test_successes += 1
            frames_per_episode += frame_counter
            total_test_episodes += 1

        log_done_counter += 1
        print("Level %s Done!" % log_done_counter)

        obs = env.reset()
        obs = env.render('rgb_array', tile_size=TILE_SIZE)

        frame_counter = 0

frames_per_episode /= float(total_test_episodes)
R = float(test_successes) / float(total_test_episodes)

print("==> Frames per episode = ", frames_per_episode)
print("==> Success rate = %s (%s failures)" % (R, total_test_episodes - test_successes))

if out is not None:
    out.release()

