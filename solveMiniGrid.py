from gym_minigrid.wrappers import *
import torch
import utils
from model.MotionPlannerV2 import MotionPlanner
import cv2
import numpy as np
from gym_minigrid.window import Window
from DQfD_baseline.models import DQN
from SupervisedPolicy_baseline.models import PolicyClassifier
import pickle

#MODEL_NAME = 'DQFD'

# 'BC' : behavioural cloning model
# 'DQFD' : Deep Q-learning from demonsrations
# anything else: our proposed solution
MODEL_NAME = 'DQFD'

device = 'cuda'
ACTION_SPACE = 3
Z_DIM=512
NC = 3
W = 192
H = 192
#W = 228
#H = 228
TILE_SIZE = 32
TEST_EPISODE = 111
NUM_EPISODES = 200
SKIP_LEVELS = 111

model_path = './DQfD_baseline/model.p'
data_dir = 'data/'

print("Initializing environment...")

#env = gym.make('MiniGrid-MultiRoom-N6-v0')
env = gym.make('MiniGrid-Empty-Random-6x6-v0')
#env = gym.make('MiniGrid-FourRooms-v0')
env = RGBImgObsWrapper(env)  # Get pixel observations
env = ImgObsWrapper(env)            # Get rid of the 'mission' field

window = Window('gym_minigrid')

obs = env.reset()                   # This now produces an RGB tensor only
obs = env.render('rgb_array', tile_size=TILE_SIZE)

print("Loading pre-trained model...")

#  load model
if MODEL_NAME=='DQFD':
    # Using a pre-trained "Deep Q-Learning from Demonstrations" model
    dtype = torch.cuda.DoubleTensor if device == 'cuda' else torch.DoubleTensor
    model = DQN(dtype, (NC, W, H), ACTION_SPACE)

    model.load_state_dict(pickle.load(open(model_path, 'rb')))
elif MODEL_NAME=='BC':
    # Using a pre-trained Behavioral cloning model
    dtype = torch.cuda.DoubleTensor if device == 'cuda' else torch.DoubleTensor
    model = PolicyClassifier(dtype, (NC, W, H), ACTION_SPACE)

    model.load_state_dict(pickle.load(open(model_path, 'rb')))
else:
    # TODO: give my class a more generic/relevant name
    model = MotionPlanner(action_space=ACTION_SPACE, device=device, dict_len=25000)

    # Load neural network module
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
    for tmp_a in training_a:
        step_counter += len(tmp_a)

    print("step_counter = ", step_counter)

    def prime_DND(x, a):
        for a_idx in range(ACTION_SPACE):
            model.memory[a_idx].reset_memory()

        for seq_idx in range(len(x)):
            embeddings = torch.reshape(model.encoder(x[seq_idx].to(device)), [x[seq_idx].shape[0], Z_DIM])

            actions = np.reshape(a[seq_idx], [-1]).astype(int)
            for i in range(embeddings.shape[0] - 1):
                model.memory[actions[i + 1]].save_memory(embeddings[i], embeddings[i + 1])

    with torch.no_grad():
        prime_DND(training_X, training_a)
        print("Num dict keys (0) = ", len(model.memory[0].keys))
        print("Num dict keys (1) = ", len(model.memory[1].keys))
        print("Num dict keys (2) = ", len(model.memory[2].keys))

prev_obs = None
prev_a = 0
def get_actions(obs):
    global prev_obs
    global prev_a

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
        logits = model(state_batch)
        print("action logits = ", logits.cpu().data.numpy())
        return np.argmax(logits.cpu().data.numpy())

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

