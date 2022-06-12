import time
from datetime import date
import pickle
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

from utils import loadMinigridDemonstrationsV2
from models import PolicyClassifier

# GPU support
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.DoubleTensor if USE_CUDA else torch.DoubleTensor

loss_fn = torch.nn.CrossEntropyLoss()
device = 'cuda'

def optimize(bsz, data):

    # Shape of X: [num episodes, num steps per episode, NC, W, H]
    # Shape of a: [num episodes, num steps per episode]
    # Shape of v: [num episodes, num steps per episode]
    X, a, v = data

    train_state_batch = []
    train_action_batch = []

    test_state_batch = []
    test_action_batch = []

    train_episodes = int(len(X) * 0.8)

    # randomly sample batches of training transitions
    for i in range(bsz):
        episode_idx = np.random.choice(np.arange(train_episodes))
        step_idx = np.random.choice(np.arange(len(X[episode_idx])-1))

        state = X[episode_idx][step_idx][0]
        train_state_batch.append(state)
        train_action_batch.append(a[episode_idx][step_idx+1])     # We care about the action taken after seeing this state

    # randomly sample batches of test transitions
    for i in range(200):
        episode_idx = np.random.choice(np.arange((len(X) - train_episodes)))
        step_idx = np.random.choice(np.arange(len(X[train_episodes + episode_idx])-1))

        state = X[train_episodes + episode_idx][step_idx][0]
        test_state_batch.append(state)
        test_action_batch.append(a[train_episodes + episode_idx][step_idx+1])     # We care about the action taken after seeing this state

    train_state_batch = torch.from_numpy(np.array(train_state_batch)).to(device)
    train_action_batch = torch.from_numpy(np.array(train_action_batch)).to(device)
    test_state_batch = torch.from_numpy(np.array(test_state_batch)).to(device)
    test_action_batch = torch.from_numpy(np.array(test_action_batch)).to(device)

    outputs = model(train_state_batch)
    train_action_batch = torch.reshape(train_action_batch, [-1])
    loss = loss_fn(outputs, train_action_batch)

    # optimization step and logging
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    optimizer.step()

    # calculate test accuracy
    with torch.no_grad():
        test_q_vals = model(test_state_batch)
        test_actions = np.argmax(test_q_vals.cpu().data.numpy(), axis=-1)

        accuracy = 0.
        for i in range(test_actions.shape[0]):
            if test_actions[i] == test_action_batch[i]:
                accuracy += 1.

        accuracy /= float(test_actions.shape[0])

    return loss.cpu().data.numpy(), accuracy

parser = argparse.ArgumentParser(description='Minigrid DQfD')

# nn optimization hyperparams
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--bsz', type=int, default=32, metavar='BSZ',
                    help='batch size (default: 32)')

# model saving and loading settings
parser.add_argument('--save-name', default='doom_dqn_model', metavar='FN',
                    help='path/prefix for the filename to save model\'s parameters')
parser.add_argument('--load-name', default=None, metavar='LN',
                    help='path/prefix for the filename to load model\'s parameters')

# RL training hyperparams
parser.add_argument('--env-name', default='DoomBasic-v0', metavar='ENV',
                    help='environment to train on (default: DoomBasic-v0')
parser.add_argument('--num-eps', type=int, default=-1, metavar='NE',
                    help='number of episodes to train (default: train forever)')
parser.add_argument('--frame-skip', type=int, default=4, metavar='FS',
                    help='number of frames to skip between agent input (must match frame skip for demos)')
parser.add_argument('--init-states', type=int, default=1000, metavar='IS',
                    help='number of states to store in memory before training (default: 1000)')
parser.add_argument('--gamma', type=float, default=1000, metavar='GAM',
                    help='reward discount per step (default: 0.99)')

# policy hyperparams
parser.add_argument('--eps-start', type=int, default=1.0, metavar='EST',
                    help='starting value for epsilon')
parser.add_argument('--eps-end', type=int, default=0.0, metavar='EEND',
                    help='ending value for epsilon')
parser.add_argument('--eps-steps', type=int, default=10000, metavar='ES',
                    help='number of episodes before epsilon equals eps-end (linearly degrades)')

# DQfD hyperparams
parser.add_argument('--demo-prop', type=float, default=0.3, metavar='DR',
                    help='proportion of batch to set as transitions from the demo file')
parser.add_argument('--demo-file', default=None, metavar='DF',
                    help='file to load pickled demonstrations')
parser.add_argument('--margin', type=float, metavar='MG',
                    help='margin for supervised loss used in DQfD (must be set)')
parser.add_argument('--lam-sup', type=float, default=1.0, metavar='LS',
                    help='weight of the supervised loss (default 1.0)')
parser.add_argument('--lam-nstep', type=float, default=1.0, metavar='LN',
                    help='weight of the n-step loss (default 1.0)')

# testing/monitoring settings
parser.add_argument('--no-train', action="store_true",
                    help='set to true if you don\'t want to actually train')
parser.add_argument('--monitor', action="store_true",
                    help='whether to monitor results')
parser.add_argument('--upload', action="store_true",
                    help='set this (and --monitor) if you want to upload monitored ' \
                        'results to gym (requires api key in an api_key.json)')


ACTION_SPACE = 3
NC = 3
# W = 228
# H = 228
W = 192
H = 192
NUM_ITERATIONS = 250
BATCH_SIZE = 64
LR = 0.001

NUM_TESTS = 500

if __name__ == '__main__':
    args = parser.parse_args()
    
    # setting the run name based on the save name set for the model and the timestamp
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    save_name = args.save_name + '_' + timestring
    if args.load_name is None:
        run_name = save_name.split('/')[-1]
    else:
        run_name = args.load_name.split('/')[-1]

    X, a, v = loadMinigridDemonstrationsV2('./data/', width=W, height=H)

    test_X = []
    test_a = []
    for i in range(NUM_TESTS):
        episode_idx = np.random.choice(np.arange(len(X)))
        step_idx = np.random.choice(np.arange(len(X[episode_idx])-1))

        test_X.append(X[episode_idx][step_idx])
        test_a.append(a[episode_idx][step_idx+1])

    # instantiating model and optimizer
    model = PolicyClassifier(dtype, (NC, W, H), ACTION_SPACE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # pre-training
    print('Pre-training')

    losses = []
    best_loss = 10000.
    best_accuracy = 0.
    for i in range(NUM_ITERATIONS):
        loss, accuracy = optimize(BATCH_SIZE, (X, a, v))

        print("Iteration #%i: loss = %.4f, test accuracy : %.2f" % (i, loss, accuracy))
        if accuracy >= best_accuracy and loss < best_loss:
            print("--> saving best new model.")
            pickle.dump(model.state_dict(), open("model" + '.p', 'wb'))
            best_accuracy = accuracy
            best_loss = loss

        losses.append(loss)


    print('Pre-training done')
    plt.plot(losses)
    plt.show()

    print('Starting in-sample evaluation of learning.')
    accuracy = 0.
    for i in range(NUM_TESTS):
        state_batch = torch.from_numpy(np.array(test_X[i])).to(device)
        action_logits = model(state_batch)
        selected_action = np.argmax(action_logits.cpu().data.numpy())

        print("Selection action: %s, action logits: %s, ground truth: %s" % (
            selected_action, action_logits.cpu().data.numpy()[0], test_a[i][0]
        ))

        if selected_action == test_a[i][0]:
            accuracy += 1.

        # plt.imshow(np.reshape(test_X[i], [W, H, NC]))
        # plt.show()

    accuracy /= float(NUM_TESTS)
    print("Action accuracy = %.2f %%" % (accuracy * 100.))
