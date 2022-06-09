import time
from datetime import date
import pickle
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import loadMinigridDemonstrationsV2
from models import DQN

# GPU support
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.DoubleTensor if USE_CUDA else torch.DoubleTensor

def optimize_dqfd(bsz, data, device='cuda'):

    # Shape of X: [num episodes, num steps per episode, NC, W, H]
    # Shape of a: [num episodes, num steps per episode]
    # Shape of v: [num episodes, num steps per episode]
    X, a, v = data

    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []

    # TODO: training vs validation set

    # re-structure X, a, v to fit the rest of the code, randomly sample batches of transitions
    for i in range(bsz):
        episode_idx = np.random.choice(np.arange(len(X)))
        step_idx = np.random.choice(np.arange(len(X[episode_idx])))

        state = X[episode_idx][step_idx][0]
        state_batch.append(state)
        action_batch.append(a[episode_idx][step_idx])
        reward_batch.append(float(v[episode_idx][step_idx]))
        if step_idx < (len(X[episode_idx]) - 1):
            next_state = X[episode_idx][step_idx+1][0]
            next_state_batch.append(next_state)
        else:
            next_state_batch.append(state)

    state_batch = torch.from_numpy(np.array(state_batch)).to(device)
    action_batch = torch.from_numpy(np.array(action_batch)).to(device)
    reward_batch = torch.reshape(torch.from_numpy(np.ones(action_batch.shape)).to(device), [action_batch.shape[0], 1])
    next_state_batch = torch.from_numpy(np.array(next_state_batch)).to(device)

    # TODO: does the n-step return approach make sense for the way we attribute rewards to demo data?
    #n_reward_batch = Variable(torch.cat(batch.n_reward))

    q_vals = model(state_batch)
    # print("qvals = ", q_vals)
    # print("action_batch = ", action_batch)

    state_action_values = q_vals.gather(1, action_batch)
    #print("state_action_values = ", state_action_values)

    # comparing the q values to the values expected using the next states and reward
    #next_q_vals = model(next_state_batch)
    #print("next_q_vals = ", next_q_vals)
    #next_state_values = next_q_vals.data.max(-1).values
    #print("next_state_values = ", next_state_values)
    #print("reward_batch shape = ", reward_batch.shape)
    #expected_state_action_values = next_state_values.unsqueeze(-1) + reward_batch

    # calculating the q loss and n-step return loss
    #print("state_action_values shape = ", state_action_values.shape)
    #print("expected_state_action_values shape = ", expected_state_action_values.shape)
    #q_loss = F.mse_loss(state_action_values, expected_state_action_values, reduction='sum')
    #print("reward_batch = ", reward_batch)
    reward_batch = reward_batch.unsqueeze(-1)
    q_loss = F.mse_loss(state_action_values, reward_batch, reduction='sum')
    #n_step_loss = F.mse_loss(state_action_values, n_reward_batch, size_average=False)

    # calculating the supervised loss
    num_actions = q_vals.size(1)
    margins = (torch.ones(num_actions, num_actions) - torch.eye(num_actions)) * MARGIN
    batch_margins = margins[action_batch.data.squeeze().cpu()]
    q_vals = q_vals + Variable(batch_margins).type(dtype)
    max_qvals = q_vals.max(1)[0].unsqueeze(1)
    supervised_loss = (max_qvals - state_action_values).pow(2).sum()

    loss = q_loss + LAMBDA * supervised_loss #+ args.lam_nstep * n_step_loss

    # optimization step and logging
    optimizer.zero_grad()
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    optimizer.step()

    return loss.cpu().data.numpy()

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
W = 228
H = 228
NUM_ITERATIONS = 1000
MARGIN = .5
BATCH_SIZE = 64
LR = 0.0005
LAMBDA = 1.0

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

    # instantiating model and optimizer
    model = DQN(dtype, (NC, W, H), ACTION_SPACE)
    if args.load_name is not None:
        model.load_state_dict(pickle.load(open(args.load_name, 'rb')))

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # pre-training
    print('Pre-training')

    for i in range(NUM_ITERATIONS):
        loss = optimize_dqfd(BATCH_SIZE, (X, a, v))

        # saving the model every 100 episodes
        if i % 100 == 0:
            pickle.dump(model.state_dict(), open("model" + '.p', 'wb'))

        print("Iteration #%i: loss = %s" % (i, loss))

    print('Pre-training done')
