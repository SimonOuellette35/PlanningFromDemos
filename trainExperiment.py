from model.PolicyModel import PolicyModel
import numpy as np
import utils
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from os import path
import matplotlib.pyplot as plt
import pickle
import model.dnd as dnd

data_dir = './SupervisedPolicy_baseline/data/'
model_dir = './saved_models/'
DEVICE = 'cuda'
ACTION_SPACE = 3

# la version emptyRoom utilise width=192, height=192
X, a, v = utils.loadMinigridDemonstrationsV2(data_dir, width=228, height=228)

# Note: la version simple "emptyRoom" utilise T = 100
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
for a in training_a:
  step_counter += len(a)

print("step_counter = ", step_counter)

Z_DIM = 512
LR = 0.0001
model = PolicyModel(action_space=ACTION_SPACE, device=DEVICE, dict_len=10000)
model.train()
model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), weight_decay=0.01, lr=LR)

best_loss = np.inf

NUM_EPOCHS = 1000

best_loss = np.inf
BATCH_SIZE = 12
for epoch in range(NUM_EPOCHS):

  optimizer.zero_grad()

  losses = []
  for i in range(BATCH_SIZE):
    episode_idx = np.random.choice(len(training_X))
    # Must be CUDA tensors
    batchX = training_X[episode_idx].to(DEVICE)

    # Must be numpy arrays
    batchA = np.reshape(np.array(training_a[episode_idx]), [-1, 1])
    batchV = np.reshape(np.array(training_v[episode_idx]), [-1, 1])

    loss = model.trainSequence(batchX, batchA, batchV)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.cpu().data.numpy())

  # print("Testing...")
  # # calculate test set performance to check for overfitting...
  # with torch.no_grad():
  #   test_loss = 0.
  #   for i in range(len(test_X)):
  #     # Must be CUDA tensors
  #     batchX = test_X[i].to(DEVICE)
  #
  #     # Must be numpy arrays
  #     batchA = np.reshape(np.array(test_a[i]), [-1, 1])
  #     batchV = np.reshape(np.array(test_v[i]), [-1, 1])
  #
  #     test_loss += model.trainSequence(batchX, batchA, batchV, eval=True)
  #
  #   test_loss /= float(len(test_X))

  #print("Epoch# %s: loss = %s (test_loss = %s)" % (epoch + 1, loss.cpu().data.numpy(), test_loss.cpu().data.numpy()))
  avg_loss = np.mean(losses)
  print("Epoch# %s: loss = %s" % (epoch + 1, avg_loss))

  # save best model to file
  # if test_loss.cpu().data.numpy() < best_loss:
  #   best_loss = test_loss.cpu().data.numpy()
  #
  #   torch.save(model.state_dict(), '%sbest_full_model' % model_dir)

  if avg_loss < best_loss:
    best_loss = avg_loss

    torch.save(model.state_dict(), '%sbest_policy_model' % model_dir)
