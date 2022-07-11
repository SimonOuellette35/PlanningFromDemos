from DeltaSymbolicPlanner import SymbolicPlanner
#from BaseDNDSymbolicPlanner import SymbolicPlanner
#from BaseSymbolicPlanner import SymbolicPlanner
import numpy as np
import utils
import torch
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

data_dir = 'data/'
model_dir = './saved_models/'
DEVICE = 'cuda'
ACTION_SPACE = 3
BATCH_SIZE = 32

X, a, v = utils.loadMinigridSymbolicDemos(data_dir)

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

Z_DIM = 100
LR = 0.0001
model = SymbolicPlanner(z_dim=Z_DIM, action_space=ACTION_SPACE, device=DEVICE, dict_len=500)

model.train()
model.to(DEVICE)
#optimizer = optim.AdamW(model.parameters(), weight_decay=0.01, lr=LR)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_loss = np.inf

NUM_EPOCHS = 10000

best_loss = np.inf
# TODO: same problem here: must predict done = True? Predict success frame with special representation?
def generateBatch(tmp_X, tmp_a, tmp_v):
  batch_X, batch_nextX, batch_actions = list(), list(), list()

  for _ in range(BATCH_SIZE):
    episode_ok = False
    while not episode_ok:
      episode_idx = np.random.choice(np.arange(len(tmp_X)))
      if len(tmp_X[episode_idx]) > 2:
        episode_ok = True

    step_idx = np.random.choice(np.arange(len(tmp_X[episode_idx])-1))

    batch_X.append(tmp_X[episode_idx][step_idx])
    batch_nextX.append(tmp_X[episode_idx][step_idx+1])
    batch_actions.append(tmp_a[episode_idx][step_idx+1][0])

  return np.array(batch_X), np.array(batch_nextX), np.array(batch_actions)

torch.autograd.set_detect_anomaly(True)
train_losses = []
test_losses = []
for epoch in range(NUM_EPOCHS):

  optimizer.zero_grad()

  batch_X, batch_nextX, batch_actions = generateBatch(training_X, training_a, training_v)

  loss = model.trainBatch(batch_X, batch_nextX, batch_actions)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  train_losses.append(loss.cpu().data.numpy())

  # calculate test set performance to check for overfitting...
  with torch.no_grad():
    batch_X, batch_nextX, batch_actions = generateBatch(test_X, test_a, test_v)

    test_loss = model.trainBatch(batch_X, batch_nextX, batch_actions, eval=True)

  test_losses.append(test_loss.cpu().data.numpy())
  print("Epoch# %s: loss = %s (test_loss = %s)" % (epoch + 1, loss.cpu().data.numpy(), test_loss.cpu().data.numpy()))

  # save best model to file
  if test_loss.cpu().data.numpy() < best_loss:
    best_loss = test_loss.cpu().data.numpy()
    print("--> Saving Best Model...")
    torch.save(model.state_dict(), '%sbest_full_modelV3' % model_dir)

    # print("Pickling the DND...")
    # with open('./saved_models/dnd.pkl', 'wb') as pkl_file:
    #   pickle.dump(model.memory, pkl_file)

plt.plot(train_losses, color='blue')
plt.plot(test_losses, color='red')
plt.show()