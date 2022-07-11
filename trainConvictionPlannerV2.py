from model.ConvictionPlannerV2 import ConvictionPlanner
import numpy as np
import utils
import torch
import torch.optim as optim

data_dir = 'BC_baseline/data/'
model_dir = './saved_models/'
DEVICE = 'cuda'
ACTION_SPACE = 3
BATCH_SIZE = 16

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
model = ConvictionPlanner(action_space=ACTION_SPACE, device=DEVICE, dict_len=25000)

model.train()
model.to(DEVICE)
optimizer = optim.AdamW(model.parameters(), weight_decay=0.01, lr=LR)

best_loss = np.inf

NUM_EPOCHS = 5000

best_loss = np.inf
def generateBatch(tmp_X, tmp_a, tmp_v):
  batch_X, batch_nextX, batch_done, batch_actions, batch_values = list(), list(), list(), list(), list()

  for _ in range(BATCH_SIZE):
    episode_ok = False
    while not episode_ok:
      episode_idx = np.random.choice(np.arange(len(tmp_X)))
      if len(tmp_X[episode_idx]) > 6:
        episode_ok = True

    step_idx = np.random.choice(np.arange(len(tmp_X[episode_idx]) - 5))

    batch_X.append(tmp_X[episode_idx][step_idx])
    for t in range(1, 6):
      batch_nextX.append(tmp_X[episode_idx][step_idx+t])

    for t in range(6):
      batch_values.append(tmp_v[episode_idx][step_idx+t])
      if step_idx + t + 1 >= len(tmp_X[episode_idx]):
        batch_done.append(1.)
      else:
        batch_done.append(0.)

      if (step_idx + t + 1) == len(tmp_X[episode_idx]):
        batch_actions.append(2.)  # The last action of an episode is always "move forward" (towards the green square)
      else:
        batch_actions.append(tmp_a[episode_idx][step_idx + t + 1][0])

  return torch.from_numpy(np.array(batch_X)).to(DEVICE), \
         torch.from_numpy(np.array(batch_nextX)).to(DEVICE), \
         np.array(batch_done), \
         np.array(batch_actions), \
         np.array(batch_values)

torch.autograd.set_detect_anomaly(True)
for epoch in range(NUM_EPOCHS):

  optimizer.zero_grad()

  batch_X, batch_nextX, batch_done, batch_actions, batch_values = generateBatch(training_X, training_a, training_v)

  loss = model.trainSequence(batch_X, batch_nextX, batch_done, batch_actions, batch_values)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # calculate test set performance to check for overfitting...
  with torch.no_grad():
    batch_X, batch_X2, batch_done, batch_actions, batch_values = generateBatch(test_X, test_a, test_v)

    test_loss = model.trainSequence(batch_X, batch_X2, batch_done, batch_actions, batch_values, eval=True)

  print("Epoch# %s: loss = %s (test_loss = %s)" % (epoch + 1, loss.cpu().data.numpy(), test_loss.cpu().data.numpy()))

  # save best model to file
  if test_loss.cpu().data.numpy() < best_loss:
    best_loss = test_loss.cpu().data.numpy()
    print("--> Saving Best Model...")
    torch.save(model.state_dict(), '%sbest_full_modelV3' % model_dir)