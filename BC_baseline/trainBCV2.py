from BC_baseline.BCModelV2 import BCModel
import numpy as np
import utils
import torch
import torch.optim as optim

# This version predicts 5 steps ahead

data_dir = 'data/'
model_dir = '../saved_models/'
DEVICE = 'cuda'
ACTION_SPACE = 2
W = 228
H = 228
NC = 3

# la version emptyRoom utilise width=192, height=192
X, a, v = utils.loadMinigridDemonstrationsV2(data_dir, width=228, height=228)

step_counter = 0
for tmp_a in a:
  step_counter += len(tmp_a)

print("step_counter = ", step_counter)

Z_DIM = 512
LR = 0.0005
model = BCModel(action_space=ACTION_SPACE, device=DEVICE, z_dim=Z_DIM)
model.train()
model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), weight_decay=0.01, lr=LR)

best_loss = np.inf

NUM_EPOCHS = 10000

best_loss = np.inf
best_accuracy = 0.
BATCH_SIZE = 32

def calculateTarget(actions):
  deltaX = 0.
  deltaY = 0.

  orientation = 0

  for a in actions:

    if a == 0:  # left
      orientation += np.pi/2.
    elif a == 1:  # right
      orientation -= np.pi/2.
    elif a == 2:  # forward
      deltaX += np.cos(orientation)
      deltaY += np.sin(orientation)

  return [deltaX, deltaY]

def do_iteration():
    batchX = []
    batchY = []
    for i in range(BATCH_SIZE):
        # episode_ok = False
        # while not episode_ok:
        #   episode_idx = np.random.choice(np.arange(len(X)))
        #   if len(X[episode_idx]) > 8:
        #     episode_ok = True

        episode_idx = np.random.choice(np.arange(len(X)))
        step_idx = np.random.choice(np.arange(len(X[episode_idx]) - 1))

        batchX.append(np.array(X[episode_idx][step_idx]))

        end_idx = min(step_idx + 6, len(X[episode_idx]))
        tmp_actions = list(a[episode_idx][step_idx + 1:end_idx])
        if end_idx == len(X[episode_idx]):
            tmp_actions.append([2.])

        batchY.append(calculateTarget(tmp_actions))

    batchX = torch.from_numpy(np.array(batchX)).to(DEVICE)
    batchY = np.array(batchY)

    loss = model.trainSequence(batchX, batchY)

    return loss

for epoch in range(NUM_EPOCHS):

  optimizer.zero_grad()

  loss = do_iteration()

  loss.backward()
  optimizer.step()

  print("Epoch# %s: loss = %s" % (epoch + 1, loss.cpu().data.numpy()))

  if loss.cpu().data.numpy() < best_loss:
    best_loss = loss.cpu().data.numpy()
    print("--> saving new best model.")
    torch.save(model.state_dict(), '%sbest_policy_model' % model_dir)

print("==> Running tests... (using last model, not best)")
NUM_TESTS = 25
# accuracy = 0.
for i in range(NUM_TESTS):
    print("test %i" % i)

    with torch.no_grad():
        loss = do_iteration()
        print("loss = ", loss)

    # fig, axarr = plt.subplots(3, 3)
    # for idx_i in range(3):
    #     for idx_j in range(3):
    #         idx = (idx_i * 3) + idx_j
    #         img_x = batch_X[idx]
    #         outp = outputs[idx].cpu().data.numpy()
    #
    #         axarr[idx_i][idx_j].imshow(np.reshape(img_x, [W, H, NC]))
    #         print("%i, %i ==> %s (label: %s)" % (idx_i, idx_j, outp, batch_a[idx]))
    #
    # plt.show()