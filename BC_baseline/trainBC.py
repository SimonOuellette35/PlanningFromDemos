from BC_baseline.BCModel import BCModel
import numpy as np
import utils
import torch
import torch.optim as optim

# This version predicts 1-step ahead at a time

data_dir = 'data/'
model_dir = '../saved_models/'
DEVICE = 'cuda'
ACTION_SPACE = 3

# la version emptyRoom utilise width=192, height=192
X, a, v = utils.loadMinigridDemonstrationsV2(data_dir, width=228, height=228)

step_counter = 0
for tmp_a in a:
  step_counter += len(tmp_a)

print("step_counter = ", step_counter)

Z_DIM = 512
LR = 0.0001
model = BCModel(action_space=ACTION_SPACE, device=DEVICE)
model.train()
model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), weight_decay=0.01, lr=LR)

best_loss = np.inf

NUM_EPOCHS = 1000

best_loss = np.inf
best_accuracy = 0.
BATCH_SIZE = 12

# TODO: does it have to do with the fact that training samples aren't taken randomly, instead taken from a sequence?
for epoch in range(NUM_EPOCHS):

  losses = []
  accuracies1 = []
  for i in range(BATCH_SIZE):
    optimizer.zero_grad()

    episode_idx = np.random.choice(len(X))

    # Must be CUDA tensors
    batchX = torch.from_numpy(np.array(X[episode_idx])).to(DEVICE)

    # Must be numpy arrays
    batchA = np.reshape(np.array(a[episode_idx]), [-1, 1])
    batchV = np.reshape(np.array(v[episode_idx]), [-1, 1])

    loss, _ = model.trainSequence(batchX, batchA, batchV)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #accuracies.append(accuracy)
    losses.append(loss.cpu().data.numpy())

    # accuracy on exact same batch as training
    with torch.no_grad():
      outputs = model(batchX)
      selected_actions = np.argmax(outputs.cpu().data.numpy(), axis=-1)
      accuracy1 = 0.
      for j in range(batchX.shape[0]):
        if selected_actions[j] == batchA[j][0]:
          accuracy1 += 1.

      accuracy1 /= float(batchX.shape[0])
      accuracies1.append(accuracy1)

  # Now check accuracy on randomly sampled batch
  batch_X = []
  batch_a = []
  for j in range(100):
      episode_idx = np.random.choice(np.arange(len(X)))
      step_idx = np.random.choice(np.arange(len(X[episode_idx])-1))

      batch_X.append(X[episode_idx][step_idx])
      batch_a.append(a[episode_idx][step_idx+1][0])

  with torch.no_grad():
    batch_X = torch.from_numpy(np.array(batch_X)).to(DEVICE)
    outputs = model(batch_X)

    output_actions = np.argmax(outputs.cpu().data.numpy(), axis=-1)

  accuracy = 0.
  for j in range(100):

      if output_actions[j] != 2:
        print("Output = %i, ground truth = %i" % (output_actions[j], batch_a[j]))

      if output_actions[j] == batch_a[j]:
        accuracy += 1.

  accuracy /= 100.

  avg_loss = np.mean(losses)
  print("Epoch# %s: loss = %s, accuracy on same batch = %s, accuracy on other batch= %s" % (epoch + 1,
                                                                                            avg_loss,
                                                                                            np.mean(accuracies1),
                                                                                            accuracy))

  if avg_loss < best_loss and accuracy >= best_accuracy:
    best_accuracy = accuracy
    best_loss = avg_loss
    print("--> saving new best model.")
    torch.save(model.state_dict(), '%sbest_policy_model' % model_dir)
