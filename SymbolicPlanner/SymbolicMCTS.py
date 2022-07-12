import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# In this implementation, the action prior probabilities are not conditioned on the state. They are simply the
# baseline frequency of each action throughout the demonstration dataset.
W = 228
H = 228

DEBUG_MODE = False

class MCTS:

    def __init__(self, model, z_dim=512, action_priors=[0.15, 0.15, 0.70], action_dim=3, num_simulations=50, device='cuda'):
        self.model = model
        self.N = num_simulations
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.action_priors = action_priors
        self.device = device

    def _pick_action(self):
        tmp = random.choices(np.arange(self.action_dim).tolist(), weights=self.action_priors, k=1)[0]
        return int(tmp)

    # TODO:
    #  1) optimize this by calling valuePredictor only once for all self.N states?
    #  2) parallelize this?
    def plan(self, data_row, steps=5):

        action_seqs = []
        values = []

        data_row = np.reshape(data_row, [1, -1])

        for i in range(self.N):
            action_seq = []

            next_state = data_row
            for step in range(steps):
                a = self._pick_action()
                action_seq.append(a)

                next_state, _ = self.model.predict(next_state, np.array([a]))

            final_state = torch.from_numpy(next_state).to(self.device)
            value = self.model.valueModel(final_state)

            values.append(value.cpu().data.numpy())
            action_seqs.append(np.array(action_seq))

        # pick best action sequence based on value to uncertainty ratio ("Conviction")
        best_idx = np.argmax(values)
        # print("best_idx = ", best_idx)
        # print("Selected action sequence: ", action_seqs[best_idx])

        return action_seqs[best_idx]