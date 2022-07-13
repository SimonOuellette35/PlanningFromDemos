import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# In this implementation, the action prior probabilities are not conditioned on the state. They are simply the
# baseline frequency of each action throughout the demonstration dataset.

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
        convictions = []

        data_row = np.reshape(data_row, [1, -1])

        for i in range(self.N):
            action_seq = []

            uncertainty = 0.
            done_uncertainty = 0.
            next_state = data_row
            is_done = 0.
            for step in range(steps):
                a = self._pick_action()

                next_state, uncertainties, done_preds = self.model.predict(next_state, np.array([a]))

                if done_preds[0] > 0.5:
                    done_uncertainty = uncertainties[0]
                    is_done = 10.
                    action_seq.append(2)
                    break
                else:
                    uncertainty += uncertainties[0]
                    action_seq.append(a)

            #print("==> Tested action sequence: ", action_seq)

            # TODO: how to best combine done prediction with uncertainty?
            final_state = torch.from_numpy(next_state).to(self.device)
            value = self.model.valueModel(final_state)

            conviction = (value.cpu().data.numpy()) * np.exp(uncertainty) + (is_done / np.exp(done_uncertainty))

            #print("\tis_done: %i, estimated value: %.2f, uncertainty: %.2f, conviction: %.2f" % (is_done, value, uncertainty, conviction))

            convictions.append(conviction)
            action_seqs.append(np.array(action_seq))

        # pick best action sequence based on value to uncertainty ratio ("Conviction")
        best_idx = np.argmax(convictions)
        # print("best_idx = ", best_idx)
        # print("Selected action sequence: ", action_seqs[best_idx])

        return action_seqs[best_idx]