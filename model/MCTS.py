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

    def __init__(self, model, action_priors=[0.33, 0.33, 0.33], action_dim=3, num_simulations=50):
        self.model = model
        self.N = num_simulations
        self.action_dim = action_dim

        self.action_priors = action_priors

    def _pick_action(self):
        tmp = random.choices(np.arange(self.action_dim).tolist(), weights=self.action_priors, k=1)[0]
        return int(tmp)

    # TODO:
    #  1) optimize this by calling valuePredictor only once for all self.N states?
    #  2) parallelize this?
    def plan(self, initial_img, steps=5):

        action_seqs = []
        convictions = []
        values = []
        uncertainties = []

        state = torch.reshape(self.model.encoder(initial_img), [1, 512])

        if DEBUG_MODE:
            current_value = self.model.valuePredictor(state)
            print("current_value = ", current_value)
            self.N = 10

        for i in range(self.N):
            uncertainty = 0.
            action_seq = []

            next_state = state
            for step in range(steps):
                a = self._pick_action()
                action_seq.append(a)

                next_state, similarity = self.model.memory[a].get_memory(next_state)

                if DEBUG_MODE:
                    print("uncertainty at step %i: %s" % (step, -similarity.cpu().data.numpy()))
                uncertainty -= similarity.cpu().data.numpy()

            value = self.model.valuePredictor(next_state)

            current_conviction = value.cpu().data.numpy() * np.exp(uncertainty)

            values.append(value.cpu().data.numpy())
            uncertainties.append(uncertainty)
            convictions.append(current_conviction)
            action_seqs.append(np.array(action_seq))

        if DEBUG_MODE:
            print("action_seqs = ", action_seqs)
            print("convictions = ", convictions)
            print("values = ", values)
            print("uncertainties = ", uncertainties)

            plt.imshow(np.reshape(initial_img.cpu().data.numpy(), [W, H, 3]))
            plt.show()

        # pick best action sequence based on value to uncertainty ratio ("Conviction")
        best_idx = np.argmax(convictions)

        if DEBUG_MODE:
            print("Selected action sequence: ", action_seqs[best_idx])

        return action_seqs[best_idx]