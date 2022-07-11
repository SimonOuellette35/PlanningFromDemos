import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

# In this verison, a priori action probs are generated by the predictor, rather than being static. Also, the predictor
# predicts the value of the done flag, which is used to determine when to stop planning early.
W = 228
H = 228

DEBUG_MODE = False

class MCTS:

    def __init__(self, model, action_dim=3, num_simulations=50):
        self.model = model
        self.N = num_simulations
        self.action_dim = action_dim

    def _pick_action(self, action_priors):

        weights = action_priors.tolist()
        action_range = np.arange(self.action_dim)
        population = action_range.tolist()

        tmp = random.choices(population, weights=weights, k=1)
        return int(tmp[0])

    def plan(self, initial_img, steps=5):

        action_seqs = []
        convictions = []
        values = []
        uncertainties = []

        state = torch.reshape(self.model.encoder(initial_img), [1, 512])

        current_value, done, action_probs = self.model.predict(state)

        if DEBUG_MODE:
            print("current_value = ", current_value)
            self.N = 10

        # 1) stop the steps iteration if done is predicted.
        # 2) self._pick_action must now use action_probs
        # 3) predicting done vastly increases value of sequence.
        for i in range(self.N):
            uncertainty = 0.
            action_seq = []

            next_state = state
            last_value = -np.inf
            for step in range(steps):
                action_probs = F.softmax(action_probs)
                a = self._pick_action(action_probs.cpu().data.numpy()[0])
                action_seq.append(a)

                if done > 0.5:
                    # predicts done = True
                    value = 0
                    break

                next_state, similarity = self.model.memory[a].get_memory(next_state)

                if DEBUG_MODE:
                    print("uncertainty at step %i: %s" % (step, -similarity.cpu().data.numpy()))
                uncertainty -= similarity.cpu().data.numpy()

                value, done, action_probs = self.model.predict(next_state)
                last_value = value.cpu().data.numpy()

            current_conviction = last_value * np.exp(uncertainty)

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