import torch.nn.functional as F
import torch.nn as nn
import torch
from dnd import DND
import numpy as np

class SymbolicPlanner(nn.Module):

    def __init__(self, x_dim=362, z_dim=100, action_space=3, stop_threshold=0., dict_len=25000, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.action_space = action_space
        self.STOP_THRESHOLD = stop_threshold

        # TODO: the reason this doesn't work is that it can't become a positionally invariant representation. It
        #  can't be that because when it reconstructs the delta to apply to the original image, it would no longer
        #  have the required position information.
        # TODO: solution? What if representation was somehow relative to the pointer itself? Even better if it
        #  automatically learns to represent it like that...
        # encoder from symbolic space (dim = 362) to transition space (Z_DIM)
        # self.encoder = torch.nn.Sequential(
        #     torch.nn.Linear(x_dim, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 200),
        #     torch.nn.ReLU(),
        #     # torch.nn.Linear(200, 150),
        #     # torch.nn.ReLU(),
        #     # torch.nn.Linear(150, 125),
        #     # torch.nn.ReLU(),
        #     # torch.nn.Linear(125, 125),
        #     # torch.nn.ReLU(),
        #     torch.nn.Linear(200, z_dim),
        #     torch.nn.Sigmoid()
        # ).to(device).double()

        self.dynamics_models = []

        for i in range(action_space):
            module = torch.nn.Sequential(
                torch.nn.Linear(6, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 400),
                torch.nn.ReLU(),
                torch.nn.Linear(400, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 6)
            ).to(device).double()

            super(SymbolicPlanner, self).add_module("module%i" % i, module)

            self.dynamics_models.append(
                module
            )

    # TODO: normalize batch_X/batch_next?
    # batch_x : [BATCH_SIZE, 362], symbolic space frame representation at time t
    # batch_next: [BATCH_SIZE, 362], symbolic space frame representation at time t+1
    # batch_a: [BATCH_SIZE], action taken to go from batch_x to batch_next

    # TODO: other possible cheat: assign model based on relative direction instead of absolute? This way direction of
    #  forward motion doesn't have to be conditional on current direction?
    # def trainBatch(self, batch_x, batch_next, batch_a, eval=False):
    #
    #     pointer_indices = []
    #     for i in range(batch_x.shape[0]):
    #         tmp_idx = batch_x[i].index(10.)
    #         pointer_indices.append(tmp_idx)
    #
    #     # 1) encode batch_X into transition space (bottleneck for planning)
    #     batch_next = torch.from_numpy(batch_next).to(self.device)
    #     batch_x = torch.from_numpy(batch_x).to(self.device)
    #     embeddings = self.encoder(batch_x)
    #
    #     pred_delta = []
    #     # 2) given batch_a, fetch predicted transition deltas from DND
    #     for i in range(embeddings.shape[0]):
    #         tmp_x = torch.unsqueeze(embeddings[i], dim=0)
    #         tmp_pred = self.dynamics_models[int(batch_a[i])](tmp_x)
    #         pred_delta.append(tmp_pred)
    #
    #     pred_delta = torch.cat(pred_delta, dim=0).to(self.device)
    #
    #     # 3) apply transition delta to batch_X to get pred_next
    #     actual_delta = batch_next - batch_x
    #
    #     # calculate loss
    #     return F.mse_loss(pred_delta[:, 1:], actual_delta[:, 1:]) + F.mse_loss(pred_delta[:, 0], actual_delta[:, 0])

    # TODO: try hard-coded attention mechanism (problem-specific, but tests the idea)
    # TODO: the problem with the kind of attention mechanism I'm looking for is that it's position-invariant
    #  (perhaps even, ideally, rotation-invariant). However, how do you "stitch back" the delta onto the original
    #  frame to get the full next frame prediction if you lost position and/or rotation information? NOTE: the solution
    #  to this problem should be generic, i.e. not only work on minigrid...
    # TODO: Solution 1: memorize the indices from which the attended view was retrieved, so that it can be remapped
    #  appropriately at the end. How would this solve rotation invariance?
    #  conditionally calculated homography matrix applied to original matrix produced the attended view (includes
    #  operations like rotation, masking, repositioning). The inverse operation must be available for retranslation
    #  into original coordinates.
    def trainBatch(self, batch_x, batch_next, batch_a, eval=False):

        def attention(data, indices):
            attended_cells = []
            attended_indices = []

            for batch_idx in range(data.shape[0]):
                tmp_idx = indices[batch_idx]

                tmp_idx_above = tmp_idx - 1
                tmp_idx_below = tmp_idx + 1
                tmp_idx_right = tmp_idx + 19
                tmp_idx_left = tmp_idx - 19

                cell_list = [0, tmp_idx, tmp_idx_left, tmp_idx_right, tmp_idx_above, tmp_idx_below]
                attended_indices.append(cell_list)
                attended_cells.append(torch.unsqueeze(data[batch_idx][cell_list], dim=0))

            return torch.cat(attended_cells, dim=0).to(self.device), np.array(attended_indices)

        # def inverse_transform(data, indices):
        #     output = np.zeros([1, 362])
        #
        #     for idx in range(len(indices)):
        #         target_idx = indices[idx]
        #
        #         output[0, target_idx] = data[idx]
        #
        #     return torch.from_numpy(output)

        pointer_indices = []
        for i in range(batch_x.shape[0]):
            tmp_idx = np.where(batch_x[i] == 10.)[0][0]
            pointer_indices.append(tmp_idx)

        # 1) attention mechanism
        batch_x = torch.from_numpy(batch_x)
        batch_next = torch.from_numpy(batch_next)

        attended_view, attended_indices = attention(batch_x, pointer_indices)
        attended_next, attended_next_indices = attention(batch_next, pointer_indices)

        #print("attended_view.shape - ", attended_view.shape)

        # 2) run attended_view through transition model, conditional on action. Get predicted transition.
        batch_preds = []
        for i in range(batch_a.shape[0]):
            pred_transition = self.dynamics_models[batch_a[i]](torch.unsqueeze(attended_view[i], dim=0))

            # 3) re-map predicted transform into original space using inverse attention.
            #batch_preds.append(inverse_transform(pred_transition, attended_indices[i]))
            batch_preds.append(pred_transition)

        batch_preds = torch.cat(batch_preds, dim=0)
        actual_deltas = attended_next - attended_view

        #print("batch_preds shape = %s, actual_deltas shape = %s" % (batch_preds.shape, actual_deltas.shape))

        # TODO: no gradient problem here. Why?
        return F.mse_loss(batch_preds, actual_deltas)


