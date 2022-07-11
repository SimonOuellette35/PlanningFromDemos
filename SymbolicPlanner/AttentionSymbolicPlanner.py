import torch.nn.functional as F
import torch.nn as nn
import torch
from dnd import DND

class SymbolicPlanner(nn.Module):

    def __init__(self, x_dim=362, z_dim=100, action_space=3, stop_threshold=0., dict_len=25000, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.action_space = action_space
        self.STOP_THRESHOLD = stop_threshold

        # encoder from symbolic space (dim = 362) to transition space (Z_DIM)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(x_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 200),
            torch.nn.ReLU(),
            # torch.nn.Linear(200, 150),
            # torch.nn.ReLU(),
            # torch.nn.Linear(150, 125),
            # torch.nn.ReLU(),
            # torch.nn.Linear(125, 125),
            # torch.nn.ReLU(),
            torch.nn.Linear(200, z_dim),
            torch.nn.Sigmoid()
        ).to(device).double()

        self.dynamics_models = []

        for i in range(action_space):
            module = torch.nn.Sequential(
                torch.nn.Linear(z_dim, 150),
                torch.nn.ReLU(),
                torch.nn.Linear(150, 250),
                torch.nn.ReLU(),
                torch.nn.Linear(250, 300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, x_dim)
            ).to(device).double()

            super(SymbolicPlanner, self).add_module("module%i" % i, module)

            self.dynamics_models.append(
                module
            )

    # TODO: normalize batch_X/batch_next?
    # batch_x : [BATCH_SIZE, 362], symbolic space frame representation at time t
    # batch_next: [BATCH_SIZE, 362], symbolic space frame representation at time t+1
    # batch_a: [BATCH_SIZE], action taken to go from batch_x to batch_next
    def trainBatch(self, batch_x, batch_next, batch_a, eval=False):

        # 1) encode batch_X into transition space (bottleneck for planning)
        batch_next = torch.from_numpy(batch_next).to(self.device)
        batch_x = torch.from_numpy(batch_x).to(self.device)
        embeddings = self.encoder(batch_x)

        pred_delta = []
        # 2) given batch_a, fetch predicted transition deltas from DND
        for i in range(embeddings.shape[0]):
            tmp_x = torch.unsqueeze(embeddings[i], dim=0)
            tmp_pred = self.dynamics_models[int(batch_a[i])](tmp_x)
            pred_delta.append(tmp_pred)

        pred_delta = torch.cat(pred_delta, dim=0).to(self.device)

        # 3) apply transition delta to batch_X to get pred_next
        actual_delta = batch_next - batch_x

        # calculate loss
        return F.mse_loss(pred_delta[:, 1:], actual_delta[:, 1:]) + F.mse_loss(pred_delta[:, 0], actual_delta[:, 0])
