import torch.nn.functional as F
import torch.nn as nn
import torch

class SymbolicPlanner(nn.Module):

    def __init__(self, x_dim=362, z_dim=100, action_space=3, stop_threshold=0., dict_len=25000, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.action_space = action_space
        self.STOP_THRESHOLD = stop_threshold

        self.dynamics_models = []

        for i in range(action_space):
            module = torch.nn.Sequential(
                    torch.nn.Linear(x_dim, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 2048),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2048, 2048),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2048, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, x_dim)
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

        batch_x = torch.from_numpy(batch_x).to(self.device)
        batch_next = torch.from_numpy(batch_next).to(self.device)

        pred_next = []
        for i in range(batch_a.shape[0]):
            tmp_x = torch.unsqueeze(batch_x[i], dim=0)
            tmp_pred = self.dynamics_models[batch_a[i]](tmp_x)
            pred_next.append(tmp_pred)

        pred_next = torch.cat(pred_next, dim=0)

        # calculate loss
        return F.mse_loss(pred_next, batch_next)
