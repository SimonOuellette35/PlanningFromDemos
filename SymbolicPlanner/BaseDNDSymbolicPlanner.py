import torch.nn.functional as F
import torch.nn as nn
import torch
from model.dnd import DND

class SymbolicPlanner(nn.Module):

    def __init__(self, x_dim=362, z_dim=100, action_space=3, stop_threshold=0., dict_len=25000, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.action_space = action_space
        self.STOP_THRESHOLD = stop_threshold

        self.DICT_LEN = dict_len
        self.memory = []

        for i in range(action_space):
            module = DND(self.DICT_LEN, z_dim)
            super(SymbolicPlanner, self).add_module("module%i" % i, module)

            self.memory.append(
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
            tmp_pred = self.memory[batch_a[i]].get_memory(tmp_x)
            pred_next.append(tmp_pred)

        pred_next = torch.cat(pred_next, dim=0)

        if not eval:
            for i in range(batch_a.shape[0]):
                tmp_x = torch.unsqueeze(batch_x[i], dim=0)
                tmp_next = torch.unsqueeze(batch_next[i], dim=0)
                self.memory[batch_a[i]].save_memory(tmp_x, tmp_next)

        # calculate loss
        return F.mse_loss(pred_next, batch_next)
