import torch.nn.functional as F
import torch.nn as nn
import torch
from dnd import DND
import numpy as np
import utils

class SymbolicPlanner(nn.Module):

    def __init__(self, x_dim=362, z_dim=6, action_space=3, stop_threshold=0., dict_len=500, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.action_space = action_space
        self.STOP_THRESHOLD = stop_threshold

        # DND-based transition model
        self.DICT_LEN = dict_len
        self.memory = []
        for i in range(action_space):
            module = DND(self.DICT_LEN, 6+1)    # +1 for the done flag prediction
            self.memory.append(module)

        # self.valueModel = torch.nn.Sequential(
        #         torch.nn.Linear(x_dim, 50000),
        #         torch.nn.ReLU(),
        #         torch.nn.Linear(50000, 1)
        # ).to(device).double()

        self.valueModel = torch.nn.Sequential(
                torch.nn.Linear(x_dim, 10000),
                torch.nn.ReLU(),
                torch.nn.Linear(10000, 10000),
                torch.nn.ReLU(),
                torch.nn.Linear(10000, 1),
        ).to(device).double()


    def attention(self, data, indices):
        attended_cells = []
        attended_indices = []

        def bind(idx):
            if idx < 1:
                idx = 1

            if idx > 361:
                idx = 361

            return idx

        for batch_idx in range(data.shape[0]):
            tmp_idx = indices[batch_idx]

            tmp_idx_above = bind(tmp_idx - 1)
            tmp_idx_below = bind(tmp_idx + 1)
            tmp_idx_right = bind(tmp_idx + 19)
            tmp_idx_left = bind(tmp_idx - 19)

            cell_list = [0, tmp_idx, tmp_idx_left, tmp_idx_right, tmp_idx_above, tmp_idx_below]

            attended_indices.append(cell_list)
            attended_cells.append(torch.unsqueeze(data[batch_idx][cell_list], dim=0))

        return torch.cat(attended_cells, dim=0).to(self.device), np.array(attended_indices)

    def predict(self, batch_x, batch_a):
        pointer_indices = []
        for i in range(batch_x.shape[0]):
            tmp_idx = np.argmax(batch_x[i])
            #print("pointer index = ", tmp_idx)
            pointer_indices.append(tmp_idx)

        # 1) attention mechanism
        batch_x = batch_x.astype(float)
        batch_x = torch.from_numpy(batch_x).to(self.device)

        attended_view, attended_indices = self.attention(batch_x, pointer_indices)

        # print(attended_view)
        # print(attended_indices)

        # 2) run attended_view through transition model, conditional on action. Get predicted transition.
        batch_preds = []
        batch_done_preds = []
        uncertainties = []
        for i in range(batch_a.shape[0]):
            d_input = torch.unsqueeze(attended_view[i], dim=0)
            pred_transition, similarity = self.memory[batch_a[i]].get_memory(d_input)

            # 3) re-map predicted transform into original space using inverse attention.
            batch_preds.append(pred_transition[:, :-1])
            batch_done_preds.append(pred_transition[:, -1])
            uncertainties.append(-similarity.cpu().data.numpy())

        batch_preds = torch.cat(batch_preds, dim=0)
        batch_done_preds = torch.cat(batch_done_preds, dim=0).to(self.device)

        # 4) inverse map to output prediction in original latent space, not the compressed one.
        batch_preds = batch_preds.cpu().data.numpy()

        def inverse_transform(data, indices):
            output = np.zeros([362])
            for idx in range(len(indices)):
                target_idx = indices[idx]

                output[target_idx] = data[idx]

            return output

        outputs = []
        for i in range(batch_preds.shape[0]):
            total_delta = inverse_transform(batch_preds[i], attended_indices[i])
            pred_next = batch_x[i].cpu().data.numpy() + total_delta
            outputs.append(pred_next)

        return np.array(outputs), np.array(uncertainties), batch_done_preds.cpu().data.numpy()

    def trainBatch(self, batch_x, batch_next, batch_a, batch_v, batch_nd, eval=False):

        def display_attended(tmp_view):
            direction = tmp_view[0]
            for i in range(3):
                str_row = ""
                for j in range(3):
                    if (i == 0 and j == 0) or (i == 2 and j == 0) or (i == 0 and j == 2) or (i == 2 and j == 2):
                        str_row += "| ? |"
                    else:
                        actual_idx = 0
                        if i == 0 and j == 1:
                            actual_idx = 4
                        elif i == 2 and j == 1:
                            actual_idx = 5
                        elif i == 1 and j == 0:
                            actual_idx = 2
                        elif i == 1 and j == 2:
                            actual_idx = 3
                        elif i == 1 and j == 1:
                            actual_idx = 1

                        if round(tmp_view[actual_idx]) == 1.0:
                            # empty cell
                            str_row += "|   |"
                        elif round(tmp_view[actual_idx]) == 2.0:
                            # wall
                            str_row += "|||||"
                        elif round(tmp_view[actual_idx]) == 10.0:
                            # cursor
                            if round(direction) == 0:
                                str_row += "| > |"
                            elif round(direction) == 1:
                                str_row += "| V |"
                            elif round(direction) == 2:
                                str_row += "| < |"
                            elif round(direction) == 3:
                                str_row += "| A |"
                            else:
                                str_row += "| ? |"
                        elif round(tmp_view[actual_idx]) == 8.0:
                            # goal
                            str_row += "| # |"
                        else:
                            str_row += "|%.2f|" % tmp_view[actual_idx]

                print(str_row)
                print("------------------------------------------------------------------------------------------")

        pointer_indices = []
        for i in range(batch_x.shape[0]):
            tmp_idx = np.where(batch_x[i] == 10.)[0][0]
            pointer_indices.append(tmp_idx)

        # 1) attention mechanism
        batch_x = torch.from_numpy(batch_x)
        batch_next = torch.from_numpy(batch_next)

        attended_view, attended_indices = self.attention(batch_x, pointer_indices)
        attended_next, attended_next_indices = self.attention(batch_next, pointer_indices)

        # 2) run attended_view through transition model, conditional on action. Get predicted transition.
        batch_preds = []
        batch_done_preds = []
        for i in range(batch_a.shape[0]):
            # if batch_nd[i] == 1:
            #     print("Full grid: ")
            #     utils.display(batch_x[i].cpu().data.numpy())

            pred_transition, _ = self.memory[batch_a[i]].get_memory(torch.unsqueeze(attended_view[i], dim=0))

            # 3) re-map predicted transform into original space using inverse attention.
            batch_preds.append(pred_transition[:, :-1])
            batch_done_preds.append(pred_transition[:, -1])

            # print("Attended region: ")
            # display_attended(attended_view[i].cpu().data.numpy())
            # print("done_pred: %s, actual_done: %s" % (batch_done_preds[i].cpu().data.numpy(), batch_nd[i]))

        batch_preds = torch.cat(batch_preds, dim=0).to(self.device)
        batch_done_preds = torch.cat(batch_done_preds, dim=0).to(self.device)

        actual_deltas = attended_next - attended_view

        dynamics_loss = F.mse_loss(batch_preds, actual_deltas)

        actual_dones = torch.from_numpy(batch_nd).to(self.device).double()

        done_loss = F.binary_cross_entropy(batch_done_preds.double(), actual_dones)

        if not eval:
            for i in range(batch_x.shape[0]):
                memory_entry = torch.cat((actual_deltas[i], torch.unsqueeze(actual_dones[i], dim=0)))
                self.memory[batch_a[i]].save_memory(attended_view[i], memory_entry)

        value_preds = self.valueModel(batch_x.to(self.device))

        actual_values = torch.from_numpy(batch_v).to(self.device)

        # pred_target_pairs = np.concatenate((value_preds.cpu().data.numpy(), batch_v), axis=1)
        # print("value pairs: ", pred_target_pairs)

        value_loss = F.mse_loss(value_preds, actual_values)

        return dynamics_loss + value_loss + done_loss, dynamics_loss, value_loss, done_loss
