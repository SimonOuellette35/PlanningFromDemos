import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from model.dnd import DND
import torchvision.models as models

# In V2, we predict the a priori action probs as well as the value.
# Also, we try to predict the "done" flag, so that we stop planning when we think we reached the end goal.

class ConvictionPlanner(nn.Module):

    def __init__(self, z_dim=512, action_space=3, stop_threshold=0., dict_len=25000, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.action_space = action_space
        self.STOP_THRESHOLD = stop_threshold

        # encoder
        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.encoder.double()

        # prediction model
        # Feed-forward, regression (single scalar)
        self.valuePredictor = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 2 + action_space)
        ).to(device)

        self.valuePredictor.double()

        # DND-based transition model
        self.DICT_LEN = dict_len
        self.memory = []
        for _ in range(action_space):
            self.memory.append(DND(self.DICT_LEN, z_dim))

    def calculate_loss(self, pred_transition_values, actual_transition_values,
                       pred_done, done,
                       pred_actions, actions):

        value_loss = F.mse_loss(pred_transition_values, actual_transition_values)
        done_loss = F.binary_cross_entropy(pred_done, done)
        action_loss = F.cross_entropy(pred_actions, actions)
        # TODO: add transition embeddings consistency loss?

        return value_loss + done_loss + action_loss

    def predict(self, x):
        preds = self.valuePredictor(x)
        tmp_value = preds[:, 0]
        tmp_done = preds[:, 1]
        tmp_action = preds[:, 2:]

        return torch.reshape(tmp_value, [x.shape[0], 1]), \
               F.sigmoid(torch.reshape(tmp_done, [x.shape[0], 1])), \
               tmp_action

    # data structure:
    #  train_images: Tensor([BATCH_SIZE, NC, W, H])  --> current image
    #  next_images: Tensor([BATCH_SIZE * 5, NC, W, H])   --> next images
    #  done: [BATCH_SIZE * 6]                     --> 1 if done, 0 otherwise, for current + next 5 steps
    #  train_actions: [BATCH_SIZE * 6]               --> following actions for current + next 5 steps
    #  train_values: [BATCH_SIZE * 6, 1]             --> current value + that of next 5 images
    def trainSequence(self, train_images, next_images, train_done, train_actions, train_values, eval=False):

        # predict the next 5 transitions and estimate each of their values. They must match the next 5 train_values.
        embeddings = torch.reshape(self.encoder(train_images), [train_images.shape[0], self.z_dim]).to(self.device)
        target_embeddings = torch.reshape(self.encoder(next_images), [embeddings.shape[0], 5, self.z_dim]).to(self.device)
        target_values = torch.from_numpy(train_values).to(self.device).double()
        target_done = torch.from_numpy(train_done).to(self.device).double()
        target_actions = torch.from_numpy(train_actions).to(self.device).long()

        pred_nStep_values = []
        pred_nStep_done = []
        pred_nStep_actions = []

        actions = np.reshape(train_actions, [-1, 6]).astype(np.int)

        # Note: not sure how to do this intelligently without hard-coding the number of steps ahead, without messing
        #  up the computational graph...
        tmp_value, tmp_done, tmp_action = self.predict(embeddings)

        for i in range(embeddings.shape[0]):

            pred_nStep_values.append(tmp_value[i])
            pred_nStep_done.append(tmp_done[i])
            pred_nStep_actions.append(tmp_action[i])

            tmp_emb_output1, _ = self.memory[actions[i, 0]].get_memory(embeddings[i])
            tmp_emb_output1 = torch.reshape(tmp_emb_output1, [1, self.z_dim])

            tmp_emb_output2, _ = self.memory[actions[i, 1]].get_memory(tmp_emb_output1)
            tmp_emb_output2 = torch.reshape(tmp_emb_output2, [1, self.z_dim])

            tmp_emb_output3, _ = self.memory[actions[i, 2]].get_memory(tmp_emb_output2)
            tmp_emb_output3 = torch.reshape(tmp_emb_output3, [1, self.z_dim])

            tmp_emb_output4, _ = self.memory[actions[i, 3]].get_memory(tmp_emb_output3)
            tmp_emb_output4 = torch.reshape(tmp_emb_output4, [1, self.z_dim])

            tmp_emb_output5, _ = self.memory[actions[i, 4]].get_memory(tmp_emb_output4)
            tmp_emb_output5 = torch.reshape(tmp_emb_output5, [1, self.z_dim])

            tmp_emb_output = tmp_emb_output1.to(self.device).double()
            tmp_emb_output2 = tmp_emb_output2.to(self.device).double()
            tmp_emb_output3 = tmp_emb_output3.to(self.device).double()
            tmp_emb_output4 = tmp_emb_output4.to(self.device).double()
            tmp_emb_output5 = tmp_emb_output5.to(self.device).double()

            # TODO: faster to concatenate all the tmp_emb_outputs into one tensor and call self.predict once on them?

            tmp_values1, tmp_done1, tmp_actions1 = self.predict(tmp_emb_output)
            tmp_values2, tmp_done2, tmp_actions2 = self.predict(tmp_emb_output2)
            tmp_values3, tmp_done3, tmp_actions3 = self.predict(tmp_emb_output3)
            tmp_values4, tmp_done4, tmp_actions4 = self.predict(tmp_emb_output4)
            tmp_values5, tmp_done5, tmp_actions5 = self.predict(tmp_emb_output5)

            pred_nStep_values.append(tmp_values1[0])
            pred_nStep_values.append(tmp_values2[0])
            pred_nStep_values.append(tmp_values3[0])
            pred_nStep_values.append(tmp_values4[0])
            pred_nStep_values.append(tmp_values5[0])

            pred_nStep_done.append(tmp_done1[0])
            pred_nStep_done.append(tmp_done2[0])
            pred_nStep_done.append(tmp_done3[0])
            pred_nStep_done.append(tmp_done4[0])
            pred_nStep_done.append(tmp_done5[0])

            pred_nStep_actions.append(tmp_actions1[0])
            pred_nStep_actions.append(tmp_actions2[0])
            pred_nStep_actions.append(tmp_actions3[0])
            pred_nStep_actions.append(tmp_actions4[0])
            pred_nStep_actions.append(tmp_actions5[0])

        # add the new transitions to the DND.
        if not eval:
            for i in range(actions.shape[0]):
                for j in range(actions[i].shape[0]-1):
                    if j == 0:
                        self.memory[actions[i, j]].save_memory(embeddings[i], target_embeddings[i, 0])
                    else:
                        self.memory[actions[i, j]].save_memory(target_embeddings[i, j-1], target_embeddings[i, j])

        pred_nStep_values = torch.unsqueeze(torch.cat(pred_nStep_values, dim=0), dim=-1)
        pred_nStep_done = torch.cat(pred_nStep_done, dim=0)
        pred_nStep_actions = torch.reshape(torch.cat(pred_nStep_actions, dim=0), [-1, self.action_space])

        return self.calculate_loss(pred_nStep_values, target_values,
                                   pred_nStep_done, target_done,
                                   pred_nStep_actions, target_actions)
