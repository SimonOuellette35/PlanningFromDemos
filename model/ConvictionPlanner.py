import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from model.dnd import DND
import torchvision.models as models

# This adds an n-step loss to help with planning accuracy.

class ConvictionPlanner(nn.Module):

    def __init__(self, z_dim=512, action_space=5, stop_threshold=0., dict_len=10000, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.action_space = action_space
        self.STOP_THRESHOLD = stop_threshold

        # encoder
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.encoder.double()

        # value model
        # Feed-forward, regression (single scalar)
        # self.valuePredictor = nn.Sequential(
        #     nn.Linear(z_dim, 256),
        #     nn.ELU(),
        #     nn.Linear(256, 128),
        #     nn.ELU(),
        #     nn.Linear(128, 1)
        # ).to(device)

        self.valuePredictor = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        ).to(device)

        self.valuePredictor.double()

        # DND-based transition model
        self.DICT_LEN = dict_len
        self.memory = []
        for _ in range(action_space):
            self.memory.append(DND(self.DICT_LEN, z_dim))

    # TODO: add some kind of regularization term

    # TODO: penalize a large variety of distinct states in the DND? how?

    # TODO: add a term that forces conviction of non-relevant action to be worse, somehow? A kind of clustering term?
    #  at this point, if I'm going to use a DQfD-like supervised margin loss, just use a neural network for the
    #  transition model?
    def calculate_loss(self, pred_values, actual_values,
                       pred_transition_values, actual_transition_values,
                       pred_embs, actual_embs):

        # This loss term enforces instantaneous value estimation accuracy
        loss_t1 = F.mse_loss(pred_values, actual_values)

        # This loss term enforces 5-step ahead value prediction consistency
        loss_t2 = F.mse_loss(pred_transition_values, actual_transition_values)

        #loss_t3 = F.mse_loss(pred_embs, actual_embs)

        return loss_t1 + loss_t2 #+ loss_t3

    # data structure:
    #  train_images: Tensor([BATCH_SIZE, NC, W, H])
    #  train_next_images: Tensor([BATCH_SIZE, 5, NC, W, H])
    #  train_current_val: [BATCH_SIZE, 1]   --> the actual value of current image
    #  train_actions: [BATCH_SIZE, 5]   --> the 5 following actions in the sequence for each batch element
    #  train_values: [BATCH_SIZE, 5]    --> the 5 following actual values in the sequence of each batch element
    def trainSequence(self, train_images, train_next_images, train_current_val, train_actions, train_values, eval=False):

        # predict the next 5 transitions and estimate each of their values. They must match the next 5 train_values.

        embeddings = torch.reshape(self.encoder(train_images), [train_images.shape[0], self.z_dim]).to(self.device)
        next_embedding1 = torch.reshape(self.encoder(train_next_images[:, 0]),
                                        [train_next_images.shape[0], self.z_dim]).to(self.device)
        next_embedding2 = torch.reshape(self.encoder(train_next_images[:, 1]),
                                        [train_next_images.shape[0], self.z_dim]).to(self.device)
        next_embedding3 = torch.reshape(self.encoder(train_next_images[:, 2]),
                                        [train_next_images.shape[0], self.z_dim]).to(self.device)
        next_embedding4 = torch.reshape(self.encoder(train_next_images[:, 3]),
                                        [train_next_images.shape[0], self.z_dim]).to(self.device)
        next_embedding5 = torch.reshape(self.encoder(train_next_images[:, 4]),
                                        [train_next_images.shape[0], self.z_dim]).to(self.device)

        # Pred_values = [N, 1], actual_values = [N, 1]
        pred_values = self.valuePredictor(embeddings)
        actual_values = torch.from_numpy(train_current_val).to(self.device).double()

        pred_nStep_values = []
        pred_nStep_embs = []
        next_embeddings = []
        actions = np.reshape(train_actions, [-1, 5]).astype(np.int)

        # Note: not sure how to do this intelligently without hard-coding the number of steps ahead, without messing
        #  up the computational graph...
        for i in range(actions.shape[0]):
            tmp_emb_output1, _ = self.memory[actions[i, 0]].get_memory(embeddings[i])
            tmp_emb_output1 = torch.reshape(tmp_emb_output1, [self.z_dim])

            tmp_emb_output2, _ = self.memory[actions[i, 1]].get_memory(tmp_emb_output1)
            tmp_emb_output2 = torch.reshape(tmp_emb_output2, [self.z_dim])

            tmp_emb_output3, _ = self.memory[actions[i, 2]].get_memory(tmp_emb_output2)
            tmp_emb_output3 = torch.reshape(tmp_emb_output3, [self.z_dim])

            tmp_emb_output4, _ = self.memory[actions[i, 3]].get_memory(tmp_emb_output3)
            tmp_emb_output4 = torch.reshape(tmp_emb_output4, [self.z_dim])

            tmp_emb_output5, _ = self.memory[actions[i, 4]].get_memory(tmp_emb_output4)
            tmp_emb_output5 = torch.reshape(tmp_emb_output5, [self.z_dim])

            tmp_emb_output = tmp_emb_output1.to(self.device).double()
            tmp_emb_output2 = tmp_emb_output2.to(self.device).double()
            tmp_emb_output3 = tmp_emb_output3.to(self.device).double()
            tmp_emb_output4 = tmp_emb_output4.to(self.device).double()
            tmp_emb_output5 = tmp_emb_output5.to(self.device).double()

            tmp_values1 = self.valuePredictor(tmp_emb_output)
            tmp_values2 = self.valuePredictor(tmp_emb_output2)
            tmp_values3 = self.valuePredictor(tmp_emb_output3)
            tmp_values4 = self.valuePredictor(tmp_emb_output4)
            tmp_values5 = self.valuePredictor(tmp_emb_output5)

            pred_nStep_embs.append(tmp_emb_output)
            pred_nStep_embs.append(tmp_emb_output2)
            pred_nStep_embs.append(tmp_emb_output3)
            pred_nStep_embs.append(tmp_emb_output4)
            pred_nStep_embs.append(tmp_emb_output5)

            next_embeddings.append(next_embedding1[i])
            next_embeddings.append(next_embedding2[i])
            next_embeddings.append(next_embedding3[i])
            next_embeddings.append(next_embedding4[i])
            next_embeddings.append(next_embedding5[i])

            pred_nStep_values.append(tmp_values1)
            pred_nStep_values.append(tmp_values2)
            pred_nStep_values.append(tmp_values3)
            pred_nStep_values.append(tmp_values4)
            pred_nStep_values.append(tmp_values5)

        # add the new transition (only 1-step ahead) to the DND.
        if not eval:
            for i in range(actions.shape[0]):
                self.memory[actions[i, 0]].save_memory(embeddings[i], next_embedding1[i])

        pred_nStep_values = torch.cat(pred_nStep_values, dim=0)
        pred_nStep_embs = torch.cat(pred_nStep_embs, dim=0)
        next_embeddings = torch.cat(next_embeddings, dim=0)

        train_values = torch.reshape(torch.from_numpy(train_values), [-1]).to(self.device).double()

        return self.calculate_loss(pred_values, actual_values,
                                   pred_nStep_values, train_values,
                                   pred_nStep_embs, next_embeddings)
