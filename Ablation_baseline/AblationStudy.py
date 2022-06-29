import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from model.dnd import DND
import torchvision.models as models

class AblationModel(nn.Module):

    def __init__(self, z_dim=512, action_space=5, stop_threshold=0., dict_len=1000, device='cuda'):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.action_space = action_space
        self.STOP_THRESHOLD = stop_threshold

        # encoder
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.encoder.double()

        # value model
        # Feed-forward, regression (single scalar)
        self.valuePredictor = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        ).to(device)

        self.valuePredictor.double()

        # DND-based transition model
        self.DICT_LEN = dict_len
        self.memory = []
        for _ in range(action_space):
            self.memory.append(DND(self.DICT_LEN, z_dim))

    def calculate_loss(self, pred_values, actual_values, pred_transition_values, actual_transition_values):
        loss_t1 = F.mse_loss(pred_values, actual_values)
        loss_t2 = F.mse_loss (pred_transition_values, actual_transition_values)

        return loss_t1 + loss_t2

    def trainSequence(self, train_images, train_actions, train_values, eval=False):

        embeddings = torch.reshape(self.encoder(train_images), [train_images.shape[0], self.z_dim]).to(self.device)

        # Pred_values = [N, 1], actual_values = [N, 1]
        pred_values = self.valuePredictor(embeddings)
        actual_values = torch.from_numpy(train_values).to(self.device).double()

        pred_transition_confidences = []
        actual_transition_confidences = []

        actions = np.reshape(train_actions, [-1]).astype(np.int)
        for i in range(embeddings.shape[0] - 1):
            pred_embedding, _ = self.memory[actions[i+1]].get_memory(embeddings[i])

            if not eval:
                self.memory[actions[i+1]].save_memory(embeddings[i], embeddings[i+1])

            # pred_t_value = [1, 1]
            pred_t_value = self.valuePredictor(pred_embedding.to(self.device).double())

            pred_transition_confidences.append(pred_t_value)
            actual_transition_confidences.append(torch.unsqueeze(actual_values[i+1], dim=0))

        # pred_transition_confidences = [N-1, 1]
        # actual_transition_confidences = [N-1, 1]
        pred_transition_confidences = torch.cat(pred_transition_confidences, dim=0)
        actual_transition_confidences = torch.cat(actual_transition_confidences, dim=0)

        return self.calculate_loss(pred_values, actual_values, pred_transition_confidences, actual_transition_confidences)
