import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import torchvision.models as models


class BCModel(nn.Module):

    def __init__(self, z_dim=512, action_space=3, stop_threshold=0., device='cuda'):
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
        # Feed-forward, classification
        self.policyPredictor = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_space)
        ).to(device)

        self.policyPredictor.double()

    def calculate_loss(self, pred_values, actual_values):
        loss = F.cross_entropy(pred_values, actual_values)

        return loss

    def trainSequence(self, train_images, train_actions, train_values, eval=False):
        embeddings = torch.reshape(self.encoder(train_images[:-1]), [train_images.shape[0]-1, self.z_dim]).to(self.device)

        # Pred_logits = [N, 3], actual_actions = [N, 1]
        pred_logits = F.softmax(self.policyPredictor(embeddings), dim=-1)

        act = np.reshape(train_actions[1:], [-1])
        actual_actions = torch.from_numpy(act).to(self.device)

        pred_actions = np.argmax(pred_logits.cpu().data.numpy(), axis=-1)

        accuracy = 0.
        for i in range(pred_actions.shape[0]):
            if pred_actions[i] == act[i]:
                accuracy += 1

        accuracy /= float(pred_actions.shape[0])

        return self.calculate_loss(pred_logits, actual_actions), accuracy

    def forward(self, X):
        embeddings = torch.reshape(self.encoder(X), [X.shape[0], self.z_dim])
        pred_actions = F.softmax(self.policyPredictor(embeddings), dim=-1)

        return pred_actions