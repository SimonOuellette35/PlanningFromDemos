import torch.nn as nn
import torch
import torchvision.models as models

class BCModel(nn.Module):

    def __init__(self, z_dim=512, action_space=2, stop_threshold=0., device='cuda'):
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

        self.policyPredictor = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_space)
        ).to(device)

        self.policyPredictor.double()
        self.loss_fn = torch.nn.MSELoss()

    def calculate_loss(self, pred_values, actual_values):
        a = (pred_values[:, 0] - actual_values[:, 0]) ** 2.0
        b = (pred_values[:, 1] - actual_values[:, 1]) ** 2.0

        return torch.mean(a+b)

    def trainSequence(self, train_images, train_actions):
        preds = self.forward(train_images)

        labels = torch.from_numpy(train_actions).to(self.device)
        return self.calculate_loss(preds, labels)

    def forward(self, X):
        embeddings = self.encoder(X)
        embeddings = torch.reshape(embeddings, [X.shape[0], self.z_dim])
        pred_actions = self.policyPredictor(embeddings)

        return pred_actions