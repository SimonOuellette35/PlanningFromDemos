import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

USE_RESNET = True

class DQN(nn.Module):
    def __init__(self, dtype, input_shape, num_actions):
        super(DQN, self).__init__()
        self.dtype = dtype

        # encoder
        if USE_RESNET:
            resnet = models.resnet18(pretrained=True)
            modules = list(resnet.children())[:-1]
            self.encoder = nn.Sequential(*modules)

            self.encoder_lin = nn.Linear(512, 289)
            self.lin1 = nn.Linear(289, 256)
        else:
            self.conv1 = nn.Conv2d(input_shape[0], 32, 5, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

            conv_out_size = self._get_conv_output(input_shape)

            self.lin1 = nn.Linear(conv_out_size, 256)

        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, num_actions)

        self.type(dtype)

    def _get_conv_output(self, shape):
        input = Variable(torch.rand(1, *shape))
        output_feat = self._forward_conv(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_conv(self, x):
        if USE_RESNET:
            tmp = self.encoder(x)
            tmp = tmp.view(tmp.size(0), -1)

            tmp = F.relu(tmp)
            return self.encoder_lin(tmp)
        else:
            x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
            x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
            return x

    def forward(self, states):
        x = self._forward_conv(states)

        if not USE_RESNET:
            x = x.view(states.size(0), -1)

        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        return self.lin3(x)