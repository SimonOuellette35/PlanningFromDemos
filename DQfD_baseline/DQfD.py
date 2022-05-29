import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import utils
import torchvision.models as models

plot_not_use_per = []
plot_use_per = []

PRETRAIN_STEP = 1000
MINIBATCH_SIZE = 20
RUNNING_MINIBATCH_SIZE = 20

class DQfDNetwork(nn.Module):

    def __init__(self, in_size, out_size):
        super(DQfDNetwork, self).__init__()
        HIDDEN_SIZE = 256

        self.f1 = nn.Linear(in_size, HIDDEN_SIZE)
        self.f2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.f3 = nn.Linear(HIDDEN_SIZE, out_size)

        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)
        nn.init.xavier_uniform_(self.f3.weight)

        self.opt = torch.optim.Adam(self.parameters())
        self.loss = torch.nn.MSELoss()

    def forward(self,x):
        x1 = F.relu(self.f1(x))
        x2 = F.relu(self.f2(x1))
        x3 = self.f3(x2)
        res = F.softmax(x3)
        return res
    
class DQfDAgent(object):
    def __init__(self, use_per, n_episode, state_size=512, action_size=3, device='cuda'):
        self.n_EPISODES = n_episode
        self.use_per = use_per
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        # encoder
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.encoder.double()

        # policy and target networks
        self.policy_network = DQfDNetwork(self.state_size, self.action_size).to(self.device)
        self.target_network = DQfDNetwork(self.state_size, self.action_size).to(self.device)

        self.frequency = 1
        self.memory = Memory()
        print('device is', self.device)

    def train_network(self, args=None, pretrain=False, minibatch_size=MINIBATCH_SIZE):
        l1 = l2 = l3 = 0.23

        if pretrain:
            self.n = minibatch_size
            minibatch = self.sample_minibatch(self.n)
        else:
            self.n = 1
            minibatch = [args]

        for episode in range(self.n):
            self.policy_network.eval()
            self.target_network.eval()

            frame, action, reward, next_frame, done = minibatch[episode]
            frame = torch.from_numpy(frame).float().to(self.device)
            frame.requires_grad = True
            next_frame = torch.from_numpy(next_frame).float().to(self.device)
            next_frame.requires_grad = True

            # Get image embeddings

            state = self.encoder(frame)
            next_state = self.encoder(next_frame)

            # DQN loss

            next_action = self.policy_network(next_state).argmax()
            Q_target = self.target_network(next_state)[next_action]
            Q_predict = self.policy_network(state)[action]

            double_dqn_loss = reward + self.gamma * Q_target - Q_predict
            double_dqn_loss = torch.pow(double_dqn_loss, 2)

            def margin(action1, action2):
                if action1 == action2:
                    return torch.Tensor([0]).to(self.device)
                return torch.Tensor([0.2]).to(self.device)

            # margin_classification_loss

            partial_margin_classification_loss = torch.Tensor([0]).to(self.device)
            for selected_action in range(self.action_size):
                expect = self.target_network(state)[selected_action]
                partial_margin_classification_loss = max(partial_margin_classification_loss,
                                                         expect + margin(action, selected_action))
            margin_classification_loss = partial_margin_classification_loss - Q_predict

            # n-step returns #

            n_step_returns = torch.Tensor([reward]).to(self.device)
            current_n_step_next_state = next_state.detach().cpu().numpy()
            n = min(self.n - episode, 10)
            for exp in range(1, n):
                _, _, current_n_step_reward, current_n_step_next_state, __done__, _ = minibatch[episode + exp]
                if __done__:
                    break
                n_step_returns = n_step_returns + (self.gamma ** exp) * current_n_step_reward

            expect = self.target_network(torch.from_numpy(current_n_step_next_state).to(self.device))[action]
            partial_n_step_returns = (self.gamma ** 10) * expect
            n_step_returns = n_step_returns + partial_n_step_returns
            self.policy_network.train()
            self.target_network.train()

            self.policy_network.opt.zero_grad()

            L2_loss = self.policy_network.loss(Q_target, Q_predict)

            loss = double_dqn_loss + l1 * margin_classification_loss + l2 * n_step_returns + l3 * L2_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
            self.policy_network.opt.step()

    def sample_minibatch(self, n=1):
        sample_fn = self.memory.sample_original if self.use_per else self.memory.sample
        result = sample_fn(k=n)
        return result

    def pretrain(self):
        for i in range(PRETRAIN_STEP):
            if i % 100 == 0:
                print(f"{i} pretrain step")

            self.train_network(pretrain=True)
            if i % self.frequency == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self, X, a, v):
        # here a is of shape [N, T, 1] where N is the number of episodes, T is the number of steps in that
        #  episode (varies from episode to episode). Same for v. For X, the only difference is that instead of a
        #  1 for the last dimension, we have a 3 additional dimensions for the image (num channels x width x height)

        # Put X, a, v into memory in format: [frame, action, reward, next_frame, done]
        for n in range(len(X)):
            for t in range(len(X[n])):
                frame = np.array(X[n][t])
                action = a[n][t]
                reward = v[n][t]

                if t < len(X[n])-1:
                    next_frame = np.array(X[n][t+1])
                    done = False
                else:
                    next_frame = np.zeros_like(np.array(X[n][t]))
                    done = True

                self.memory.push([frame, action, reward, next_frame, done], self)

        self.pretrain()

class Memory():

    def __init__(self, length=10000):
        self.idx = 0
        self.length = length
        self.container = [None for _ in range(length)]
        self.td_errors = [None for _ in range(length)]
        self.priority = [None for _ in range(length)]
        self.max = 0
        self.epsilon = 0.001
        self.alpha = 2
        self.beta = 0

    def push(self, obj, agent: DQfDAgent):
        if self.idx == self.length:
            self.idx = 0
        self.container[self.idx] = obj
        state, action, reward, next_state, done = obj
        state = torch.from_numpy(state).to(agent.device)
        next_state = torch.from_numpy(next_state).to(agent.device)
        self.td_errors[self.idx] = abs(reward + agent.gamma * agent.target_network(next_state).max() - agent.policy_network(state)[action]) + self.epsilon
        self.idx += 1
        self.max = max(self.max, self.idx-1)
        tmp = self.td_errors[:self.max+1]
        self.priority = torch.stack(tmp)
        self.priority = torch.pow(self.priority, self.alpha)
        sum_ = self.priority.sum()
        self.priority = torch.pow(self.priority, -1) / sum_
        self.priority = torch.pow(self.priority, self.beta)
        self.priority = self.priority.shape[0] / self.priority
        self.priority = self.priority.detach().numpy()

    def sample(self, sample_index=False, k=1):
        result = []
        for _ in range(k):
            if sample_index:
                choice = random.randint(0, self.max-k+1)
            else:
                choice = random.randint(0, self.max)
                choice = self.container[choice]
            result.append(choice)
        return result

    def sample_original(self, sample_index=False, k=1):
        if sample_index:
            result = random.choices(list(range(0, len(self.priority)-k+1)), weights=self.priority, k=k)
        else:
            result = random.choices(self.container[:self.max+1], weights=self.priority, k=k)
        return result

    def plot_priority(self):
        plt.plot(self.priority)
        plt.show()

def plot(use_per=False):
    filename = f"./plot_use_per_{str(use_per).lower()}"
    title = "original" if use_per else "simple"
    arr = plot_use_per if use_per else plot_not_use_per
    cnt = 0
    for e in arr:
        cnt += 1
        plt.plot(e, label=f"{title} {cnt}")
    plt.xlabel("Episode")
    plt.ylabel("Average 20 latest step rewards")
    # plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(f"{filename}.png")
    plt.clf()
