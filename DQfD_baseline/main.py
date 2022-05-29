import warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym
import argparse
import utils
from DQfD import DQfDAgent

W = 192
H = 192

def getDemoData():
    data_dir = './data/'

    X, a, v = utils.loadMinigridDemonstrationsV2(data_dir, W, H)

    training_X = X[:100]
    training_v = v[:100]
    training_a = a[:100]

    test_X = X[100:]
    test_a = []
    for i in range(len(a) - 100):
        test_a.append(np.array(a[100 + i]))

    test_v = []
    for i in range(len(v) - 100):
        test_v.append(np.array(v[100 + i]))

    return training_X, training_a, training_v, test_X, test_a, test_v

def main(use_PER):
    
    episode_rewards = []
    n_episode = 250

    for _ in range(5):
        env = gym.make('CartPole-v1')
        
        # DQfDagent
        dqfd_agent = DQfDAgent(use_PER, n_episode)
        
        # DQfD agent train (pre-training only on demonstrations)
        train_X, train_a, train_v, test_X, test_a, test_v = getDemoData()
        dqfd_agent.train(train_X, train_a, train_v)

        # TODO: evaluate performance

        env.close()

    return np.mean(episode_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_per', default=False)
    args = parser.parse_args()

    mean_reward = main(args.use_per)
