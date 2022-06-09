import torch.nn as nn
import torch.nn.functionals as F
from gym_minigrid.wrappers import *
import cv2
import numpy as np
from gym_minigrid.window import Window

# TODO: in simplified, we assume the conscious bottleneck is already taken care of (by hard-cording focus on the
# area surrounding the agent), and focus on correctly training the generate_delta that can learn the fundamental
# principles.

# TODO: given a symbolic version of the input, flattened as a single vector, train a dynamics model that learns the purely
# generalizable first principles of the environment, using a conscious bottleneck.
class ConsciousPlanner(nn.Module):

    def __init__(self, gamma=0.1):
        self.gamma = gamma

    def forward(self, x, verbose=False):
        # TODO: in this version we hard-code the attention mechanism.
        masked_x = self.multi_head_attention(x)
        if verbose:
            # TODO: can also visualize the attention filter learned in the multi-head attention block.
            print("For input %s, using masked x is %s" % (x, masked_x))

        # TODO: generate_delta might need to be a recurrent mechanism (or transformer?) to successfully learn
        # reasoning of the kind: "if agent has a wall on its right cell, going right shouldn't do anything".
        x_delta = self.generate_delta(masked_x)

        return x_delta + x

    def calculate_loss(self, frames, predictions, attention_filters):
        prediction_loss = F.mse_loss(frames, predictions)
        regularization_loss = F.l2_loss(attention_filters)

        return prediction_loss + self.gamma * regularization_loss

def flatten_obs(obs):
    return np.reshape(obs, [-1])

model = ConsciousPlanner()

# instantiate minigrid env, getting states in raw format (symbolic form) -- flattened
env = gym.make('MiniGrid-FourRooms-v0')
obs = env.reset()
state = flatten_obs(obs)

# TODO: training loop, just grab a bunch of (s, a, r, s') transitions from a lot of distinct grid instances. Load into
#  an episode replay memory, and train from random minibatches of that memory.
while log_done_counter < NUM_EPISODES:
    if log_done_counter >= SKIP_LEVELS:
        if log_done_counter == TEST_EPISODE:
            is_test = True
            fourcc = cv2.VideoWriter_fourcc('p', 'n', 'g', ' ')
            out = cv2.VideoWriter("%s-%s.avi" % ('test_examples', log_done_counter - (TEST_EPISODE-1)), fourcc, 60, (W, H))

        tmp_obs = obs
        actions = get_actions(obs)

        env.render()
        obs, rewards, done, _ = env.step(actions)
        obs = env.render('rgb_array', tile_size=TILE_SIZE)
        if out is not None:
            out.write(obs)

        # if is_test:
        #     add_to_memory(actions, tmp_obs, obs)

        frame_counter += 1

    else:
        obs, rewards, done, _ = env.step(6)

    if done:

        if is_test:
            if frame_counter <= 90:
                test_successes += 1
            frames_per_episode += frame_counter
            total_test_episodes += 1

        log_done_counter += 1
        print("Level %s Done!" % log_done_counter)

        obs = env.reset()
        obs = env.render('rgb_array', tile_size=TILE_SIZE)

        frame_counter = 0

# TODO: eval loop with visualization of masks