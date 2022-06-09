import random
from collections import namedtuple
import cv2
import os
import csv
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from gym.core import ObservationWrapper

def loadMinigridDemonstrationsV2(data_dir, width=192, height=192):

    def mapMinigridAction(a_str):
        if a_str == '':
            return 0

        if a_str == 'Actions.left':
            return 0

        if a_str == 'Actions.right':
            return 1

        if a_str == 'Actions.forward':
            return 2

        if a_str == 'Actions.toggle':
            return 3

    WIDTH = width
    HEIGHT = height
    NC = 3

    #  N = number of total sequences to train on
    #  T = number of timesteps in the sequences: this can vary from sequence to sequence, hence we have embedded lists,
    #   but not numpy arrays (or tensors, either)

    X = []      # shape = [N, T, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS]
    a = []      # shape = [N, T, 1]
    v = []      # shape = [N, T, 1]

    # Load data_fourrooms: each sequence is a sequence of images combined with a sequence of preceding actions and a
    # confidence value associated with each image.
    directory = os.fsencode(data_dir)

    demos = []
    for file in os.listdir(directory):
        basename = os.path.splitext(file)[0].decode("utf-8")
        if basename not in demos:
            demos.append(basename)

    print("Found demos: ", demos)

    for demo in demos:
        print("Loading demo %s" % demo)
        csv_filename = os.fsdecode("%s.csv" % demo)
        avi_filename = os.fsdecode("%s.avi" % demo)

        action_sequence = []
        with open("%s%s" % (data_dir, csv_filename), 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')

            for row in datareader:
                tmp_a = mapMinigridAction(row[0])
                action_sequence.append(tmp_a)

        a_seq = np.reshape(action_sequence, [-1, 1])
        a.append(a_seq)

        image_sequence = []

        # load video frame by frame
        cap = cv2.VideoCapture('%s%s' % (data_dir, avi_filename))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frame_count += 1

                # convert OpenCV image to array or tensor
                image = cv2.resize(frame, (WIDTH, HEIGHT))

                # plt.imshow(image)
                # plt.show()

                # Convert the image to PyTorch tensor
                np_image = np.reshape(image, [1, NC, HEIGHT, WIDTH])
                image_sequence.append(np_image/255.)
            else:
                cap.release()

        cap.release()
        cv2.destroyAllWindows()

        X.append(image_sequence)

        value_sequence = []
        for i in range(frame_count):
            value_sequence.append(-(frame_count-i))

        v.append(np.reshape(value_sequence, [-1, 1]))

    return X, a, v

# environment wrapper to process the input frames and turn them into pytorch tensors
class PreprocessImage(ObservationWrapper):
    def __init__(self, env, width=80, height=80):
        if env is not None:
            super(PreprocessImage, self).__init__(env)
        self.resize = T.Compose([T.ToPILImage(),
                    T.Scale((width,height), interpolation=Image.CUBIC),
                    T.ToTensor()])

# image transformations, to a 1x1x80x80 tensor
    def _observation(self, screen):
        screen = torch.from_numpy(screen)
        screen = screen.permute(2, 1, 0)
        screen = self.resize(screen)
        screen = screen.mean(0, keepdim=True)
        screen = screen.unsqueeze(0)
        return screen



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'n_reward'))

# object to hold all the transformations from which batches are sampled
class ReplayMemory(object):

    # capacity == -1 means unlimited capacity
    def __init__(self, capacity=-1):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, trans):
        if len(self.memory) < self.capacity or self.capacity < 0:
            self.memory.append(None)
        self.memory[self.position] = trans
        self.position = self.position + 1
        if self.capacity > 0:
            self.position = self.position % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class EpsGreedyPolicy(object):
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        eps_steps = max(eps_steps, 1)
        self.steps_done = 0

    # epsilon changes linearly from eps_start to eps_end within the first eps_steps number of steps
    def select_action(self, q_vals, env):
        sample = random.random()
        self.steps_done += 1
        if sample > min(self.steps_done / self.eps_steps, 1) * (self.eps_end - self.eps_start) + self.eps_start:
            return q_vals.max(1)[1].cpu()
        else:
            return torch.LongTensor([env.action_space.sample()])
