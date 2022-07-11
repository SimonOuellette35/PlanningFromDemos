#!/usr/bin/env python3

import argparse
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import csv

datawriter = None

counter = 0

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def extract_direction(obs):
    direction = 0
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            if obs[i, j,0] == 10:
                direction = obs[i,j,2]

    return direction

def redrawAndSave(obs, action):
    dir = extract_direction(obs['image'])
    data = obs['image'][:, :, 0]
    data_row = np.reshape(data, [-1])

    # concatenate direction to data vector
    data_row = np.concatenate([[dir], data_row])

    img = env.render('rgb_array', tile_size=args.tile_size)

    with open("actions_%s-%s.csv" % (args.save_filename, counter), 'a') as actions_file:
        datawriter = csv.writer(actions_file, delimiter=',')
        datawriter.writerow([action])

    with open("frames_%s-%s.csv" % (args.save_filename, counter), 'a') as frames_file:
        datawriter = csv.writer(frames_file, delimiter=',')
        datawriter.writerow(data_row)

    window.show_img(img)

def reset():
    global counter

    if args.seed != -1:
        env.seed(args.seed)

    counter += 1

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    if args.save_filename is not None:
        redrawAndSave(obs, None)
    else:
        redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    #print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        if args.save_filename is not None:
            redrawAndSave(obs, action)
        else:
            redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    #default='MiniGrid-Empty-Random-6x6-v0'
    default='MiniGrid-FourRooms-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=12
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
parser.add_argument(
    '--save_filename',
    default=None,
    help="file to save the action/observation sequences (demonstrations) to"
)
args = parser.parse_args()

env = gym.make(args.env)

# if args.agent_view:
#     env = RGBImgPartialObsWrapper(env)
#     env = ImgObsWrapper(env)

env = FullyObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
