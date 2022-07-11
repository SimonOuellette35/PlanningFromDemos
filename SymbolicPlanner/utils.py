import os
import csv
import numpy as np
import cv2

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

def loadMinigridSymbolicDemos(data_dir):
    X = []      # shape = [N, T, 362]
    a = []      # shape = [N, T, 1]
    v = []      # shape = [N, T, 1]

    directory = os.fsencode(data_dir)

    demos = []
    for file in os.listdir(directory):
        basename = os.path.splitext(file)[0].decode("utf-8")
        if basename.startswith("frames"):
            basename = basename.replace("frames_", "")
            if basename not in demos:
                demos.append(basename)

    print("Found demos: ", demos)

    for demo in demos:
        print("Loading demo %s" % demo)
        frames_filename = os.fsdecode("frames_%s.csv" % demo)
        actions_filename = os.fsdecode("actions_%s.csv" % demo)

        action_sequence = []
        with open("%s%s" % (data_dir, actions_filename), 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')

            for row in datareader:
                tmp_a = mapMinigridAction(row[0])
                action_sequence.append(tmp_a)

        a_seq = np.reshape(action_sequence, [-1, 1])
        a.append(a_seq)

        frames_sequence = []
        with open("%s%s" % (data_dir, frames_filename), 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')

            for row in datareader:
                float_row = list(map(float, row))
                frames_sequence.append(float_row)

        X.append(np.array(frames_sequence))

        value_sequence = []
        frame_count = len(frames_sequence)
        for i in range(frame_count):
            value_sequence.append(-(frame_count-i))

        v.append(np.reshape(value_sequence, [-1, 1]))

    total_interactions = 0
    for a_episode in a:
        total_interactions += len(a_episode)

    print("Loaded a total of %i interactions." % total_interactions)

    return X, a, v


def loadMinigridDemonstrationsV2(data_dir, width=192, height=192):


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
                np_image = np.reshape(image, [NC, HEIGHT, WIDTH]) / 255.
                image_sequence.append(np_image)
            else:
                cap.release()

        cap.release()
        cv2.destroyAllWindows()

        #img_sequence_tensor = torch.cat(image_sequence, axis=0)

        X.append(image_sequence)

        value_sequence = []
        for i in range(frame_count):
            value_sequence.append(-(frame_count-i))

        v.append(np.reshape(value_sequence, [-1, 1]))

    total_interactions = 0
    for a_episode in a:
        total_interactions += len(a_episode)

    print("Loaded a total of %i interactions." % total_interactions)

    return X, a, v
