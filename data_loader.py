import pdb
import pandas as pd
import numpy as np
from gulpio import GulpDirectory
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset, GulpVideoSegment
from pathlib import Path
from PIL import Image


def load_data_example(dataset, index=10, frame_length=32, image_size=[64, 64]):
    '''load and resample the selected video from epic-kitchen dataset.
    
    Parameters
    ----------
    index : int
        the index of target video in dataset, max of 28471 for training set.
    frame_length : int
        length of video frames after resampling.
    image_szie : (H, W) array like of int 
        output image size of RGB image
    ----------
    output : (X, y) ~ (4Darray, int)
    '''

    '''
    number of training samples : 28472
    Number of unique verb classes in training : 119
    Number of unique noun classes in training : 325
    '''

    #train_labels = pd.read_pickle('../epic/data/processed/train_labels.pkl')
    #test_seen_labels = pd.read_pickle('../epic/data/processed/test_seen_labels.pkl')
    #test_unseen_labels = pd.read_pickle('../epic/data/processed/test_unseen_labels.pkl')

    gulp_root = Path('../epic/data/processed/gulp')
    class_type = 'verb'
    #rgb_train = EpicVideoDataset(gulp_root / 'rgb_train', class_type)
    rgb_train = dataset

    example_segment = rgb_train.video_segments[index]

    example_frames = rgb_train.load_frames(example_segment)
    example_label = example_segment.label
    
    frames_np = _precess_frames(example_frames, frame_length, image_size)
    label_int = example_label
    return frames_np, label_int

def _precess_frames(frames_pil, frame_length, image_size):
    frames_np = np.zeros((frame_length, image_size[0], image_size[1], 3))
    input_len = len(frames_pil)

    mode = None
    if input_len < frame_length:
        mode = 'up_sample'
        offset = frame_length - input_len
    if input_len > frame_length:
        mode = 'down_sample'
        selection = (np.linspace(0, input_len - 1, num=frame_length)).astype(int).tolist()

    for idx, temp_pil in enumerate(frames_pil):
        if mode == 'down_sample':
            if idx in selection:
                temp_pos = selection.index(idx)
                temp_np = np.array(temp_pil.convert('RGB').resize(image_size))
                frames_np[temp_pos] = temp_np
        elif mode == 'up_sample':
            temp_pos = offset + idx
            temp_np = np.array(temp_pil.convert('RGB').resize(image_size))
            frames_np[temp_pos] = temp_np
        else:
            raise ValueError('Invalid mode.')
    return frames_np

class Generator:

    def __init__(self, dataset):

        self.dataset = dataset

        self.image_size = [64,64]
        self.frame_length = 32
        self.shuffle = True

        self.batch_size = 8
        raise NotImplementedError('Fix the next line')
        self.number_of_indexes = len(dataset)

        self.indexes = np.arange(self.number_of_indexes)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.i = 0

    def at_each_yield(self):
        if self.i > self.number_of_indexes - self.batch_size:
            if self.shuffle:
                np.random.shuffle(self.indexes)
            self.i = 0

    def fetch(self):

        batch_size = self.batch_size

        # Allocate batch space
        batch_X = np.zeros([batch_size,self.frame_length]+self.image_size+[3])
        batch_y = np.zeros((batch_size,))
        while True:

            # Fill current yield (batch)
            for b in range(batch_size):

                curr_index = self.indexes[self.i]
                X,y = load_data_example(self.dataset, index=curr_index, frame_length=self.frame_length, image_size=self.image_size)
                batch_X[b], batch_y[b] = X,y

                self.i += 1

            self.at_each_yield()

            yield batch_X, batch_y

if __name__ == "__main__":
    (example_frames, example_label) = load_data_example()

    rgb_train = EpicVideoDataset(gulp_root / 'rgb_train', class_type)
    gen_train = Generator(rgb_train)
    # gen_valid = Generator(rgb_valid)
    # steps_per_epoch = number of data / batch_size of the train generator (default 8)
    # epochs = number of epochs
    # validation_steps = number of validation data / batch_size of the validation generator (default 8)
    # model.fit_generator(rgb_gen_train.fetch(), steps_per_epoch, epochs, validation_data=gen_valid.fetch(), validation_steps=validation_steps)
