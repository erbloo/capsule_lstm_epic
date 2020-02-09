import pdb
import pandas as pd
import os
import numpy as np
from gulpio import GulpDirectory
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset, GulpVideoSegment
import keras
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class DataGenerator(keras.utils.Sequence):
    def __init__(self, number_of_samples, n_classes, image_size=[64, 64], frame_length=32, shuffle=False, batch_size=8):

        self.image_size = image_size
        self.frame_length = frame_length
        self.shuffle = shuffle

        self.batch_size = batch_size
        
        self.n_classes = n_classes
        self.number_of_samples = number_of_samples

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.number_of_samples / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return [X, y], [y, X]

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, self.frame_length, self.image_size[0], self.image_size[1], 3))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        for idx, dataset_index in enumerate(indexes):
            temp_X, temp_y = self.load_data_example(dataset_index)
            X[idx] = temp_X
            y[idx] = temp_y
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.number_of_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load_data_example(self, index):
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
        Number of unique verb classes in training : 125
        Number of unique noun classes in training : 331
        '''
        n_classes = self.n_classes
        # train_labels = pd.read_pickle('../epic/data/processed/train_labels.pkl')
        # test_seen_labels = pd.read_pickle('../epic/data/processed/test_seen_labels.pkl')
        # test_unseen_labels = pd.read_pickle('../epic/data/processed/test_unseen_labels.pkl')

        # https://github.com/epic-kitchens/starter-kit-action-recognition/tree/master/notebooks
        gulp_root = Path('/data1/yantao/epic/epic/data/processed/gulp')
        class_type = 'verb'  # 'verb+noun'
        rgb_train = EpicVideoDataset(gulp_root / 'rgb_train', class_type)

        example_segment = rgb_train.video_segments[index]

        example_frames = rgb_train.load_frames(example_segment)
        example_label = example_segment.label
        
        frames_np = self._precess_frames(example_frames)
        label_int = example_label
        return frames_np, keras.utils.to_categorical(label_int, num_classes=n_classes)

    def _precess_frames(self, frames_pil):
        frame_length = self.frame_length
        image_size = self.image_size
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
                    temp_np = np.array(temp_pil.convert('RGB').resize(image_size)).astype(float) / 255.
                    frames_np[temp_pos] = temp_np
            elif mode == 'up_sample':
                temp_pos = offset + idx
                temp_np = np.array(temp_pil.convert('RGB').resize(image_size)).astype(float) / 255.
                frames_np[temp_pos] = temp_np
            else:
                raise ValueError('Invalid mode.')
        return frames_np


def traverse_and_save(output_dir):
    gen = DataGenerator(-1, -1, image_size=[32, 32], frame_length=4, shuffle=False, batch_size=1)

    gulp_root = Path('/data1/yantao/epic/epic/data/processed/gulp')
    class_type = 'verb+noun'
    rgb_train = EpicVideoDataset(gulp_root / 'rgb_train', class_type)
    for index, temp_sample in enumerate(tqdm(rgb_train.video_segments)):
        example_frames = rgb_train.load_frames(temp_sample)
        example_label = temp_sample.label
        frames_np = gen._precess_frames(example_frames)
        np.savez(os.path.join(output_dir, 'sample_{0:06d}.npz'.format(index)), frames=frames_np, verb=example_label['verb'], noun=example_label['noun'])

if __name__ == "__main__":
    traverse_and_save('/data1/yantao/epic_saved')