import pdb
import pandas as pd
import numpy as np
from gulpio import GulpDirectory
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, EpicVideoFlowDataset, GulpVideoSegment
from pathlib import Path
from PIL import Image


def load_data_example(index=10, frame_length=32, image_size=[64, 64]):
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
    rgb_train = EpicVideoDataset(gulp_root / 'rgb_train', class_type)
    
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


if __name__ == "__main__":
    (example_frames, example_label) = load_data_example()