import tensorflow as tf
import os
import re
import scipy.io.wavfile as sio
import numpy as np
import python_speech_features

max_length = 13554
num2word = ['zero', 'one', 'two', 'three', 'four',
            'five', 'six', 'seven', 'eight', 'nine']


def get_files(path, coding_method=0):
    """a function mapping the wavfiles' path to it contents
    Args:
        path: path to the wav files
        coding_method: how to encode the number, 0 not coding,
            1 for one-hot embedding, 2 for ctc.
    Returns:
        wav_dict: dictionary mapping file path to it's content
    """
    files = os.listdir(path)
    wav_dict = {}
    for wavfile in files:
        t = re.match(r"(\d+)_(\w+)_(.*)", wavfile)
        label = t.group(1)
        if coding_method == 0:
            wav_dict[os.path.join(path, wavfile)] = label
        elif coding_method == 1:
            wav_dict[os.path.join(path, wavfile)] = tf.keras.utils.to_categorical(label, num_classes=10)
    return wav_dict


def get_dataset(d):
    """generate the dataset array from the dictionary
    Args:
        d: dictionary
    Returns:
        x: path array
        y: label array
    """
    x = []
    y = []
    for path in d:
        x.append(path)
        y.append(d[path][0])
    return x, np.array(y)


def process(path):
    """decode the wav file and pad zeros
    Args:
        path: path to the wav file
    Returns:
        decoded files
    """
    content = sio.read(path)
    sample_rate = content[0]
    content = content[1] / 255.
    content = np.concatenate([content, (max_length - len(content)) * [0]])
    return np.expand_dims(python_speech_features.mfcc(content, sample_rate), axis=2)



