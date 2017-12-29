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
        else:
            tmp = []
            for char in num2word[int(label)]:
                tmp.append(ord(char) - ord('a'))
            wav_dict[os.path.join(path, wavfile)] = tmp + (5 - len(tmp)) * [-1]
    return wav_dict


def get_dataset(d, mode=0):
    """generate the dataset array from the dictionary
    Args:
        d: dictionary
        mode: 0 used for convnet, 1 used for ctc
    Returns:
        x: path array
        y: label array
    """
    x = []
    y = []
    if mode == 0:
        for path in d:
            x.append(path)
            y.append(d[path][0])
    elif mode == 1:
        for path in d:
            x.append(path)
            y.append(d[path])
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


def process_ctc(path):
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
    return python_speech_features.mfcc(content, sample_rate)


def batch2sparse(batch):
    """change a batch of target from dense tensor to sparse tensor
    Args:
        batch: a batch of targets
    Returns:
        indices, values, max_l for initialize the sparse tensor
        seq_len: sequence length
    """
    indices = []
    values = []
    max_l = 0
    for x in range(len(batch)):
        for y in range(len(batch[x])):
            if batch[x][y] == -1:
                break
            indices.append([x, y])
            values.append(batch[x][y])
            max_l = max(max_l, y + 1)
    dense_shape = [len(batch), max_l]
    return indices, values, dense_shape


# 具体数据格式参见以下链接
# https://github.com/jonrein/tensorflow_CTC_example/blob/master/bdlstm_train.py

def load_npy(mfcc_path, target_path):
    """load data in npy format
    Args:
        mfcc_path: path to mfcc file
        target_path: path to target npy file
    Returns:
        x: mfcc
        target: target
        seq_len: true sequence length of mfcc before padding
    """
    x = []
    target = []
    seq_len = []
    for p in os.listdir(mfcc_path):
        t_p = os.path.join(mfcc_path, p)
        mfcc = np.transpose(np.load(t_p))
        seq_len.append(len(mfcc))
        x.append(mfcc)

    for p in os.listdir(target_path):
        t_p = os.path.join(target_path, p)
        t = np.load(t_p)
        target.append(t)
    max_len = max(seq_len)
    for i in range(len(x)):
        x[i] = np.concatenate((x[i], np.zeros([max_len - len(x[i]), 26])), axis=0)
    return x, target, seq_len
