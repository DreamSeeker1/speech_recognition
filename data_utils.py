import tensorflow as tf
import os
import scipy.io.wavfile as sio
import python_speech_features
from sphfile import SPHFile
import re
import pickle
import numpy as np

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


def convert_wav(path):
    """convert nist sphere file to wav file
    Args:
        path: path to dialect folder in timit directory
    """
    speaker_list = os.listdir(path)
    speaker_list = list(map(lambda x: os.path.join(path, x), speaker_list))
    for speaker in speaker_list:
        # list of files in a speaker folder
        sentence_list = list(map(lambda x: os.path.join(speaker, x), os.listdir(speaker)))
        # a folder to store the converted files
        wav_folder = os.path.join(speaker, 'wav_files')
        if os.path.exists(wav_folder):
            continue
        os.makedirs(wav_folder)
        for f in sentence_list:
            if re.match(r'.*\.WAV', f):
                name = f.split(os.path.sep)[-1]
                sph = SPHFile(f)
                sph.write_wav(os.path.join(wav_folder, name))


def build_sentence_dict(path, dict_path):
    """build dictionary mapping sentence to sentence type and number
    Args:
        path: path to prompts.txt
        dict_path: path to store the dictionary
    """

    tmp_set = set([])
    for i in range(26):
        tmp_set.add(chr(97 + i))
    # blank
    tmp_set.add(' ')
    # prime
    tmp_set.add('\'')
    # unknown
    tmp_set.add('U')

    chr2idx = {}
    idx2chr = {}

    for idx, ch in enumerate(tmp_set):
        idx2chr[idx] = ch
        chr2idx[ch] = idx

    sentence_dict = {}

    with open(path) as f:
        for line in f:
            m = re.match(r'(.*)\((.*)\)', line)
            if m:
                content = m.group(1).lower()
                idx = m.group(2)
                if idx in sentence_dict:
                    continue
                else:
                    seq = []
                    u_idx = chr2idx['U']
                    for ch in content:
                        seq.append(chr2idx.get(ch, u_idx))
                    sentence_dict[idx] = seq
    with open(dict_path, 'wb') as f:
        pickle.dump((chr2idx, idx2chr, sentence_dict), f)


def get_dataset_timit(path_list, path_dict):
    """prepare the dataset for training
    Args:
        path_list: list of path to dialect folder in timit directory
        path_dict: path to sentence dict
    Returns:
        x: mfcc of wav files
        y: target sequence
    """
    speaker_list = []
    for path in path_list:
        speaker_list += list(map(lambda _: os.path.join(path, _), os.listdir(path)))
    with open(path_dict, 'rb') as f:
        chr2idx, idx2chr, sentence_dict = pickle.load(f)
    x = []
    y = []
    seq_length = []
    for speaker in speaker_list:
        wav_path = os.path.join(speaker, 'wav_files')
        wav_files = os.listdir(wav_path)
        for t in wav_files:
            file_name = re.match(r'(.*).WAV', t).group(1)
            content = sio.read(os.path.join(wav_path, t))
            sample_rate = content[0]
            mfcc = python_speech_features.mfcc(content[1], sample_rate)
            x.append(mfcc)
            y.append(sentence_dict[file_name.lower()])
            seq_length.append(len(mfcc))
    return x, y, seq_length


def get_batch(batchsize, x, y, seq_length):
    """generate batch given x, y and seq_length
    Args:
        batchsize: batchsize
        x: mfcc feature sequence
        y: target sequence
        seq_length: sequence length of the mfcc features
    Yields:
        res_x: padded mfcc batch
        sparse: sparse representation of target sequence
        t_seq: true sequence of the input sequence
    """
    pad = [0. for _ in range(len(x[0][0]))]
    for i in range(len(x) // batchsize):
        t_x = x[i * batchsize: (i + 1) * batchsize]
        t_y = y[i * batchsize: (i + 1) * batchsize]
        t_seq = seq_length[i * batchsize: (i + 1) * batchsize]
        max_l = max(t_seq)
        res_x = []
        for mfcc in t_x:
            mfcc = np.concatenate((mfcc, np.tile(pad, (max_l - len(mfcc), 1))), axis=0)
            res_x.append(mfcc)
        sparse = batch2sparse(t_y)
        yield res_x, sparse, t_seq

    t_x = x[-batchsize:]
    t_y = y[-batchsize:]
    t_seq = seq_length[-batchsize:]
    max_l = max(t_seq)
    res_x = []
    for mfcc in t_x:
        mfcc = np.concatenate((mfcc, np.tile(pad, (max_l - len(mfcc), 1))), axis=0)
        res_x.append(mfcc)
    sparse = batch2sparse(t_y)
    yield res_x, sparse, t_seq


def to_sentence(token_seq, idx2chr):
    """convert the sentence sequence in tokens to letters
    Args:
        token_seq: token sequence
        idx2chr: dictionary mapping tokens to letters
    Returns:
        sentence: reconstructed sentence
    """
    res = []
    for i in token_seq:
        res.append(idx2chr[i])
    sentence = ''.join(res).strip('v')
    return sentence
