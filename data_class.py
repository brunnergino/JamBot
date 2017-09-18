# Author: Jonas Wiesendanger wjonas@student.ethz.ch
from settings import *
import numpy as np
import _pickle as pickle
import os
import midi_functions as mf


def get_chord_train_and_test_set(train_set_size, test_set_size):
    data = make_chord_data_set()
    train_set = data[:train_set_size]
    test_set = data[train_set_size:train_set_size+test_set_size]
    return train_set, test_set

def get_ind_train_and_test_set(train_set_size, test_set_size):
    data, chord_data = make_ind_data_set()
    train_set = data[:train_set_size]
    test_set = data[train_set_size:train_set_size+test_set_size]
    chord_train_set = chord_data[:train_set_size]
    chord_test_set = chord_data[train_set_size:train_set_size+test_set_size]
    return train_set, test_set, chord_train_set, chord_test_set 


def get_train_and_test_set(train_set_size, test_set_size):
    data = make_data_set()
    train_set = data[:train_set_size]
    test_set = data[train_set_size:train_set_size+test_set_size]
    return train_set, test_set


def make_data_sets(train_set_size, test_set_size):
    data = make_data_set()
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for song in data[:train_set_size]:
        if one_hot_input:
            x_song = []
            for chord in song[:-1]:
                x = [0]*num_chords
                x[chord] = 1
                x_song.append(x)
            X_train.append(x_song)
        else:
            X_train.append(song[:-1])
        y_song = []
        for chord in song[1:]:
            y = [0]*num_chords
            y[chord] = 1
            y_song.append(y)
        Y_train.append(y_song)
    for song in data[train_set_size:train_set_size+test_set_size]:
        if one_hot_input:
            x_song = []
            for chord in song[:-1]:
                x = [0]*num_chords
                x[chord] = 1
                x_song.append(x)
            X_test.append(x_song)
        else:
            X_test.append(song[:-1])
        y_song = []
        for chord in song[1:]:
            y = [0]*num_chords
            y[chord] = 1
            y_song.append(y)
        Y_test.append(y_song)
    return X_train, Y_train, X_test, Y_test


def make_data_location_list():
    data = []
    for path, subdirs, files in os.walk(chords_index_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            song = _path + _name
            data.append(song)
    return data

def load_data_set(data_string):
    data = []
    for path in data_string:
        song = pickle.load(open(path, 'rb'))
        data.append(song)
    return data


def make_chord_data_set():
    data = []
    for path, subdirs, files in os.walk(chords_index_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            song = pickle.load(open(_path + _name, 'rb'))
            data.append(song)
    return data


def make_data_set():
    data = []
    for path, subdirs, files in os.walk(tempo_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            pianoroll = mf.get_pianoroll(_name, _path, melody_fs)
            song = mf.pianoroll_to_note_index(pianoroll)
            data.append(song)
    return data

def make_ind_data_set():
    data = []
    chord_data = []
    for path, subdirs, files in os.walk(roll_folder):
        for name in files:
            _path = path.replace('\\', '/') + '/'
            _name = name.replace('\\', '/')
            song = pickle.load(open(_path + _name, 'rb'))
            _chord_path = _path.replace('indroll', 'chord_index')
            song_chords = pickle.load(open(_chord_path + _name, 'rb'))
            data.append(song)
            chord_data.append(song_chords)
    return data, chord_data


def make_one_hot_vector(song, num_chords):
    onehot_song = []
    for chord in song:
        onehot_chord = [0]*num_chords
        onehot_chord[chord] = 1
        onehot_song.append(onehot_chord)
    return onehot_song


def make_one_hot_note_vector(song, num_notes):
    onehot_song = []
    for step in song:
        onehot_step = [0]*num_notes
        for note in step:
            onehot_step[note] = 1
        onehot_song.append(onehot_step)
    return onehot_song



def truncate_pianoroll(pianoroll_set, max_notes):
    # Removes the highest note if more than max notes are played at the same time
    for i, song in enumerate(pianoroll_set['train']):
        for j, chord in enumerate(song):
            if len(pianoroll_set['train'][i][j]) > max_notes:
                pianoroll_set['train'][i][j] = pianoroll_set['train'][i][j][0:3]
            if len(pianoroll_set['train'][i][j]) < max_notes:
                for k in range(0,max_notes-len(pianoroll_set['train'][i][j])):
                    pianoroll_set['train'][i][j].append(0)

    for i, song in enumerate(pianoroll_set['test']):
        for j, chord in enumerate(song):
            if len(pianoroll_set['test'][i][j]) > max_notes:
                pianoroll_set['test'][i][j] = pianoroll_set['test'][i][j][0:3]
            if len(pianoroll_set['test'][i][j]) < max_notes:
                for k in range(0,max_notes-len(pianoroll_set['test'][i][j])):
                    pianoroll_set['test'][i][j].append(0)
    return pianoroll_set

def load_muse_pianoroll_data(data_path):
    file = open(data_path, 'rb')
    return pickle.load(file)


def max_length_pianoroll_set(pianoroll_set):
    max_len = 0
    for pianoroll in pianoroll_set:
        if len(pianoroll) > max_len:
            max_len = len(pianoroll)
    return max_len


def pianoroll_2_onehot_vector(pianoroll, vec_size):
    onehot_vector = np.zeros(vec_size)
    for note in pianoroll:
        onehot_vector[note] = 1
    return onehot_vector   

def pianoroll_2_onehot_matrix(pianoroll, vec_size):
    one_hot_matrix = np.zeros((len(pianoroll), vec_size), dtype=np.int16)
    for i, vec in enumerate(pianoroll):
        for note in vec:
            one_hot_matrix[i][note] = 1
    return one_hot_matrix


def pad_with_zeros(onehot_matrix, max_len, side):
    if side == 'right':
        return_matrix = np.pad(
            onehot_matrix, ((0, max_len-len(onehot_matrix)), (0, 0)),
            'constant', constant_values=((0, 0), (0, 0)))
    if side == 'left':
        return_matrix = np.pad(
            onehot_matrix, ((max_len-len(onehot_matrix), 0), (0, 0)),
            'constant', constant_values=((0, 0), (0, 0)))

    return return_matrix


def pianoroll_set_2_onehot_matrix_list(pianoroll_set, vec_size, padding,  pad_len):
    onehot_matrices = []
    for i, pianoroll in enumerate(pianoroll_set):
        onehot_matrix = pianoroll_2_onehot_matrix(pianoroll, vec_size)
        if padding is True:
            onehot_matrix = pad_with_zeros(onehot_matrix, pad_len, 'left')
        onehot_matrices.append(onehot_matrix)
        
    return onehot_matrices


def make_targets(X, vec_size, step):
    Y = []
    for x in X:
        y = np.zeros((len(x), vec_size))
        for i in range(len(x) - step):
            y[i] = x[i + step]
        Y.append(y)
    return Y


def make_targets2(X, vec_size, step):
    Y = []
    for x in X:
        y = [[0]*vec_size]*len(x)
        for i in range(len(x) - step):
            y[i] = x[i + step]
        Y.append(y)
    return Y


class Data:
    'contains the training and test sets'
    def __init__(self, data_path='data/muse/MuseData.pickle', vec_size=128,
                 source='pianoroll', padding=False, step=1, pad_len = 4500):
        if source is 'pianoroll':
            dataset_list = load_muse_pianoroll_data(data_path)
            pianoroll_trainset = np.array(dataset_list['train'])
            pianoroll_testset = np.array(dataset_list['test'])
            self.vec_size = vec_size
            self.step = step
            self.max_len = max(max_length_pianoroll_set(pianoroll_testset),
                               max_length_pianoroll_set(pianoroll_trainset))
            self.padding = padding
            self.pad_len = pad_len
            self.X_train = pianoroll_set_2_onehot_matrix_list(pianoroll_trainset, self.vec_size,
                                                              self.padding, self.pad_len)
            self.X_test = pianoroll_set_2_onehot_matrix_list(pianoroll_testset, self.vec_size,
                                                             self.padding, self.pad_len)
            self.Y_train = make_targets(self.X_train, self.vec_size, self.step)
            self.Y_test = make_targets(self.X_test, self.vec_size, self.step)
        elif source == 'midi':
            print('Not implemented yet')
        else:
            print('Unknown Source')


class Dataset:
    'contains X and y of the dataset'
    def __init__(self, dataset, lookback=1):
        dataX, dataY = np.ndarray(shape=()), np.array([])
        for t, song in enumerate(dataset):
            songX, songY = [], []
            for i in range(len(song)-lookback-1):
                a = song[i:(i+lookback), 0]
                songX.append(a)
                songY.append(song[i + lookback, 0])
            dataX[i] = songX
            dataY[i] = songY
        print(dataX.shape)
        self.X = dataX
        self.y = dataY

        


# Test Shit:
#pianoroll1 = [[3], [1, 4]]
#pianoroll2 = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
#pianoroll_set = [pianoroll1, pianoroll2]
#trainset = pianoroll_set_2_onehot_matrices(pianoroll_set, vec_size)
#print(pianoroll1)
#print(pianoroll2)
#print(trainset)