# Author: Jonas Wiesendanger wjonas@student.ethz.ch
from settings import *
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
# from keras.utils import to_categorical
from keras.utils import np_utils
from keras.layers.wrappers import Bidirectional
from random import shuffle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import _pickle as pickle

import data_class
import chord_model


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Uncomment next block if you only want to use a fraction of the GPU memory:

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#set_session(tf.Session(config=config))




# Path to the fully trained chord model for the chord embeddings:
chord_model_path = 'models/chords/1523433134-Shifted_True_Lr_1e-05_EmDim_10_opt_Adam_bi_False_lstmsize_512_trainsize_4_testsize_1_samples_per_bar8/model_Epoch10_4.pickle'
# Path where the polyphonic models are saved:
model_path = 'models/chords_mldy/'
model_filetype = '.pickle'


epochs = 100
train_set_size = 4
test_set_size = 1
test_step = 360          # Calculate error for test set every this many songs

verbose = False
show_plot = False
save_plot = True
lstm_size = 512
batch_size = 1
learning_rate = 1e-06
step_size = 1
save_step = 1
shuffle_train_set = True
bidirectional = False
embedding = False
optimizer = 'Adam'


fd = {'shifted': shifted, 'next_chord_feature': next_chord_feature, 'chord_embed_method': chord_embed_method, 'counter': counter_feature, 'highcrop': high_crop, 'lowcrop':low_crop, 'lr': learning_rate, 'opt': optimizer,
'bi': bidirectional, 'lstms': lstm_size, 'trainsize': train_set_size, 'testsize': test_set_size}
model_name = 'Shifted_%(shifted)s_NextChord_%(next_chord_feature)s_ChordEmbed_%(chord_embed_method)s_Counter_%(counter)s_Highcrop_%(highcrop)s_Lowcrop_%(lowcrop)s_Lr_%(lr)s_opt_%(opt)s_bi_%(bi)s_lstmsize_%(lstms)s_trainsize_%(trainsize)s_testsize_%(testsize)s' % fd

model_path = model_path + model_name + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path) 


print('loading data...')
# Get Train and test sets
train_set, test_set, chord_train_set, chord_test_set = data_class.get_ind_train_and_test_set(train_set_size, test_set_size)

if chord_embed_method == 'embed':
    chord_dim = chord_embedding_dim
elif chord_embed_method == 'onehot':
    chord_dim = num_chords
elif chord_embed_method == 'int':
    chord_dim = 1

if next_chord_feature:
    chord_dim = chord_dim*2

# Load model for chord embeddings
chord_embed_model = chord_model.Embed_Chord_Model(chord_model_path)



# Build Melody Model
print('creating model...')
model = Sequential()
# model.add(LSTM(lstm_size, batch_size=batch_size, input_shape=(step_size, new_num_notes+chord_dim+counter_size), stateful=True))
model.add(LSTM(lstm_size,  batch_input_shape=(batch_size,step_size, new_num_notes+chord_dim+counter_size), stateful=True))

model.add(Dense(new_num_notes))
model.add(Activation('sigmoid'))
if optimizer == 'RMS': optimizer = RMSprop(lr=learning_rate)
if optimizer == 'Adam': optimizer = Adam(lr=learning_rate)
loss = 'categorical_crossentropy'
model.compile(optimizer, loss)


# initialize loss arrays
total_test_loss_array = [] 
total_train_loss_array = []
total_test_loss = 0
total_train_loss = 0


# Test function
def test():
    print('\nTesting:')
    total_test_loss = 0

    bar = progressbar.ProgressBar(maxval=test_set_size, redirect_stdout=False)
    for i, test_song in enumerate(test_set):
        X_test, Y_test = make_feature_vector(test_song, chord_test_set[i], chord_embed_method)

        loss = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=verbose)
        model.reset_states()
        total_test_loss += loss
        bar.update(i)
    total_test_loss_array.append(total_test_loss/test_set_size)
    print('\nTotal test loss: ', total_test_loss/test_set_size)
    print('-'*50)
    plt.plot(total_test_loss_array, 'b-')
    plt.plot(total_train_loss_array, 'r-')
#    plt.axis([0, epochs, 0, 5])
    if show_plot: plt.show()
    if save_plot: plt.savefig(model_path+'plot.png')
    pickle.dump(total_test_loss_array,open(model_path+'total_test_loss_array.pickle', 'wb'))
    pickle.dump(total_train_loss_array,open(model_path+'total_train_loss_array.pickle', 'wb'))

# Make feature vectors with the notes and the chord information

def make_feature_vector(song, chords, chord_embed_method):
    
    if  next_chord_feature:
        X = np.array(data_class.make_one_hot_note_vector(song[:(((len(chords)-1)*fs*2)-1)], num_notes))
    else:
        X = np.array(data_class.make_one_hot_note_vector(song[:((len(chords)*fs*2)-1)], num_notes))
#    print(X.shape)
    X = X[:,low_crop:high_crop]
#    print(X.shape)
    if chord_embed_method == 'embed':
        X_chords = list(chord_embed_model.embed_chords_song(chords))
    elif chord_embed_method == 'onehot':
        X_chords = data_class.make_one_hot_vector(chords, num_chords)
    elif chord_embed_method == 'int':
        X_chords = [[x] for x in chords]
    X_chords_new = []
    Y = X[1:]
    
    for j, _ in enumerate(X):
        ind = int(((j+1)/(fs*2)))
        
        if next_chord_feature:
            ind2 = int(((j+1)/(fs*2)))+1
#            print(j)
#            print(ind, ' ', ind2)
#            print(X_chords[ind].shape)
            X_chords_new.append(list(X_chords[ind])+list(X_chords[ind2]))
        else:
            X_chords_new.append(X_chords[ind])
            
    X_chords_new = np.array(X_chords_new)    
    X = np.append(X, X_chords_new, axis=1)
    
        
    
    if counter_feature:
        counter = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
        if next_chord_feature:
            counter = np.array(counter*(len(X_chords)-1))[:-1]
        else:
            counter = np.array(counter*len(X_chords))[:-1]
        X = np.append(X, counter, axis=1)
    X = X[:-1]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    
    return X, Y

# Save Parameters to text file
with open(model_path + 'params.txt', "w") as text_file:
    text_file.write("Chord Model: %s" % chord_model_path + '\n')
    text_file.write("epochs: %s" % epochs + '\n')
    text_file.write("train_set_size: %s" % train_set_size + '\n')
    text_file.write("test_set_size: %s" % test_set_size + '\n')
    text_file.write("lstm_size: %s" % lstm_size + '\n')
    text_file.write("learning_rate: %s" % learning_rate + '\n')
    text_file.write("save_step: %s" % save_step + '\n')
    text_file.write("shuffle_train_set: %s" % shuffle_train_set + '\n')
    text_file.write("test_step: %s" % test_step + '\n')
    text_file.write("bidirectional: %s" % bidirectional + '\n')
    text_file.write("num_chords: %s" % num_chords + '\n')
    text_file.write("chord_n: %s" % chord_n + '\n')


# Train model
print('training model...')
for e in range(1, epochs+1):
    
    print('Epoch ', e, 'of ', epochs, 'Epochs\nTraining:')
    
    # Shuffle training set order
    if shuffle_train_set:
        # Zip lists together an shuffle and unzip again
        ziperoni = list(zip(train_set, chord_train_set))
        shuffle(ziperoni)
        train_set, chord_train_set = zip(*ziperoni)

    bar = progressbar.ProgressBar(maxval=train_set_size)
    
    # Train model with each song seperatly
    for i, song in enumerate(train_set):
        X, Y = make_feature_vector(song, chord_train_set[i], chord_embed_method)
        hist = model.fit(X, Y, batch_size=batch_size, shuffle=False, verbose=verbose)
        model.reset_states()
        bar.update(i)
        total_train_loss += hist.history['loss'][0]
        if (i+1)%test_step is 0:
            total_train_loss = total_train_loss/test_step
            total_train_loss_array.append(total_train_loss)
            test()
            total_train_loss = 0

    if e%save_step is 0:
        print('saving model')
        model_save_path = model_path + 'model' + 'Epoch' + str(e) + model_filetype
        model.save(model_save_path)


