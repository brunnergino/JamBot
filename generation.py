from settings import *
from keras.models import load_model
import numpy as np
from numpy import array
import _pickle as pickle
import os

import data_processing
import chord_model
import midi_functions as mf
import data_class




chord_model_folder = 'models/chords/1523433134-Shifted_True_Lr_1e-05_EmDim_10_opt_Adam_bi_False_lstmsize_512_trainsize_4_testsize_1_samples_per_bar8/'
chord_model_name = 'model_Epoch10_4.pickle'

melody_model_folder = 'models/chords_mldy/Shifted_True_NextChord_True_ChordEmbed_embed_Counter_True_Highcrop_84_Lowcrop_24_Lr_1e-06_opt_Adam_bi_False_lstmsize_512_trainsize_4_testsize_1/'
melody_model_name = 'modelEpoch2.pickle'

midi_save_folder = 'predicted_midi/'

seed_path = 'data/' + shift_folder + 'indroll/'
seed_chord_path = 'data/' + shift_folder + 'chord_index/'

seed_name = 'Piano Concerto n2 op19 1mov.mid.pickle'


# Parameters for song generation:
BPM = 100
note_cap = 5
chord_temperature = 1

# Params for seed:
# length of the predicted song in bars:
num_bars =64
# The first seed_length number of bars from the seed will be used: 
seed_length = 4

#pred_song_length = 8*16-seed_length







with_seed = True

chord_to_index, index_to_chord = data_processing.get_chord_dict()

    
def sample_probability_vector(prob_vector):
    # Sample a probability vector, e.g. [0.1, 0.001, 0.5, 0.9]
            
    sum_probas = sum(prob_vector)
    
    
    if sum_probas > note_cap:
        prob_vector = (prob_vector/sum_probas)*note_cap
    
    note_vector = np.zeros((prob_vector.size), dtype=np.int8)
    for i, prob in enumerate(prob_vector):
        note_vector[i] = np.random.multinomial(1, [1 - prob, prob])[1]
    return note_vector

def ind_to_onehot(ind):
    onehot = np.zeros((len(ind), num_notes))
    for i, step in enumerate(ind):
        for note in step:
            onehot[i,note]=1
    return onehot

sd = pickle.load(open(seed_path+seed_name, 'rb'))[:8*seed_length]
seed_chords = pickle.load(open(seed_chord_path+seed_name, 'rb'))[:seed_length]

seed = ind_to_onehot(sd)[:,low_crop:high_crop]

print('loading polyphonic model ...')
melody_model = load_model(melody_model_folder+melody_model_name)
melody_model.reset_states()

ch_model = chord_model.Chord_Model(
        chord_model_folder+chord_model_name,
        prediction_mode='sampling',
        first_chords=seed_chords,
        temperature=chord_temperature)

chords = []

for i in range((num_bars+2)):
    ch_model.predict_next()


if chord_embed_method == 'embed':
    embedded_chords = ch_model.embed_chords_song(ch_model.song)
elif chord_embed_method == 'onehot':
    embedded_chords = data_class.make_one_hot_vector(ch_model.song, num_chords)
elif chord_embed_method == 'int':
    embedded_chords = [[x] for x in ch_model.song]


chords = []

for j in range((len(ch_model.song)-2)*fs*2):
    ind = int(((j+1)/(fs*2)))
    if next_chord_feature:
        ind2 = int(((j+1)/(fs*2)))+1
#        print(j)
#        print(ind, ' ', ind2)
        chords.append(list(embedded_chords[ind])+list(embedded_chords[ind2]))
    else:
        chords.append(embedded_chords[ind])

#    print(ind)

chords=np.array(chords)

if counter_feature:
    counter = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    counter = np.array(counter*(len(ch_model.song)-2))
    chords = np.append(chords, counter, axis=1)


seed = np.append(seed, chords[:seed.shape[0]], axis=1)    

seed = np.reshape(seed, (seed.shape[0], 1, 1, seed.shape[1]))

next_step = None

for step in seed:
    
    next_step = melody_model.predict(step)
    
    
notes = sample_probability_vector(next_step[0])

rest = []
rest.append(notes)


for chord in chords[seed.shape[0]:]:
    next_input = np.append(notes, chord, axis=0)
    next_input = np.reshape(next_input, (1, 1, next_input.shape[0]))
    next_step = melody_model.predict(next_input)
    notes = sample_probability_vector(next_step[0])
    rest.append(notes)

rest = np.array(rest)
rest = np.pad(rest, ((0,0),(low_crop,num_notes-high_crop)), mode='constant', constant_values=0)
ind = np.nonzero(rest)
#rest = np.reshape(rest, (rest.shape[1], rest.shape[0]))
#note_ind = mf.pianoroll_to_note_index(rest)
#print(ch_model.song)

instrument_names = ['Electric Guitar (jazz)', 'Acoustic Grand Piano',
'Bright Acoustic Piano', 'Electric Piano 1', 'Electric Piano 2', 'Drawbar Organ',
'Rock Organ', 'Church Organ', 'Reed Organ', 'Cello', 'Viola', 'Honky-tonk Piano', 'Glockenspiel',
'Percussive Organ', 'Accordion', 'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)', 'Electric Guitar (clean)',
'Electric Guitar (muted)', 'Overdriven Guitar', 'Distortion Guitar', 'Tremolo Strings', 'Pizzicato Strings',
'Orchestral Harp', 'String Ensemble 1', 'String Ensemble 2', 'SynthStrings 1', 'SynthStrings 2']

for instrument_name in instrument_names:
    
    mf.pianoroll_to_midi_continous(rest, midi_save_folder, instrument_name, instrument_name, BPM)
#    mf.pianoroll_to_midi(rest, 'test/midi/', instrument_name, instrument_name, BPM)
