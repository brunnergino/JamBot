#Imports
from settings import *
import numpy as np
import midi_functions as mf
import _pickle as pickle
import os
import sys
import pretty_midi as pm
import mido


#p = pickle.load(open(path + name + '.pickle', 'rb'))
##pp = pm.PrettyMIDI(tempo_folder+name).get_piano_roll(fs = fs)
#x = [[1,0,0,0,0,1,0,0],[1,0,0,0,0,1,0,0],[0,0,1,1,0,1,0,0],[0,0,0,1,0,1,0,0],[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],
#     [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0] ,[0,0,0,0,1,1,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], 
#     [0,0,1,0,0,0,1,0],[0,0,0,1,0,0,1,0] ,[0,0,0,0,1,1,1,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,1,0]]
#x = np.array(x)
#x = np.reshape(x,(x.shape[1], x.shape[0]))
#
#histo = pickle.load(open(histo_folder + '/miditest.mid.pickle', 'rb'))
#chords = pickle.load(open(chords_folder + '/miditest.mid.pickle', 'rb'))



def shift_midi(shift, name, tempo_path, target_path):
    
    midi = pm.PrettyMIDI(tempo_path + name)
    
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch -= shift
    
    midi.write(target_path + name)




def chords_to_index(chords,chord_to_index):
    chords_index = []
    for chord in chords:
        if chord in chord_to_index:
            chords_index.append(chord_to_index[chord])
        else:
            chords_index.append(chord_to_index[UNK])
    return chords_index

def chords_to_index_save(name, chords_folder, chords_index_folder, chord_to_index):
    chords = pickle.load(open(chords_folder + name, 'rb'))
    chords_index = chords_to_index(chords, chord_to_index)
    pickle.dump(chords_index,open(chords_index_folder + name , 'wb'))


def histo_to_key(histo, key_n):
    max_n = histo.argsort(axis=0)[-key_n:]
    max_n.sort()
    return tuple(max_n)


def histo_to_chords(histo, chord_n):
    max_n = histo.argsort(axis=0)[-chord_n:]
    chords = []
    for i in range(0,max_n.shape[1]):
        chord = []
#        print('chord: ')
        for note in max_n[:,i]:
#            print(note)
            if histo[note,i] != 0:
                chord.append(note)
        chord.sort()
#        print(chord)
        chords.append(tuple(chord))
#    print(chords)
    return chords

def print_song(song):
    for step in song:
        notes = []
        for note in step:
            notes.append(pm.note_number_to_name(note))
        print(notes)
    


def pianoroll_to_note_index(pianoroll):
    note_ind = []
    for i in range(0,pianoroll.shape[1]):
        step = []
        for j, note in enumerate(pianoroll[:,i]):
#            print(note, '  ', i, '    ', pianoroll[note,i])
            if note != 0:
#                print('note!')
                step.append(j)
        note_ind.append(tuple(step))
#    print(chords)
    return note_ind


def load_histo_save_song_histo(name, histo_path, song_histo_path):
    histo = pickle.load(open(histo_path + name, 'rb'))
    song_histo = np.sum(histo, axis=1)
    pickle.dump(song_histo, open(song_histo_path + name , 'wb'))


def load_histo_save_chords(chord_n, name, histo_path, chords_path):
    histo = pickle.load(open(histo_path + name, 'rb'))
    chords = histo_to_chords(histo, chord_n)
    pickle.dump(chords,open(chords_path + name , 'wb'))


def pianoroll_to_histo_bar(pianoroll, samples_per_bar):
    # Make histogramm for every samples_per_bar samples
    histo_bar = np.zeros((pianoroll.shape[0], int(pianoroll.shape[1]/samples_per_bar)))
    for i in range(0,pianoroll.shape[1]-samples_per_bar+1,samples_per_bar):
    #    print(i/samples_per_bar)
    #    print('i: ',i)
    #    print(i+samples_per_bar)
        histo_bar[:,int(i/samples_per_bar)] = np.sum(pianoroll[:,i:i+samples_per_bar], axis=1)
    return histo_bar


def pianoroll_to_histo_song(pianoroll, samples_per_bar):
    # Make histogramm for every samples_per_bar samples
    histo_song = np.zeros((pianoroll.shape[0], int(pianoroll.shape[1]/samples_per_bar)))
    for i in range(0,pianoroll.shape[1]-samples_per_bar+1,samples_per_bar):
    #    print(i/samples_per_bar)
    #    print('i: ',i)
    #    print(i+samples_per_bar)
        histo_bar[:,int(i/samples_per_bar)] = np.sum(pianoroll[:,i:i+samples_per_bar], axis=1)
    return histo_bar


def histo_bar_to_histo_oct(histo_bar, octave):
    histo_oct = np.zeros((octave, histo_bar.shape[1]))
    for i in range(0, histo_bar.shape[0]-octave+1, octave):
#        print('i: ',i)
#        print(i+octave)
        histo_oct = np.add(histo_oct, histo_bar[i:i+octave])
    return histo_oct

    
def save_pianoroll_to_histo_oct(samples_per_bar,octave, name, path, histo_path):
#    print(path + name)
    pianoroll = pickle.load(open(path + name, 'rb'))
    histo_bar = pianoroll_to_histo_bar(pianoroll, samples_per_bar)
    histo_oct = histo_bar_to_histo_oct(histo_bar, octave)
    pickle.dump(histo_oct,open(histo_path + name , 'wb'))


def midi_to_histo_oct(samples_per_bar,octave, fs, name, path, histo_path):
#    print(path + name)
    pianoroll = get_pianoroll(name, path, fs)
    histo_bar = pianoroll_to_histo_bar(pianoroll, samples_per_bar)
    histo_oct = histo_bar_to_histo_oct(histo_bar, octave)
    pickle.dump(histo_oct,open(histo_path + name + '.pickle' , 'wb'))


def save_pianoroll(name, path, target_path, fs):
    mid = pm.PrettyMIDI(path + name)
    p = mid.get_piano_roll(fs=fs)
    for i, _ in enumerate(p):
        for j, _ in enumerate(p[i]):
            if p[i, j] != 0:
                p[i,j] = 1
#    print(np.argwhere(p[:,:]))
    pickle.dump(p,open(target_path + name + '.pickle', 'wb'))


def double_sample(mid):
    p_double = mid.get_piano_roll(fs=fs*sample_factor)
    p = []
    for i in range(0,p_double.shape[1], sample_factor):
        vec = np.sum(p_double[:,i:(i+sample_factor)], axis=1)
#        print(vec[36])
        
        p.append(vec)
    p = np.array(p)
#    print(p[0,36])
    p = np.transpose(p)
#    print(p[36,0])
#    for i, _ in enumerate(p):
#        for j, _ in enumerate(p[i]):
#            if p[i, j] != 0:
#                p[i,j] = 1
#    n_double = mf.pianoroll_to_note_index(p_double)
#    n_new = mf.pianoroll_to_note_index(p)
    return p



def save_note_ind(name, path, target_path, fs):
    mid = pm.PrettyMIDI(path + name)
    if double_sample_notes:
        p = double_sample(mid)
    else:
        p = mid.get_piano_roll(fs=fs)
    for i, _ in enumerate(p):
        for j, _ in enumerate(p[i]):
            if p[i, j] != 0:
                p[i,j] = 1
    n = mf.pianoroll_to_note_index(p)
#    print(np.argwhere(p[:,:]))
    pickle.dump(n,open(target_path + name + '.pickle', 'wb'))


def get_notes(name, path, fs):
    mid = pm.PrettyMIDI(path + name)
    if double_sample_notes:
        p = double_sample(mid)
    else:
        p = mid.get_piano_roll(fs=fs)
    return p

    
def get_pianoroll(name, path, fs):
    p = get_notes(name, path, fs)
    for i, _ in enumerate(p):
        for j, _ in enumerate(p[i]):
            if p[i, j] != 0:
                p[i,j] = 1
#    print(np.argwhere(p[:,:]))
    return p


def pianoroll_to_midi(pianoroll, midi_folder, filename, instrument_name, bpm):
    if not os.path.exists(midi_folder):
        os.makedirs(midi_folder) 
    midi = pm.PrettyMIDI(initial_tempo=bpm, resolution=200)    
    midi.time_signature_changes.append(pm.TimeSignature(4, 4, 0))
    piano_program = pm.instrument_name_to_program(instrument_name)
    piano = pm.Instrument(program=piano_program)
    ind = np.nonzero(pianoroll)
    for i in range(ind[0].shape[0]):
        note = pm.Note(velocity=80, pitch=ind[1][i], start=(60/(2*bpm))*ind[0][i], end=(60/(2*bpm))*ind[0][i] + 0.25)
        piano.notes.append(note)
    midi.instruments.append(piano)
#    print(midi.get_tempo_changes())
    midi.write(midi_folder + filename+'.mid')
    
    
def pianoroll_to_midi_continous(pianoroll, midi_folder, filename, instrument_name, bpm):
    if not os.path.exists(midi_folder):
        os.makedirs(midi_folder)
    midi = pm.PrettyMIDI(initial_tempo=bpm, resolution=200)    
    midi.time_signature_changes.append(pm.TimeSignature(4, 4, 0))
    piano_program = pm.instrument_name_to_program(instrument_name)
    piano = pm.Instrument(program=piano_program)
        
    tracker = []
    start_times  = dict()
    for i, note_vector in enumerate(pianoroll):
        notes = list(note_vector.nonzero()[0])
#        print('notes',notes)
        removal_list = []
        for note in tracker:
            if note in notes and (i)%8 is not 0:
#                print('removing', note, 'from notes')
                notes.remove(note)
            else:
                midi_note = pm.Note(velocity=80, pitch=note, start=(60/(2*bpm))*start_times[note], end=(60/(2*bpm))*i)
                piano.notes.append(midi_note)
#                print('removing', note, 'from tracker')
                removal_list.append(note)
        for note in removal_list:
            tracker.remove(note)
#        print('tracker',tracker)
#        print('notes',notes)

        for note in notes:
            tracker.append(note)
            start_times[note]=i
#        print('tracker',tracker)
#        print('-'*50)
    midi.instruments.append(piano)
#    print(midi.get_tempo_changes())
    midi.write(midi_folder + filename+'.mid')



        

def print_type_of_folder(folderpath):
    filenames = os.listdir(data_path)
    for filename in filenames:
        print(get_type(folderpath + filename))


def get_type(filepath):
    return mido.MidiFile(filepath).type


def change_tempo_folder(data_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path) 
    for path, subdirs, files in os.walk(data_path):
        print(path)
        print(subdirs)
        print(files)
        for name in files:
            print('Name: ', name)
            change_tempo2(name, path, target_path)

def myround(x, base):
    return int(base * round(float(x)/base))

def change_tempo(filename, data_path, target_path):
    mid = mido.MidiFile(data_path + filename)
    new_mid = mido.MidiFile()
    new_mid.ticks_per_beat = mid.ticks_per_beat
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            new_msg = msg.copy()
            if new_msg.type == 'set_tempo':
                new_msg.tempo = 500000
#            if msg.type == 'note_on' or msg.type == 'note_off':
            if discretize_time:
                print(msg.time)
                new_msg.time = myround(msg.time, base=mid.ticks_per_beat/(discritezition/4) )
#                msg.time = myround(msg.time, base=mid.ticks_per_beat/(discritezition/4) )
            if offset_time:
#                print('first:', time)
                
                print((mid.ticks_per_beat/(offset/4)))
                new_msg.time = int(msg.time + mid.ticks_per_beat/(offset))
#                print('second:', new_time)
#                print('diff:',time )
#            msg.time = time
            new_track.append(new_msg)
        new_mid.tracks.append(new_track)
    new_mid.save(target_path + filename)

                
def change_tempo2(filename, data_path, target_path):
    mid = mido.MidiFile(data_path + filename)
    new_mid = mido.MidiFile()
    new_mid.ticks_per_beat = mid.ticks_per_beat
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        new_mid.tracks.append(new_track)
        for msg in track:
            if msg.type == 'set_tempo':
                print(msg)
                msg.tempo = 500000
                print(msg)
                
            new_track.append(msg)
    new_mid.save(target_path + filename)
                
def get_ticks_per_beat(data_path):
    filenames = os.listdir(data_path)
    for filename in filenames:
        try:
            print( MidiFile(data_path + filename).ticks_per_beat)
        except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
            exception_str = 'Unexpected error in ' + filename  + ':\n', e, sys.exc_info()[0]
            print(exception_str)   


def get_tempi_of_folder(data_path):
    filenames = os.listdir(data_path)
    for filename in filenames:
        try:
            print(filename, ':\n', pm.PrettyMIDI(data_path + filename).get_tempo_changes())
        except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
            exception_str = 'Unexpected error in ' + filename  + ':\n', e, sys.exc_info()[0]
            print(exception_str)

def get_time_signature_of_folder(data_path):
    filenames = os.listdir(data_path)
    for filename in filenames:
        try:
            print(pm.PrettyMIDI(data_path + filename).time_signature_changes)
        except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
            exception_str = 'Unexpected error in ' + filename  + ':\n', e, sys.exc_info()[0]
            print(exception_str)

def create_tempo_histogram(data_path):
    invalid_midi_files = np.chararray([])
    exception_str_arr= np.chararray([])
    tempo_array = np.array([])
    filenames = os.listdir(data_path)
    num_files = len(filenames)
    for i, filename in enumerate(filenames):
        print('file ', i, 'of ', num_files)
        try:
            tempo = pm.PrettyMIDI(data_path + filename).estimate_tempo()
            tempo_array = np.append(tempo_array, tempo)
        except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError) as e:
            exception_str = 'Unexpected error in ' + filename  + ':\n', e, sys.exc_info()[0]
            print(exception_str)
            invalid_midi_files = np.append(invalid_midi_files, filename)
            exception_str_arr = np.append(exception_str_arr, exception_str)
    tempo_histogram = np.histogram(tempo_array)
    return tempo_histogram, invalid_midi_files, exception_str_arr
