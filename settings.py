

shifted = True

shift_folder = ''
if shifted:
    shift_folder = 'shifted/'

# If you only want to process a subfolder like '/A' or '/A/A' for tests
subfolder = '/'

source_folder = 'data/original' + subfolder
tempo_folder1 = 'data/'  + 'tempo' + subfolder
histo_folder1 = 'data/'  + 'histo' + subfolder


tempo_folder2 = 'data/' + shift_folder + 'tempo' + subfolder
shifted_folder = 'data/' + shift_folder + 'shifted' + subfolder
pickle_folder = 'data/' + shift_folder + 'pianoroll' + subfolder
roll_folder = 'data/' + shift_folder + 'indroll' + subfolder
histo_folder2 = 'data/' + shift_folder + 'histo' + subfolder
chords_folder = 'data/' + shift_folder + 'chords' + subfolder
chords_index_folder = 'data/' + shift_folder + 'chord_index' + subfolder
song_histo_folder = 'data/' + shift_folder + 'song_histo' + subfolder



# Test Paths:
#source_folder = 'data/test'
#tempo_folder = 'data/' + shift_folder + 'test_tempo'
#pickle_folder = 'data/' + shift_folder + 'test_pianoroll'
#roll_folder = 'data/' + shift_folder + 'test_indroll'
#histo_folder = 'data/' + shift_folder + 'test_histo'
#chords_folder = 'data/' + shift_folder + 'test_chords'
#chords_index_folder = 'data/' + shift_folder + 'test_chord_index'
#song_histo_folder = 'data/' + shift_folder + 'test_song_histo'
#shifted_folder = 'data/' + shift_folder + 'test_shifted'


dict_path = 'data/'
chord_dict_name = 'chord_dict.pickle'
index_dict_name = 'index_dict.pickle'

if shifted:
    chord_dict_name = 'chord_dict_shifted.pickle'
    index_dict_name = 'index_dict_shifted.pickle'


# Specifies the method how to add the chord information to the input vector
# 'embed' uses the chord embedding of the chord model
# 'onehot' encodes the chord as one hot vector
# 'int' just appends the chord id to the input vector
chord_embed_method = 'embed'

# Adds the count of the beat as a feature to the input vector
counter_feature = True
counter_size = 0
if counter_feature:
    counter_size = 3
    
# Appends also the next cord to the feature vector:
next_chord_feature = True

high_crop = 84#84
low_crop = 24#24
num_notes = 128
new_num_notes = high_crop - low_crop
chord_embedding_dim = 10

#double_sample_chords = False
double_sample_notes = True

sample_factor = 2

one_hot_input = False
collapse_octaves = True
discretize_time = False
offset_time = False
discritezition = 8
offset = 16

# Some parameters to extract the pianorolls
# fs = 4 for 8th notes
fs = 4
samples_per_bar = fs*2
octave = 12
melody_fs = 4


# Number of notes in extracted chords
chord_n = 3
# Number of notes in a key
key_n = 7
# Chord Vocabulary size
num_chords = 100

if shifted:
    num_chords = 50

UNK = '<unk>'


# Some Chords:
C = tuple((0,4,7))
Cm = tuple((0,3,7))
Csus4 = tuple((0,5,7))
Csus6 = tuple((0,7,9))
Dm = tuple((2,5,9))
D = tuple((2,6,9))
Dsus4 = tuple((2,7,9))
Em = tuple((4,7,11))
E = tuple((4,8,11))
F = tuple((0,5,9))
Fm = tuple((0,5,8))
G = tuple((2,7,11))
Gm = tuple((2,7,10))
Gsus4 = tuple((0,2,7))
Am = tuple((0,4,9))
Asus7 = tuple((4,7,9))
A = tuple((1,4,9))
H = tuple((3,6,11))
Hverm = tuple((2,5,11))
Hm = tuple((2,6,11))
B = tuple((2,5,10))
Es = tuple((3,7,10))
As = tuple((0,3,8))
Des = tuple((1,5,8))
Fis = tuple((1,6,10))



