JamBot: Music Theory Aware Chord Based Generation of Polyphonic Music with LSTMs

Instructions

Preperation:
Make shure you have the following packages installed:
keras, tensorflow, numpy, pretty_midi, mido, progressbar2, matplotlib, h5py

Data Processing:
Put the MIDI dataset in the data/original folder.
Run data_processing.py to adjust the tempo and shift the midi songs, extract the chords and piano rolls. This might take some time.
There may be some error messages printed due to invalid MIDI files.

Training:
Run chord_lstm_training.py to train the chord LSTM.
Adjust the chord_model_path string in polyphonic_lstm_training.py to point it to a trained chord LSTM model in models/ (for the chord embeddings), and run it to train the polyphonic LSTM.

Generating
Adjust the chord_model_folder, chord_model_name, melody_model_folder, melody_model_name strings in generate.py to point them to the trained chord and polyphonic LSTMs model files in models/.
Adjust the seed_path, seed_chord_path and seed_name in generate.py to point it to the extracted chords (in the data/shifted/chord_index folder) and piano roll (in the data/shifted/indroll folder) of the desired seed.
Adjust the BPM, note_cap, and chord_temperature parameters if desired and run generate.py to generate a song. The song will be saved with a few different instrumentiations in midi_save_folder (default is predicted_midi/).