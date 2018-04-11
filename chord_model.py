from settings import *
from keras.models import load_model
import keras
import numpy as np
from numpy import array
import _pickle as pickle
from keras import backend as K

from data_processing import get_chord_dict


    
class Chord_Model:
    
    def __init__(self,
                 model_path,
                 prediction_mode='sampling',
                 first_chords=[1,3,2,1,1,3,2,1],
                 resample='none',
                 dim_factor=2,
                 temperature=1.0):
        
        print('loading chord model ...')
        
        self.model = keras.models.load_model(model_path)
        self.model.reset_states()
        self.embed_layer_output = K.function([self.model.layers[0].input], [self.model.layers[0].output])
        self.embed_model = keras.models.Model(inputs=self.model.input,outputs=self.model.get_layer(name="embedding").output)
        self.chord_to_index, self.index_to_chords = get_chord_dict()
        self.prediction_mode = prediction_mode
        self.temperature = temperature
        self.resample = resample
        self.dim_factor = dim_factor
        self.song = []
     
        for chord in first_chords[:-1]:
#            print(chord)
            self.model.predict(array([[chord]]))
            self.song.append(chord)
            
        chord = first_chords[-1]
    
        self.song.append(chord)
        self.current_chord = array([[chord]])
    
    
        
    def predict_next(self):
     
        prediction = self.model.predict(self.current_chord)[0]
        
        
        if self.resample=='hard':
            prediction[self.current_chord] = 0
            prediction = prediction/np.sum(prediction)
     
        elif self.resample=='soft':
            prediction[self.current_chord] /= self.dim_factor
            prediction = prediction/np.sum(prediction)
        
#        print(prediction)
        prediction = np.log(prediction) / self.temperature
        prediction = np.exp(prediction) / np.sum(np.exp(prediction))
      
     
        if self.prediction_mode == 'argmax':
#            print('argmax')
            while True:
                next_chord = np.argmax(prediction)
                if next_chord !=0:
                    break
#            print(next_chord)
       
        elif self.prediction_mode == 'sampling':
            while True:
                next_chord = np.random.choice(len(prediction), p=prediction)
#                print(next_chord)
                if next_chord !=0:
                    break

#            print(next_chord)
        
        self.song.append(next_chord)
        self.current_chord = np.array([next_chord])
        return self.current_chord[0]
    
    
    def embed_chord(self, chord):
        
        return self.embed_layer_output([[[chord]]])[0][0][0]
    
    
    def embed_chords_song(self, chords):
        
        embeded_chords = []
        
        for chord in chords:
            embeded_chords.append(self.embed_chord(chord))
            
        return embeded_chords
    
    
class Embed_Chord_Model:
    
    def __init__(self, model_path):
        
        print('loading chord model ...')
        
        model = keras.models.load_model(model_path)
        model.reset_states()
        self.embed_layer_output = K.function([model.layers[0].input], [model.layers[0].output])
        self.chord_to_index, self.index_to_chords = get_chord_dict()        
        
    
    
    def embed_chord(self, chord):
        
        return self.embed_layer_output([[[chord]]])[0][0][0]
    
    
    def embed_chords_song(self, chords):
        
        embeded_chords = []
        
        for chord in chords:
            embeded_chords.append(self.embed_chord(chord))
            
        return embeded_chords





if __name__=="__main__":
    # Paths:
    model_folder = 'models/chords/standart_lr_0.00003/'
    model_name = 'modelEpoch10'

    model = Chord_Model(model_folder + model_name + '.h5', prediction_mode='sampling')
    for i in range(0, 16):
        model.predict_next()
    print(model.song)
