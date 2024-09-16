import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from numpy.random import seed, default_rng
import numpy as np
from miscellaneous.misc import Misc
from miscellaneous.set_seed import set_seed

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

SEED = 1000000

tf.config.run_functions_eagerly(False)
 
### Warning ###
import warnings
warnings.filterwarnings('ignore')

# For reproducibility
rnd_seed = 1000000
default_rng(rnd_seed)
tf.random.set_seed(
    rnd_seed
)

gpu_available = tf.test.is_gpu_available()

if gpu_available:
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))


class MLP():
    def __init__(self, X_data, y_data, floor_config):
        
        self.X_data = X_data
        self.y_data = y_data
        self.floor_config = floor_config
        self.classes_floor = np.shape(self.y_data)[1]
        self.method = None

    # Model to classify the fingerprints into floor
    def floor_model(self):
        with tf.device('/device:GPU:0'):    
            tf.keras.backend.clear_session()
            gc.collect()
            set_seed(SEED)
            self.fl_model = Sequential()
            self.fl_model.add(Dense(128, input_shape=(np.shape(self.X_data)[1],), activation='relu'))
            self.fl_model.add(Dense(64, activation='relu'))
            self.fl_model.add(Dense(32, activation='relu'))
            self.fl_model.add(Dense(16, activation='relu'))
            self.fl_model.add(Dense(8, activation='relu'))
            self.fl_model.add(Dense(self.classes_floor, activation='softmax'))

    def train(self):    
        with tf.device('/device:GPU:0'):
            tf.keras.backend.clear_session()
            gc.collect()
            set_seed(SEED)
           
            misc = Misc()

         
            # ---------------------------------------- Floor --------------------------------------------
            if self.floor_config['train'] == True:
                print("--------- FLOOR CLASSIFICATION -----------")

                self.floor_model()
                optimizer = Adam(self.floor_config['lr'], self.floor_config['momentum'])
                self.fl_model.compile(loss=self.floor_config['loss'], optimizer=optimizer, metrics=['accuracy'])

                if (np.shape(self.X_data)[0] * self.floor_config['validation_split']) > 1:
                    floor_history = self.fl_model.fit(self.X_data, self.y_data, 
                                                        epochs=self.floor_config['epochs'], 
                                                        validation_split = self.floor_config['validation_split'],
                                                        verbose=1)
                else:
                    floor_history = self.fl_model.fit(self.X_data, self.y_data, 
                                                    epochs=self.floor_config['epochs'],
                                                    verbose=1)

            
            
            return self.fl_model