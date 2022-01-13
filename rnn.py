# train our sets 

from keras.models import Sequential 
from keras.layers import Conv2D, Dense, Dropout, Flatten, UpSampling2D, MaxPooling2D
from keras.layers import LSTM
from keras.callbacks import Callback

import random 
import glob
import wandb
from wandb.keras import WandbCallback

