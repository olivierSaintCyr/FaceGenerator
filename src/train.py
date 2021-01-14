import tensorflow as tf
import glob
import imageio
import os
import PIL
import time
import tensorflow_addons as tfa
from GanModel import GanModel
import utility
from IPython import display
import pickle

import sys

########################################
config = tf.compat.v1.ConfigProto()    #
config.gpu_options.allow_growth = True #
tf.compat.v1.Session(config=config)    #
########################################

currentDir = os.getcwd().replace("source", "")
dataset_path = currentDir + '/dataset/'
checkpoint_dir = './training_checkpoints'

IMG_SIZE = (128,128,3)
N_DISCRIMINATORS = 5
LATENT_DIM = 128
BATCH_SIZE = 16
EPOCHS = 5

if __name__ == "__main__":

    real_images = tf.keras.preprocessing.image_dataset_from_directory(
        directory=dataset_path,
        label_mode=None,
        shuffle=True,
        seed=321,
        interpolation='bilinear',
        image_size=IMG_SIZE[0 : 2],
        batch_size=BATCH_SIZE)

    real_images = real_images.map(utility.preprocess_dataset)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    real_images = real_images.cache().prefetch(buffer_size=AUTOTUNE)
    
    resume_training = False
    if len(sys.argv) > 1:
        if list(sys.argv)[1] == 'resume':
            resume_training = True
    
    model = GanModel(IMG_SIZE, 
                     N_DISCRIMINATORS, 
                     LATENT_DIM, 
                     BATCH_SIZE, 
                     checkpoint_dir, 
                     reload_training=resume_training)
    
    model.train(real_images, EPOCHS)
    utility.make_gif('celebA_training_evo.gif')
