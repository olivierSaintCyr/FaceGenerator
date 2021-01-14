import tensorflow as tf
from GanModel import GanModel
from train import IMG_SIZE, N_DISCRIMINATORS, LATENT_DIM, BATCH_SIZE
from train import currentDir, checkpoint_dir
from PIL import Image
import numpy as np
import os

########################################
config = tf.compat.v1.ConfigProto()    #
config.gpu_options.allow_growth = True #
tf.compat.v1.Session(config=config)    #
########################################


if __name__ == "__main__":
    
    model = GanModel(IMG_SIZE, 
                     N_DISCRIMINATORS, 
                     LATENT_DIM, 
                     BATCH_SIZE, 
                     checkpoint_dir, 
                     True)
    
    model.save_generator('models/', 'generator')
    model.save_discriminators('models/', 'discriminator')