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
    model = GanModel(IMG_SIZE, N_DISCRIMINATORS, LATENT_DIM, BATCH_SIZE, checkpoint_dir, True)
    for i in range(10):
        latent_vector = tf.random.normal([1, LATENT_DIM])
        img = (model.generate_image(latent_vector) + 1)*127.5
        img = tf.cast(img, tf.uint8)[0]

        img = Image.fromarray(img.numpy())
        img.save(os.path.join("generated_images/test_image" + str(i) + ".jpeg"))