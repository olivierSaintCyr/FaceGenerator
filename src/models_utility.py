import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

def make_n_optimizers(n:int):
  return [tf.keras.optimizers.Adam(2e-4) for i in range(n)]

def make_n_discriminator(n:int):
  return [make_discriminator_model() for i in range(n)]


class GradientTapes(object):
    """
        This is a way to make multiple gradient tape and use it in a 
        "with" statement
    """
  def __init__(self, n : int):
    self.tapes = [tf.GradientTape() for i in range(n)]

  def __enter__(self):
    self.tapes = [tape.__enter__() for tape in self.tapes]
    return self
  
  def __exit__(self, exc_type, exc_value, tb):
    for tape in self.tapes:
      if not tape.__exit__(exc_type, exc_value, tb):
        return False
    return True

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_normal",
                                   gamma_initializer="random_normal")
  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_normal",
                                   gamma_initializer="random_normal"))
                                   
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result