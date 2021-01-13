import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

def make_n_optimizers(n:int):
  return [tf.keras.optimizers.Adam(2e-4) for i in range(n)]

def make_n_discriminator(image_size : int, n:int):
  return [make_discriminator(image_size) for i in range(n)]
  
def make_generator(latent_dim):
  inputs = tf.keras.layers.Input(shape=[latent_dim,])

  x = inputs
  x = layers.Dense(4*4*1024, use_bias=False)(x)
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU()(x)
  x = layers.Reshape((4, 4, 1024))(x)
  
  up_stack = [
          upsample(256, 4, True), # (None, 8, 8, 1024)
          upsample(256, 4, True), # (None, 16, 16, 1024)
          upsample(256, 4, True), # (None, 32, 32, 512)
          upsample(128, 4, True) # (None, 64, 64, 512)
  ]

  for up in up_stack:
    x = up(x)

  initializer = tf.random_normal_initializer(0., 0.02)

  x = tf.keras.layers.Conv2DTranspose(64, 4,
                                         strides=1,
                                         padding='same',
                                         kernel_initializer=initializer
                                          )(x)

  x = tfa.layers.InstanceNormalization(axis=3,
                                 center=True,
                                 scale=True,
                                 beta_initializer="random_normal",
                                 gamma_initializer="random_normal"
                                 )(x)

  x = tf.keras.layers.LeakyReLU()(x)

  last = tf.keras.layers.Conv2DTranspose(3, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')
  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x, name='generator')

def make_discriminator(image_size):
  initializer = tf.random_normal_initializer(0.,0.02)
  gen = tf.keras.layers.Input(shape=image_size, name='discriminator')

  x = gen

  down1 = downsample(16, 4, False)(x)
  down2 = downsample(32, 4)(down1)
  down3 = downsample(64, 4)(down2)
  
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)

  conv1 = tf.keras.layers.Conv2D(128, 4, strides=1,
                              kernel_initializer=initializer,
                              use_bias=False)(zero_pad1)
  # A fix
  batch_norm1 = tfa.layers.InstanceNormalization(axis=3, 
                                 center=True, 
                                 scale=True,
                                 beta_initializer="random_normal",
                                 gamma_initializer="random_normal"
                                 )(conv1)

  leaky_relu1 = tf.keras.layers.LeakyReLU()(batch_norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu1)
  
  conv2 = tf.keras.layers.Conv2D(128, 4, strides=1,
                              kernel_initializer=initializer,
                              use_bias=False)(zero_pad2)
  # A fix
  batch_norm2 = tfa.layers.InstanceNormalization(axis=3, 
                                 center=True, 
                                 scale=True,
                                 beta_initializer="random_normal",
                                 gamma_initializer="random_normal"
                                 )(conv2)

  leaky_relu2 = tf.keras.layers.LeakyReLU()(batch_norm2)

  zero_pad3 = tf.keras.layers.ZeroPadding2D()(leaky_relu2)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad3)
  
  return tf.keras.Model(inputs=gen, outputs=last, name='Dis')

def discriminator_loss(real_output, fake_output):
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  return cross_entropy(tf.ones_like(fake_output), fake_output)

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

def print_losses(losses_dict):
  for key, losses in losses_dict.items():
        tf.print(key, ' :', end=' ')
        for loss in losses:
          tf.print(loss, end=' ')
  tf.print(end='\r')

def average_loss(average_losses, current_losses, n):
  for loss in current_losses.keys():
    for i in range(len(average_losses[loss])):
      average_losses[loss][i] = (average_losses[loss][i]*(n-1) + current_losses[loss][i])/n
  return average_losses