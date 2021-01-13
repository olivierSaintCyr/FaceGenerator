import tensorflow as tf
import pickle
import glob
import imageio
import tensorflow_docs.vis.embed as embed
import PIL

def reload_training_info(latent_dim:int=128 , reload_info=True):
  if reload_info:
    with open('train_info.pickle', 'rb') as f:
      return pickle.load(f)
  return TrainingInfo(latent_dim)


class TrainingInfo:
  def __init__(self, latent_dim : int):
    self.start_epoch = 0
    self.seed = tf.random.normal([num_examples_to_generate, latent_dim])

  def save(self):
    with open('train_info.pickle', 'wb') as f:
        pickle.dump(self, f)

def make_gif(gif_name : str, images_name : str = 'image*.png' ):
  with imageio.get_writer(gif_name, mode='I') as writer:
    filenames = glob.glob(images_name)
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
  embed.embed_file(gif_name)

def normalize_image(image):
  tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def preprocess_dataset(image):
    image = normalize_image(image)
    return image

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))