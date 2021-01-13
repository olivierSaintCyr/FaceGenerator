import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import os

from IPython import display

from models_utility import make_generator, make_n_discriminator, make_n_optimizers
from models_utility import print_losses, average_loss, generator_loss, discriminator_loss, GradientTapes
from utility import reload_training_info

########################################
config = tf.compat.v1.ConfigProto()    #
config.gpu_options.allow_growth = True #
tf.compat.v1.Session(config=config)    #
########################################

class GanModel(object):
    def __init__(self, img_size, n_discriminators, latent_dim, batch_size, checkpoint_path, reload_training=False):
        self.IMG_SIZE = img_size    # (width, lenght, RGB)
        self.BATCH_SIZE = batch_size
        self.LATENT_DIM = latent_dim
        
        self.checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")

        self.generator = make_generator(latent_dim)
        self.discriminators = make_n_discriminator(img_size, n_discriminators)

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4)
        self.discriminator_optimizers = make_n_optimizers(len(self.discriminators))
        
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizers,
                                        generator=self.generator,
                                        discriminators=self.discriminators)

        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, 
                                                       checkpoint_path, 
                                                       max_to_keep=5)
        
        self.train_info = reload_training_info(latent_dim=latent_dim, reload_info=reload_training)

        if reload_training and self.checkpoint_manager.latest_checkpoint:
            print("Model restored at epoch : ", self.train_info.start_epoch)
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
    
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.LATENT_DIM])
  
        with tf.GradientTape() as gen_tape, GradientTapes(len(self.discriminators)) as discriminators_tape:
        
            generated_images = self.generator(noise, training=True)

            real_outputs = []
            fake_outputs = []
            for discriminator in self.discriminators:
                real_outputs.append(discriminator(images, training=True))
                fake_outputs.append(discriminator(generated_images, training=True))

            losses_gen = []
            loss_discs = []
            for real_output, fake_output in zip(real_outputs, fake_outputs):
                losses_gen.append(generator_loss(fake_output))
                loss_discs.append(discriminator_loss(real_output, fake_output))

            gradient_of_gen = gen_tape.gradient(losses_gen, self.generator.trainable_variables)

            gradient_of_discs = []
            for disc, discriminator_tape, loss_disc in zip(self.discriminators, discriminators_tape.tapes, loss_discs):
                gradient_of_discs.append(discriminator_tape.gradient(loss_disc, disc.trainable_variables))

            self.generator_optimizer.apply_gradients(zip(gradient_of_gen, self.generator.trainable_variables))

            for discriminator_optimizer, gradient_of_disc, disc in zip(self.discriminator_optimizers, gradient_of_discs, self.discriminators):
                discriminator_optimizer.apply_gradients(zip(gradient_of_disc, disc.trainable_variables))

        return {'losses_gen': losses_gen, 'loss_discs' : loss_discs}
        
    def train(self, dataset, epochs):
        start_epoch = self.train_info.start_epoch
        for epoch in range(epochs):
            start_time = time.time()
            n = 1
            for image_batch in dataset:
                losses = self.train_step(image_batch)
                
                if n >= 2:
                    average_losses = average_loss(average_losses, losses, n)
                else:
                    average_losses = losses
                
                if n % 5 == 0:
                    print_losses(average_losses)
                n += 1
            
            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            self.generate_and_save_images(epoch + start_epoch + 1)

            #Save the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                print('Saving model..')
            self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            self.train_info.start_epoch = start_epoch + epoch + 1
            self.train_info.save()
            
            print ('Time for epoch {} is {} sec'.format(epoch + start_epoch + 1, 
                                                        time.time()-start_time))

    def generate_and_save_images(self, epoch):
        predictions = self.generator(self.train_info.seed, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            image = (predictions[i, :, :, :] + 1)*0.5
            image = tf.reshape(image, (self.IMG_SIZE))
            plt.imshow(image)
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    def generate_image(self, latent_vector):
        return self.generator(latent_vector, training=False)

    def save_discriminators(self, path_to_location, name):
        for i in range(len(self.discriminators)):
            self.discriminators.save(path_to_location + name + str(i))

    def save_generator(self, path_to_location, name):
        self.generator.save(path_to_location + name)
