import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
 
from models_utility import make_generator, make_n_discriminator, make_n_optimizers
from utility import reload_training_info
########################################
config = tf.compat.v1.ConfigProto()    #
config.gpu_options.allow_growth = True #
tf.compat.v1.Session(config=config)    #
########################################

class GanModel(object):
    def __init__(self, img_size, n_discriminators, latent_dim, reload_model=False):
        self.img_size = img_size    # (width, lenght, RGB)
        
        self.generator = make_generator()
        self.discriminators = make_n_discriminator(n_discriminators)

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4)
        self.discriminator_optimizers = make_n_optimizers(n_discriminators)
        
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizers,
                                        generator=self.generator,
                                        discriminators=self.discriminators)

        self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, 
                                                       checkpoint_path, 
                                                       max_to_keep=5)
        
        self.train_info = reload_training_info(latent_dim=latent_dim, reload_info=reload_model)

        if reload_model and self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
    
# need to fix attributes for the method below
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
  
        with tf.GradientTape() as gen_tape, GradientTapes(len(discriminators)) as discriminators_tape:
        
            generated_images = generator(noise, training=True)

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
        
        gradient_of_gen = gen_tape.gradient(losses_gen, generator.trainable_variables)

        gradient_of_discs = []
        for disc, discriminator_tape, loss_disc in zip(discriminators, discriminators_tape.tapes, loss_discs):
          gradient_of_discs.append(discriminator_tape.gradient(loss_disc, disc.trainable_variables))
        
        generator_optimizer.apply_gradients(zip(gradient_of_gen, generator.trainable_variables))

        for discriminator_optimizer, gradient_of_disc, disc in zip(discriminator_optimizers, gradient_of_discs, discriminators):
          discriminator_optimizer.apply_gradients(zip(gradient_of_disc, disc.trainable_variables))
    
    def train(data):
        start_epoch = training_info.start_epoch
        for epoch in range(epochs):
            start_time = time.time()

            for image_batch in dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            self.generate_and_save_images(generator,
                                    epoch + start_epoch + 1,
                                    training_info.seed)

            #Save the model every 5 epochs
            if (epoch + 1) % 5 == 0:
            print('Saving model..')
            self.checkpoint.save(file_prefix = checkpoint_prefix)
            self.training_info.start_epoch = start_epoch + epoch + 1
            self.training_info.save()
            
            print ('Time for epoch {} is {} sec'.format(epoch + start_epoch + 1, 
                                                        time.time()-start_time))
        
        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epochs,
                                training_info.seed)

    def generate_and_save_images(self, epoch):
        predictions = self.generator(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            image = (predictions[i, :, :, :] + 1)*0.5
            image = tf.reshape(image, (self.img_size))
            plt.imshow(image)
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        #plt.show()
        plt.close()



