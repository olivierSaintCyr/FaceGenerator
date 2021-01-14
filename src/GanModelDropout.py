from GanModel import GanModel

def make_discriminator_model(image_size):
    initializer = tf.random_normal_initializer(0.,0.02)
    gen = tf.keras.layers.Input(shape=image_size, name='discriminator')

    x = gen

    down1 = downsample(16, 4, False)(x) # (None, 64, 64, 16)
    down2 = downsample(32, 4)(down1) # (None, 32, 32, 32)
    down3 = downsample(64, 4)(down2) # (None, 16, 16, 64)
  
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)

    drop1 = tf.keras.layers.Dropout(0.5)(zero_pad1)
    
    conv2 = tf.keras.layers.Conv2D(128, 4, strides=1,
                              kernel_initializer=initializer,
                              use_bias=False)(drop1)
    # A fix
    batch_norm2 = tfa.layers.InstanceNormalization(axis=3, 
                                 center=True, 
                                 scale=True,
                                 beta_initializer="random_normal",
                                 gamma_initializer="random_normal"
                                 )(conv2)

    leaky_relu2 = tf.keras.layers.LeakyReLU()(batch_norm2)

    drop2 = tf.keras.layers.Dropout(0.5)(leaky_relu2)
    
    flat = tf.keras.layers.Flatten()(drop2)
    dense = tf.keras.layers.Dense(1)(flat)
  
    return tf.keras.Model(inputs=gen, outputs=dense, name='Dis')

class GanModelDropout(GanModel):
    @staticmethod
    def make_discriminators(image_size, n):
        return [make_discriminator_model(image_size) for i in range(n)]