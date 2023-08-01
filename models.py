import tensorflow as tf

class Sampling(tf.keras.layers.Layer):

    def call(self, inputs):

        # Two inputs mean and variance
        z_mean, z_log_var = inputs

        # Batch and dimensions
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        # Normal distribution
        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 + z_log_var) * epsilon
    
def create_encoder(IMAGE_SIZE, EMBEDDING_DIM):

    """
    Encoder converts the images into latent vectors(embeddings)
    """

    encoder_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='encoder_input')

    x = tf.keras.layers.Conv2D(32, (3,3), strides=2, activation='relu', padding='same')(encoder_input)
    x = tf.keras.layers.Conv2D(64, (3,3), strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3,3), strides=2, activation='relu', padding='same')(x)

    # Decoder will need this
    shape_before_flattening = x.shape[1:]

    x = tf.keras.layers.Flatten()(x)

    z_mean = tf.keras.layers.Dense(EMBEDDING_DIM, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(EMBEDDING_DIM, name='z_log_var')(x)
    
    z = Sampling()([z_mean, z_log_var])

    return tf.keras.Model(encoder_input, [z_mean, z_log_var, z], name='encoder'), shape_before_flattening

def create_decoder(EMBEDDING_DIM, shape_before_flattening):

    """
    Decoder draws an image from a latent vector(embedding)
    """

    decoder_input = tf.keras.layers.Input(shape=(EMBEDDING_DIM,), name='decoder_input')

    x = tf.keras.layers.Dense(tf.prod(shape_before_flattening))(decoder_input)

    x = tf.keras.layers.Reshape(target_shape=shape_before_flattening)(x)

    x = tf.keras.layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3,3), strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(128, (3,3), strides=2, activation='relu', padding='same')(x)

    decoder_output = tf.keras.layers.Conv2D(3, (1,1), strides=1)(x)

    return tf.keras.Model(decoder_input, decoder_output, name='decoder')


class VAE(tf.keras.Model):

    def __init__(self, encoder, decoder, **kwargs):

        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder 
        self.total_loss_tracker = tf.keras.metrics.Mean(name = 'total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')

        self.BETA = 2000

        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):

        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]
    
    def call(self, inputs):

        """
        Call the model on a particular input
        """
        z_mean, z_log_var, z = self.encoder(inputs)

        reconstruction = self.decoder(z)

        return z_mean, z_log_var, reconstruction

    def train_step(self, data):

        """
        Step run during training
        """

        with tf.GradientTape() as tape:

            z_mean, z_log_var, reconstruction = self(data)

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(self.BETA * self.mse(data, reconstruction))

            # KL divergence loss
            kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))

            # Total loss
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)


        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result()
        }


