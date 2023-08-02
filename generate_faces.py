from models import *
import config
import numpy as np
import matplotlib.pyplot as plt
import zipfile

# Extract the zipfile
zip_ref = zipfile.ZipFile('img_align_celeba.zip')
zip_ref.extractall()
zip_ref.close()

# Load the dataset from directory
train_data = tf.keras.utils.image_dataset_from_directory('./img_align_celeba/',
                                                         labels=None,
                                                         color_mode = 'rgb',
                                                         image_size= (config.IMAGE_SIZE, config.IMAGE_SIZE),
                                                         batch_size = config.BATCH_SIZE,
                                                         shuffle=True,
                                                         seed=42,
                                                         interpolation='bilinear')

def preprocess(img):

    return tf.cast(img, 'float32') / 255.0

train_data = train_data.map(lambda x: preprocess(x))

# Build the encoder
encoder, shape_before_flattening = create_encoder(config.IMAGE_SIZE, config.EMBEDDING_DIM)

# Build the decoder
decoder = create_decoder(config.EMBEDDING_DIM, shape_before_flattening)

# Build the variational autoencoer
vae = VAE(encoder=encoder, decoder=decoder)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
vae.compile(optimizer=optimizer)

# Fit the model
history = vae.fit(train_data, epochs = 5)

# Generate faces
grid_width, grid_height = (10, 3)

z_sample = np.random.normal(size=(grid_width * grid_height, config.EMBEDDING_DIM))

reconstructions = vae.decoder.predict(z_sample)

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(grid_width * grid_height):
	ax = fig.add_subplot(grid_height, grid_width, i+1)
	ax.axis('off')
	ax.imshow(reconstructions[i,:,:])
plt.show()