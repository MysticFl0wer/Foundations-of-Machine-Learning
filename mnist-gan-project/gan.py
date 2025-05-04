import keras
from keras import layers
import keras.datasets
import keras.datasets.mnist
from keras.layers import Dense, Flatten, Reshape
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def build_generator():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=100))
    model.add(Dense(28 * 28, activation='sigmoid'))
    model.add(Reshape((28, 28)))  # Reshape output to a 28x28 image
    return model

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 image into a 1D array
    model.add(Dense(128, activation='relu'))  # Dense layer with 128 neurons and ReLU activation
    model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron, output is between 0 and 1 (real or fake)
    return model  # Return the discriminator model

generator = build_generator()  
generator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Instantiate the generator
discriminator = build_discriminator()  # Instantiate the discriminator

# Compile the discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Set optimizer, loss function, and metrics
# Combine the models to create the GAN
discriminator.trainable = False  # Freeze the discriminator during generator training
gan = Sequential([generator, discriminator])  # Create the GAN model by stacking generator and discriminator
gan.compile(optimizer='adam', loss='binary_crossentropy')  # Set optimizer and loss function for the GAN
# Training parameters.  Parameters such as the number of epochs, batch size, and noise dimension must defined to control the training process.
epochs = 151  # Number of training epochs
batch_size = 60000  # Number of samples per gradient update
noise_dim = 100  # Dimension of the noise vector input to the generator
# Function to generate random noise. Noise Generation: A function to create random noise vectors is provided. This noise serves as input to the generator to create fake images.
def generate_noise(batch_size, noise_dim):
    return np.random.normal(0, 1, (batch_size, noise_dim))  # Generate random noise from a normal distribution

# Placeholder for real data (replace this with actual training data)
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
real_data = np.array(train_images).reshape(-1, 28, 28)  # Reshape to 3D array: [samples, time steps, features]

# Function to visualize generated images. we visualize with The plot_generated_images function to generate images from the generator to create images from noise, reshaping them, and displaying them in a grid.
def plot_generated_images(generator, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = generate_noise(examples, noise_dim)  # Generate noise for the specified number of examples
    generated_images = generator.predict(noise)  # Use the generator to produce images from noise
    generated_images = generated_images.reshape(examples, 28, 28)  # Reshape the generated images to 28x28
    plt.figure(figsize=figsize)  # Set the figure size for the plot
    for i in range(examples):  # Loop through each generated image
        plt.subplot(dim[0], dim[1], i + 1)  # Create subplots
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')  # Display the image in grayscale
        plt.axis('off')  # Hide axis ticks and labels
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f'gan_generated_epoch_{epoch}.png')  # Save the generated image as a PNG file
    plt.show()  # Show the generated images

for epoch in range(epochs):  # Loop through each epoch
    # Generate fake images
    noise = generate_noise(batch_size, noise_dim)  # Generate noise for a batch
    fake_images = generator.predict(noise)  # Generate fake images using the generator
    # Train the discriminator on real and fake data
    d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))  # Train on real images with label 1
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))  # Train on fake images with label 0
    # Train the generator (via the GAN model, which tries to fool the discriminator)
    noise = generate_noise(batch_size, noise_dim)  # Generate new noise for the generator
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))  # Train the generator to make the discriminator predict 1
    # Print progress and visualize images every 100 epochs
    if epoch % 5 == 0:  # Every 5 epochs
        d_loss_real[0] = round(d_loss_real[0], 4)
        d_loss_fake[0] = round(d_loss_fake[0], 4)
        print(f'Epoch {epoch}, D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss:.4f}')
        plot_generated_images(generator, epoch)  # Call the visualization function

