import time
import os
import pandas as pd
import tensorflow as tf
import numpy as np
#from main.feature import get_all_features
from tensorflow.keras.layers import LSTM, Reshape, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU
from tensorflow.keras import Sequential

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

def make_generator_model(input_dim, feature_size) -> tf.keras.models.Model:
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(input_dim, feature_size)))
    model.add(LSTM(16))
    model.add(Dense(units=7))
    model.compile(optimizer='adam', loss='mean_squared_error')
    #model.add(Reshape(7, 1))
    return model
model1 = make_generator_model(30,31)
print(model1.summary())

def make_discriminator_model() -> tf.keras.models.Model:
    cnn_net = tf.keras.Sequential()
    cnn_net.add(Conv1D(32, input_shape = (7,1), kernel_size=3, strides=2, activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Conv1D(64, kernel_size=3, strides=2, activation=LeakyReLU(alpha=0.01)))
    #cnn_net.add(BatchNormalization())
    # cnn_net.add(Conv1D(128, kernel_size=3, strides=2, activation=LeakyReLU(alpha=0.01)))
    #cnn_net.add(BatchNormalization())
    cnn_net.add(Flatten())
    #cnn_net.add(Dense(220, use_bias=False, activation=LeakyReLU(alpha=0.01)))
    cnn_net.add(Dense(32, use_bias=False, activation='relu'))
    cnn_net.add(Dense(1, activation = 'sigmoid'))
    return cnn_net

model = make_discriminator_model()
print(model.summary())

class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.batch_size = 128
        checkpoint_dir = '../training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)


    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


    @tf.function
    def train_step(self, real_x, real_y):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(real_x, training=True)
            generated_data = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1],1])
            real_y = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1],1])

            real_output = self.discriminator(real_y, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return {'d_loss': disc_loss, 'g_loss': gen_loss}

    def train(self, real_x, real_y, epochs):
        for epoch in range(epochs):
            start = time.time()

            loss = self.train_step(real_x, real_y)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print('epoch', epoch+1, 'd_loss', loss['d_loss'].numpy(), 'g_loss', loss['g_loss'].numpy())

            # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

if __name__ == '__main__':
    input_dim = 30
    feature_size = 26
    generator = make_generator_model(30, 26)
    discriminator = make_discriminator_model()
    gan = GAN(generator, discriminator)
    gan.train(X_train, y_train, 1000)