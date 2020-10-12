import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from main.feature import get_all_features
from tensorflow.keras.layers import LSTM, Reshape, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU
from tensorflow.keras import Sequential
from pickle import load

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

def make_generator_model(input_dim, feature_size) -> tf.keras.models.Model:
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(input_dim, feature_size))) #, return_sequences=True,
    #model.add(LSTM(16))
    model.add(Dense(units=7))
    #model.compile(optimizer='adam', loss='mean_squared_error')
    #model.add(Reshape(7, 1))
    return model
model1 = make_generator_model(30,33)
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
        self.batch_size = 64
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
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1],1])
            real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1],1])

            real_output = self.discriminator(real_y_reshape, training=True)
            fake_output = self.discriminator(generated_data_reshape, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return real_y, generated_data, {'d_loss': disc_loss, 'g_loss': gen_loss}

    def train(self, real_x, real_y, epochs):
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_times'] = []
        train_hist['total_ptime'] = []


        for epoch in range(epochs):
            start = time.time()

            real_price, fake_price, loss = self.train_step(real_x, real_y)

            G_losses = []
            D_losses = []

            Real_price = []
            Predicted_price = []

            D_losses.append(loss['d_loss'].numpy())
            G_losses.append(loss['g_loss'].numpy())

            Predicted_price.append(fake_price.numpy())
            Real_price.append(real_price.numpy())

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print('epoch', epoch+1, 'd_loss', loss['d_loss'].numpy(), 'g_loss', loss['g_loss'].numpy())

            # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            # For printing loss
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - start
            train_hist['D_losses'].append(D_losses)
            train_hist['G_losses'].append(G_losses)
            train_hist['per_epoch_times'].append(per_epoch_ptime)
        # Reshape the predicted result & real
        Predicted_price = np.array(Predicted_price)
        Predicted_price = Predicted_price.reshape(Predicted_price.shape[1], Predicted_price.shape[2])
        Real_price = np.array(Real_price)
        Real_price = Real_price.reshape(Real_price.shape[1], Real_price.shape[2])

        #Predicted_price.to_csv('Predicted_price.csv')
        #Real_price.to_csv('Real_price.csv')
        print("REAL", Real_price.shape)
        print(Real_price)
        print("PREDICTED", Predicted_price.shape)
        print(Predicted_price)


        plt.plot(train_hist['D_losses'], label='D_loss')
        plt.plot(train_hist['G_losses'], label='G_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        #return Predicted_price, Real_price

if __name__ == '__main__':
    input_dim = 30
    feature_size = 33
    generator = make_generator_model(30, 33)
    discriminator = make_discriminator_model()
    gan = GAN(generator, discriminator)
    #Predicted_price, Real_price =
    gan.train(X_train, y_train, 100)

# %% --------------------------------------- Plot the result  -----------------------------------------------------------------

#Rescale back the real dataset
'''
X_scaler = load(open('X_scaler.pkl', 'rb'))
y_scaler = load(open('y_scaler.pkl', 'rb'))

dataset_train = pd.read_csv('dataset_train.csv', index_col=0)
#Predicted_price = pd.read_csv('Predicted_price.csv')
#Real_price = pd.read_csv('Real_price.csv')

rescaled_Real_price = y_scaler.inverse_transform(Real_price)
rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_price)

print("rescaled_Predicted data: ", rescaled_Predicted_price[0])
print("rescaled_Predicted shape: ", rescaled_Predicted_price.shape)
print("dataset_date_index: ", dataset_train.index[0:7])

predict_result = pd.DataFrame()
for i in range(rescaled_Predicted_price.shape[0]+1):

    y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns = ["predicted_price"], index = dataset_train.index[i:i+7])
    print("--y_predict--", y_predict)

    predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
    print("--predict_result--", predict_result)

real_price = pd.DataFrame()
for i in range(rescaled_Real_price.shape[0]):
    y_train = pd.DataFrame(rescaled_Real_price[i], columns = ["real_price"], index = dataset_train.index[i:i+7])
    real_price = pd.concat([real_price, y_train], axis=1, sort=False)

predict_result['mean'] = predict_result.mean(axis=1)
real_price['mean'] = real_price.mean(axis=1)


plt.figure(figsize=(16, 8))
plt.plot(real_price['mean'])
plt.plot(predict_result['mean'], color = 'r')
plt.xlabel("Date")
plt.ylabel("Stock price")
plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
plt.title("The result of Training", fontsize=20)
plt.show'''
