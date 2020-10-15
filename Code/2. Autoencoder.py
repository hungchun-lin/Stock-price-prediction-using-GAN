import pandas as pd
import numpy as np
import math
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import time

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_autoencoder(stock):
    dataset_total_feature = pd.read_csv("Finaldata_with_Fourier.csv")
    dataset_total_feature = dataset_total_feature.drop(['Date'], axis=1)

    dataset_total_feature.iloc[:, 1:] = pd.concat([dataset_total_feature.iloc[:, 1:].ffill(), dataset_total_feature.iloc[:, 1:].bfill()]).groupby(level=0).mean()

    def gelu(x):
        return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * math.pow(x, 3))))

    def relu(x):
        return max(x, 0)

    def lrelu(x):
        return max(0.01 * x, x)

    VAE_data = dataset_total_feature
    batch_size = 64
    n_batches = VAE_data.shape[0] / batch_size
    VAE_data = VAE_data.values

    num_training_days = int(dataset_total_feature.shape[0] * .7)
    print('Number of training days: {}. Number of test days: {}.'.format(num_training_days, \
                                                                         dataset_total_feature.shape[
                                                                             0] - num_training_days))

    train_iter = mx.io.NDArrayIter(data={'data': VAE_data[:num_training_days, :-1]},
                                   label={'label': VAE_data[:num_training_days, -1]}, batch_size=batch_size)
    test_iter = mx.io.NDArrayIter(data={'data': VAE_data[num_training_days:, :-1]},
                                  label={'label': VAE_data[num_training_days:, -1]}, batch_size=batch_size)

    model_ctx = mx.cpu()

    class VAE(gluon.HybridBlock):

        def __init__(self, n_hidden=400, n_latent=2, n_layers=1, n_output=784,
                     batch_size=100, act_type='relu', **kwargs):
            self.soft_zero = 1e-10
            self.n_latent = n_latent
            self.batch_size = batch_size
            self.output = None
            self.mu = None
            super(VAE, self).__init__(**kwargs)
            with self.name_scope():
                self.encoder = nn.HybridSequential(prefix='encoder')

            for i in range(n_layers):
                self.encoder.add(nn.Dense(n_hidden), nn.GELU())
                self.encoder.add(nn.Dense(n_latent * 2, activation=None))
                self.decoder = nn.HybridSequential(prefix='decoder')
            for i in range(n_layers):
                self.decoder.add(nn.Dense(n_hidden), nn.GELU())
                self.decoder.add(nn.Dense(n_output, activation='sigmoid'))

        def hybrid_forward(self, F, x):
            h = self.encoder(x)
            print(h)
            mu_lv = F.split(h, axis=1, num_outputs=2)
            mu = mu_lv[0]
            lv = mu_lv[1]
            self.mu = mu
            eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.n_latent), ctx=model_ctx)
            z = mu + F.exp(0.5 * lv) * eps
            y = self.decoder(z)
            self.output = y
            KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)
            logloss = F.sum(x * F.log(y + self.soft_zero) + (1 - x) * F.log(1 - y + self.soft_zero), axis=1)
            loss = -logloss - KL
            return loss

    n_hidden = 400
    n_latent = 2
    n_layers = 3
    n_output = VAE_data.shape[1] - 1

    net = VAE(n_hidden=n_hidden, n_latent=n_latent, n_layers=n_layers, n_output=n_output, batch_size=batch_size,
              act_type='relu')

    net.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .01})

    print(net)

    n_epoch = 150
    print_period = n_epoch // 10
    start = time.time()

    training_loss = []
    validation_loss = []
    for epoch in range(n_epoch):
        epoch_loss = 0
        epoch_val_loss = 0

        train_iter.reset()
        test_iter.reset()

        n_batch_train = 0
        for batch in train_iter:
            n_batch_train += 1
            data = batch.data[0].as_in_context(mx.cpu())

            with autograd.record():
                loss = net(data)
            loss.backward()
            trainer.step(data.shape[0])
            epoch_loss += nd.mean(loss).asscalar()

        n_batch_val = 0
        for batch in test_iter:
            n_batch_val += 1
            data = batch.data[0].as_in_context(mx.cpu())
            loss = net(data)
            epoch_val_loss += nd.mean(loss).asscalar()

        epoch_loss /= n_batch_train
        epoch_val_loss /= n_batch_val

        training_loss.append(epoch_loss)
        validation_loss.append(epoch_val_loss)

    end = time.time()
    print('Training completed in {} seconds.'.format(int(end - start)))

    vae_added_df = mx.nd.array(dataset_total_feature.iloc[:, :-1].values)
    print('The shape of the newly created (from the autoencoder) features is {}.'.format(vae_added_df.shape))

    pca = PCA(n_components=.8)
    x_pca = StandardScaler().fit_transform(vae_added_df.asnumpy())
    principalComponents = pca.fit_transform(x_pca)
    print("------ pca.n_components_ ------")
    print(pca.n_components_)
    #print(principalComponents)
    print(pca.explained_variance_ratio_)

    VAE_features = pd.DataFrame(principalComponents, columns = ['VAE_PCA_1', 'VAE_PCA_2', 'VAE_PCA_3', 'VAE_PCA_4'])
    print(VAE_features)
    #print(vae_added_df)

    return VAE_features

    '''
    import pickle
    f = open('./data/VAE/' + stock, 'wb')
    pickle.dump(vae_added_df, f)'''



VAE_features = get_autoencoder("Apple")
VAE_features.to_csv('VAE_fearues.csv', index=False)
