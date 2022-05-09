"""Neural network feature preprocessors"""
from __future__ import annotations

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .layers import TransposedDense
from frlearn.base import FeaturePreprocessor, Unsupervised
from frlearn.statistics.feature_preprocessors import MaxAbsNormaliser


class SAE(Unsupervised, FeaturePreprocessor):
    """
    Implementation of the Shrink Auto-Encoder (SAE) preprocessor [1]_ that was used in [2]_.
    Trains an auto-encoder with additional L2-regularisation in the hidden layer,
    inducing the model to learn a latent representation
    that resembles a gaussian distribution centred around the origin.

    Parameters
    ----------
    lambd: float = 10
        Relative weight of L2-regularisation in the learning error.

    learning_rate: float = 0.01
        Initial learning rate of the optimiser.

    num_epochs: int = 1000
        Maximum number of epochs to train for.

    validation_freq: int = 5
        Frequency (in epochs) of validation.

    validation_perc: float = 0.2
        Size of the validation set.

    random_state : int or None = 0
        Random state to use.

    preprocessors : iterable = (MaxAbsNormaliser(), )
        Preprocessors to apply. The default max abs normaliser rescales all features
        to ensure that their values lie in [-1, 1].

    References
    ----------

    .. [1] `Cao VL, Nicolau M, McDermott J (2019).
       Learning neural representations for network anomaly detection.
       IEEE Transactions on Cybernetics, vol 49, no 8, pp 3074–3087.
       doi: 10.1109/TCYB.2018.2838668
       <https://ieeexplore.ieee.org/document/8386786>`_
    .. [2] `Lenz OU, Peralta D, Cornelis C (2021).
       Average Localised Proximity: A new data descriptor with good default one-class classification performance.
       Pattern Recognition, vol 118, no 107991.
       doi: 10.1016/j.patcog.2021.107991
       <https://www.sciencedirect.com/science/article/abs/pii/S0031320321001783>`_
    """

    def __init__(
            self,
            random_state: int = 0,
            lambd: float = 10,
            learning_rate: float = 0.01,
            num_epochs: int = 1000,
            validation_freq: int = 5,
            validation_perc: float = 0.2,
            preprocessors=(MaxAbsNormaliser())
    ):
        super().__init__(preprocessors=preprocessors)
        self.random_state = random_state
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.validation_freq = validation_freq
        self.validation_perc = validation_perc

    def _construct(self, X):
        tf.random.set_seed(self.random_state)
        model: SAE.Model = super()._construct(X)
        model.random_state = self.random_state

        # Network Parameters
        n_input = model.m
        n_latent = int(np.sqrt(n_input)) + 1
        n_h1 = int(np.round((2*n_input + n_latent)/3))
        n_h2 = int(np.round((n_input + 2*n_latent)/3))

        # Training Parameters
        num_train = int((1 - self.validation_perc)*model.n)
        batch_size = min(max(int(num_train / 20), 1), 100)
        patience = round(2000 / (self.validation_freq * num_train/batch_size))
        patience += 1  # Keras patience has an off-by-one error

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.001)

        shrink_loss = tf.keras.regularizers.l2(l=self.lambd/n_latent)

        original = keras.Input(shape=(n_input, ))
        layer_1 = layers.Dense(n_h1, activation='tanh', kernel_initializer='glorot_uniform')
        layer_2 = layers.Dense(n_h2, activation='tanh', kernel_initializer='glorot_uniform')
        layer_3 = layers.Dense(n_latent, activation='tanh', kernel_initializer='glorot_uniform', activity_regularizer=shrink_loss)
        latent = layer_3(layer_2(layer_1(original)))

        layer_3_d = TransposedDense(layer_3, n_h2, activation='tanh')
        layer_2_d = TransposedDense(layer_2, n_h1, activation='tanh')
        layer_1_d = TransposedDense(layer_1, n_input, activation='tanh')

        encoder = keras.Model(original, latent, name="encoder")

        reconstruction = layer_1_d(layer_2_d(layer_3_d(latent)))
        sae = keras.Model(original, reconstruction)

        sae.compile(
            optimizer=keras.optimizers.Adadelta(learning_rate=self.learning_rate),
            loss='mse'
        )
        sae.fit(
            X, X,
            epochs=self.num_epochs,
            batch_size=batch_size,
            validation_split=self.validation_perc,
            validation_freq=self.validation_freq,
            callbacks=[es]
        )

        model.encoder = encoder
        return model

    class Model(FeaturePreprocessor.Model):

        random_state: int
        encoder: keras.Model

        def transform(self, X):
            return self.encoder.predict(X)
