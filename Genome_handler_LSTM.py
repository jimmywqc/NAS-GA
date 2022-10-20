import numpy as np
import random as rand
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Convolution1D,MaxPooling1D
from keras.layers import BatchNormalization
# from keras.layers.normalization.batch_normalization
class GenomeHandler:
    def __init__(self, max_lstm_layers, max_dense_layers,
                 max_dense_nodes, input_shape, n_classes,
                 batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None):

        if max_dense_layers < 1:
            raise ValueError(
                "At least one dense layer is required for softmax layer"
            )

        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.activation = activations or [
            'linear',
            'relu',
            'sigmoid',
            'tanh',
        ]
        self.lstm_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            # "activation",
            "dropout",
        ]
        self.dense_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout",
        ]
        self.layer_params = {
            "active": [0, 1],
            "num nodes": [2 ** i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            # "dropout": [(i if dropout else 0) for i in range(11)],
            "dropout": [0],
        }

        self.lstm_layers = max_lstm_layers
        self.lstm_layer_size = len(self.lstm_layer_shape)
        self.dense_layers = max_dense_layers - 1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes

    def lstmParam(self, i):
        key = self.lstm_layer_shape[i]
        return self.layer_params[key]

    def denseParam(self, i):
        key = self.dense_layer_shape[i]
        return self.layer_params[key]

    def mutate(self, genome, num_mutations):
        for i in range(num_mutations):
            index = np.random.choice(len(genome))
            if index < self.lstm_layers*self.lstm_layer_size:
                range_index = index % self.lstm_layer_size
                choice_range = self.lstmParam(range_index)
                genome[index] = np.random.choice(choice_range)
            elif index != len(genome)-1:
                offset = self.lstm_layers * self.lstm_layer_size
                new_index = index - offset
                range_index = new_index % self.dense_layer_size
                choice_range = self.denseParam(range_index)
                genome[index] = np.random.choice(choice_range)
            else:
                genome[index] = np.random.choice(list(range(len(self.optimizer))))

        return genome

    def decode(self, genome):
        if not self.is_compatible_genome(genome):
            raise ValueError('Invalid genome for specified configs')
        model = Sequential()
        input_layer = True
        offset = 0

        for i in range(self.lstm_layers):
            if genome[offset]:
                LSTM1 = None
                if input_layer:
                    LSTM1 = tf.keras.layers.Bidirectional(LSTM(genome[offset + 1],return_sequences=True, input_shape=self.input_shape))
                    # LSTM1 = LSTM(genome[offset + 1],return_sequences=True, input_shape=self.input_shape)
                    input_layer = False
                else:
                    LSTM1 = tf.keras.layers.Bidirectional(LSTM(genome[offset + 1],return_sequences=True))
                    # LSTM1 = LSTM(genome[offset + 1],return_sequences=True)
                model.add(LSTM1)
                # model.add(Activation(self.activation[genome[offset + 3]]))  #todo: not add activation

                if genome[offset + 2]:
                    model.add(BatchNormalization())

                if genome[offset + 4]:
                    model.add(Dropout(float(genome[offset + 4] / 20)))

            offset += self.lstm_layer_size

        if not input_layer:
            model.add( tf.keras.layers.Bidirectional(LSTM(64,return_sequences=False)))

        for i in range(self.dense_layers):
            if genome[offset]:
                dense = None
                if input_layer:
                    dense = Dense(genome[offset + 1], input_shape=self.input_shape)
                    input_layer = False
                else:
                    dense = Dense(genome[offset + 1])
                model.add(dense)
                model.add(Activation(self.activation[genome[offset + 3]]))
                if genome[offset + 2]:
                    model.add(BatchNormalization())
                if genome[offset + 4]:
                    model.add(Dropout(float(genome[offset + 4] / 20)))
            offset += self.dense_layer_size

        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer[genome[offset]],
                      metrics=['accuracy'])

        return model

    def genome_representation(self):
        encoding = []
        for i in range(self.lstm_layers):
            for key in self.lstm_layer_shape:
                encoding.append("Conv" + str(i) + " " + key)
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                encoding.append("Dense" + str(i) + " " + key)
        encoding.append("Optimizer")
        return encoding

    def generate(self):
        genome = []
        for i in range(self.lstm_layers):
            for key in self.lstm_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))
        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome[0] = 1
        return genome

    def is_compatible_genome(self, genome):
        expected_len = self.lstm_layers * self.lstm_layer_size \
                       + self.dense_layers * self.dense_layer_size + 1
        if len(genome) != expected_len:
            return False
        ind = 0
        for i in range(self.lstm_layers):
            for j in range(self.lstm_layer_size):
                if genome[ind + j] not in self.lstmParam(j):
                    return False
            ind += self.lstm_layer_size
        for i in range(self.dense_layers):
            for j in range(self.dense_layer_size):
                if genome[ind + j] not in self.denseParam(j):
                    return False
            ind += self.dense_layer_size
        if genome[ind] not in range(len(self.optimizer)):
            return False
        return True

    def best_genome(self, csv_path, metric="accuracy", include_metrics=True):
        best = max if metric is "accuracy" else min
        col = -1 if metric is "accuracy" else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])
        return genome

    def decode_best(self, csv_path, metric="accuracy"):
        return self.decode(self.best_genome(csv_path, metric, False))