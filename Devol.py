from __future__ import print_function
import numpy as np
import csv
import operator
import gc
import os
from datetime import datetime
from keras.callbacks import EarlyStopping
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import log_loss  # cross entropy
import random as rand

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if K.backend() == 'tensorflow':
    import tensorflow as tf

__all__ = ['DEvol']
METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


class DEvol:
    def __init__(self, genome_handler, data_path=''):
        self.genome_handler = genome_handler
        self.datafile = data_path or (datetime.now().ctime() + '.csv')
        self._bssf = -1


    def set_objective(self, metric):
        if metric == 'acc':
            metric == 'accuracy'
        if metric not in ['accuracy', 'loss']:
            raise ValueError('Invalid metric')
        self._metric = metric
        self._objective = 'max' if self._metric == 'accuracy' else 'min'
        self._metric_index = 1 if self._metric == 'loss' else -1
        self._metric_op = METRIC_OPS[self._objective == 'max']
        self._metric_objective = METRIC_OBJECTIVES[self._objective == 'max']

    def run(self, dataset, num_generation, pop_size, epochs, fitness=None,
            metric='accuracy'):
        self.set_objective(metric)
        if len(dataset) == 2:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset
            self.x_val = None
            self.y_val = None
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test), (self.x_val, self.y_val) = dataset

        average_acc = []
        best_acc = []
        acc=[]
        members = self._generate_random_population(pop_size)
        pop, average_acc0, best_acc0,acc0 = self._evaluate_population(members, epochs, fitness, 0, num_generation)
        average_acc.append(average_acc0)
        best_acc.append(best_acc0)
        acc.append(acc0)

        # evolve
        for gen in range(1,num_generation):
            members = self._reproduce(pop,gen)
            pop, average_acc0, best_acc0,acc0 = self._evaluate_population(members, epochs, fitness, gen, num_generation)
            average_acc.append(average_acc0)
            best_acc.append(best_acc0)
            acc.append(acc0)

        average_acc = np.array(average_acc)
        best_acc = np.array(best_acc)
        acc = np.array(acc).reshape(pop_size,num_generation)

        return 'best-model.h5', average_acc, best_acc, acc

    def _reproduce(self,pop,gen):
        # pop represents original members
        members = []

        for _ in range(int(len(pop)*0.95)):
            members.append(self._crossover(pop.select(),pop.select()))

        members += pop.get_best(len(pop)-int(len(pop)*0.95))
        for imem, mem in enumerate(members):
            members[imem] = self._mutate(mem,gen)
        return members

    def _mutate(self,genome,generation):
        num_mutations = max(3, generation//4)
        return self.genome_handler.mutate(genome,num_mutations)

    def _crossover(self,genome1,genome2):
        cross_ind = rand.randint(0,len(genome1))
        child = genome1[:cross_ind]+genome2[cross_ind:]
        return child

    def _generate_random_population(self, size):
        return [self.genome_handler.generate() for _ in range(size)]

    def _evaluate_population(self, members, epochs, fitness, igen, ngen):
        fit = []
        for imem, mem in enumerate(members):
            self._print_evaluation(imem, len(members), igen, ngen)
            res = self._evaluate(mem, epochs)  # return: model loss accuracy of test data
            v = res[self._metric_index]  # v=acc
            del res
            fit.append(v)
        fit = np.array(fit)

        average_acc = np.mean(fit)
        best_acc = np.max(fit)

        self._print_result(fit, igen)
        return _Population(members, fit, fitness, obj=self._objective), average_acc, best_acc, fit

    def _print_evaluation(self, imod, nmod, igen, ngen):
        fstr = '\nmodel {0}/{1} - gneration {2}/{3}:\n'
        print(fstr.format(imod + 1, nmod, igen + 1, ngen))



    def _evaluate(self, genome, epochs):
        model = self.genome_handler.decode(genome)
        loss, accuracy = None, None

        fit_pramas = {
            'x': self.x_train,
            'y': self.y_train,
            'validation_split': 0.1,
            'verbose': 1,
            'callbacks': [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,verbose=1,
                                  mode='auto',min_delta=0.0001,cooldown=0,min_lr=0),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)],
            # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
            'epochs': epochs,
            'batch_size': 64

        }
        if self.x_val is not None:
            fit_pramas['validation_data'] = (self.x_val, self.y_val)
        try:
            model.fit(**fit_pramas)
            loss, accuracy = model.evaluate(self.x_test, self.y_test)
        except Exception as e:
            print('There is error in model')
            loss, accuracy = 0, 0

        self._record_state(model, genome, loss, accuracy)

        return model, loss, accuracy

    def _record_state(self, model, genome, loss, accuracy):

        met = loss if self._metric == 'loss' else accuracy
        if (self._bssf is -1 or
                self._metric_op(met, self._bssf) and
                accuracy is not 0):
            try:
                os.remove('best_model.h5')
                os.remove('best_gonome.npy')
            except OSError:
                pass
            self._bssf = met
            model.save('best_model.h5')
            best_gonome = np.array(genome)
            np.save('best_gonome.npy',best_gonome)

    def _print_result(self, fitness, generation):
        result_str = ('Generation: {3}, Best {4}: {0}'
                      'Average: {1},Std: {2}')
        print(result_str.format(self._metric_objective(fitness),
                                np.mean(fitness),
                                np.std(fitness),
                                generation + 1, self._metric))


class _Population(object):
    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        scores = fitnesses - fitnesses.min()
        if scores.max() > 0:
            scores /= scores.max()  #score normolization
        if obj == 'min':
            scores = 1 - scores
        if score:
            self.scores = score(scores)
        else:
            self.scores = scores
        self.s_fit = sum(self.scores)

    def get_best(self, n):
        combined = [(self.members[i], self.scores[i])
                    for i in range(len(self.members))]
        sorted(combined, key=(lambda x: x[1]), reverse=True)  # low to high (acc)
        return [x[0] for x in combined[:n]]

    def select(self):
        dart = rand.uniform(0, self.s_fit)  # the Pr of being chosen
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits >= dart:
                return self.members[i]