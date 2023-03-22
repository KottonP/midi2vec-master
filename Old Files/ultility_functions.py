import numpy as np
import inspect

from keras.utils import plot_model
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
import math
from tensorflow import keras
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from scikeras.wrappers import KerasClassifier

# Source taken from https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb
K = keras.backend


class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)


def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10 ** -5, max_rate=10):
    init_weights = model.get_weights()
    iterations = math.ceil(len(X) / batch_size) * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.learning_rate)
    K.set_value(model.optimizer.learning_rate, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.learning_rate, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")


class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None, last_iterations=None, last_rate=None):
        super().__init__()
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.learning_rate, rate)


def sign_tests(network_1_accuracies, network_2_accuracies):
    # Calculate the p-value for the Student's t-test
    t_statistic, p_value_t = ttest_ind(network_1_accuracies, network_2_accuracies)

    # Calculate the p-value for the Mann-Whitney U test
    u_statistic, p_value_u = mannwhitneyu(network_1_accuracies, network_2_accuracies)

    # Calculate the p-value for the Wilcoxon signed-rank test
    z_statistic, p_value_z = wilcoxon(network_1_accuracies, network_2_accuracies)

    # Choose the test with the lowest p-value
    p_values = [p_value_t, p_value_u, p_value_z]
    best_test = np.argmin(p_values)

    # Print the chosen test and its p-value
    if best_test == 0:
        print('Student\'s t-test:', p_value_t)
    elif best_test == 1:
        print('Mann-Whitney U test:', p_value_u)
    elif best_test == 2:
        print('Wilcoxon signed-rank test:', p_value_z)


def get_model(clf: KerasClassifier, net_params: dict):
    clf_params = clf.get_params(deep=True)
    model = clf_params["model"]
    model_params = {key: net_params[key] for key in inspect.signature(model).parameters.keys()}
    return model(**model_params)

def get_model_plot(clf: KerasClassifier, net_params: dict):
    clf_params = clf.get_params(deep=True)
    model = clf_params["model"]
    model_params = {key: net_params[key] for key in inspect.signature(model).parameters.keys()}
    plot_model(model(**model_params))  # TODO: does not plot_model


def get_model(clf: KerasClassifier, net_params: dict):
    clf_params = clf.get_params(deep=True)
    model = clf_params["model"]
    model_params = {key: net_params[key] for key in inspect.signature(model).parameters.keys()}
    return model(**model_params)


def on_epoch_end(self, epoch, logs=None):  # start_from_epoch implementation due to older Keras version
    current = self.get_monitor_value(logs)
    if current is None or epoch < self.start_from_epoch:
        # If no monitor value exists or still in initial warm-up stage.
        return
    if self.restore_best_weights and self.best_weights is None:
        # Restore the weights after first epoch if no progress is ever made.
        self.best_weights = self.model.get_weights()


class EarlyStoppingUp(EarlyStopping):

    def __init__( self,
                  monitor="val_loss",
                  min_delta=0,
                  patience=0,
                  verbose=0,
                  mode="auto",
                  baseline=None,
                  restore_best_weights=False,
                  start_from_epoch=0):

        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)

        self.start_from_epoch = start_from_epoch

    def on_epoch_end(self, epoch, logs=None):  # start_from_epoch implementation due to older Keras version
        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()
