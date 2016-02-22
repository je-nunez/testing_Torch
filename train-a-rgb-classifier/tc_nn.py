#!/usr/bin/env python2

"""Testing a NN with flash cards."""

import logging
import itertools
import numpy as np
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten


class NNHyperParameters(object):
    """Container class for some NN hyper-parameters."""

    available_keras_optimizers = []
    available_keras_objectives = []
    available_keras_activations = []
    available_keras_initializations = []
    available_keras_border_modes = []

    @classmethod
    def get_available_optimizers(cls):
        """Get all the available optimizers available from Theano or
        TensorFlow through Keras."""

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        cls.available_keras_optimizers.append(sgd)

        rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        cls.available_keras_optimizers.append(rms_prop)

        ada_grad = Adagrad(lr=0.01, epsilon=1e-06)
        cls.available_keras_optimizers.append(ada_grad)

        ada_delta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        cls.available_keras_optimizers.append(ada_delta)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        cls.available_keras_optimizers.append(adam)

    @classmethod
    def get_available_objectives(cls):
        """Get all the available objectives available from Theano or
        TensorFlow through Keras."""

        cls.available_keras_objectives = ['mean_squared_error',
                                          'root_mean_squared_error',
                                          'mean_absolute_error',
                                          'mean_absolute_percentage_error',
                                          'mean_squared_logarithmic_error',
                                          'squared_hinge',
                                          'hinge',
                                          'binary_crossentropy',
                                          'poisson_loss']

    @classmethod
    def get_available_activations(cls):
        """Get all the available activations available from Theano or
        TensorFlow through Keras."""

        cls.available_keras_activations = ['softplus',
                                           'relu',
                                           'tanh',
                                           'sigmoid',
                                           'hard_sigmoid',
                                           'linear']

    @classmethod
    def get_available_initializations(cls):
        """Get all the available initializations available from Theano or
        TensorFlow through Keras."""

        cls.available_keras_initializations = ['uniform',
                                               'lecun_uniform',
                                               'normal',
                                               'identity',
                                               'orthogonal',
                                               'zero',
                                               'glorot_normal',
                                               'glorot_uniform',
                                               'he_normal',
                                               'he_uniform']

    @classmethod
    def get_available_border_modes(cls):
        """Get all the available border-modes available from Theano or
        TensorFlow through Keras."""

        cls.available_keras_border_modes = ['valid', 'same']


def main():
    """Main program."""

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.debug("Preparing some combinations of NN hyper-parameters")

    NNHyperParameters.get_available_optimizers()
    NNHyperParameters.get_available_objectives()
    NNHyperParameters.get_available_activations()
    NNHyperParameters.get_available_initializations()
    NNHyperParameters.get_available_border_modes()

    all_combinations_hyp_param = list(itertools.product(
        NNHyperParameters.available_keras_optimizers,
        NNHyperParameters.available_keras_objectives,
        NNHyperParameters.available_keras_activations,
        NNHyperParameters.available_keras_initializations,
        NNHyperParameters.available_keras_border_modes))

    logging.debug("Testing hyper-parameters")

    for combination in all_combinations_hyp_param:
        optimizer = combination[0]
        objective = combination[1]
        activation = combination[2]
        initialization = combination[3]
        border_mode = combination[4]

        # we need to protect the call to the NN with these hyperparameters in
        # this iteration in the loop because we don't know if this combination
        # of hyper-parameters is supported
        try:
            # run a NN with these hyper-parameters in this iteration
            pass    # TODO
        except Exception as dummy_exc:     # pylint: disable=broad-except
            logging.error("Exception caught for hyper-parameters: %s",
                          ' '.join(repr(combination)))


class FlashCards(object):
    """Container class for some flash-cards parameters."""
    # pylint: disable=too-few-public-methods

    flash_card_dims = (500, 500)

    # to make the flash-cards distinct to reinforce learning
    variation_in_width = 50


def generate_flash_cards(colors_seq, dest_flash_card_preffix):
    """Generates a set of flash cards .PNG files, all having this same color
       sequence 'colors_seq' from left to right, but differing among them, not
       in the colors, but only in the width of each color bar in the flash
       card. This is necessary to increase the dimensionality of the tensor
       for reinforcement learning. All flash cards are saved with a common
       preffix 'dest_flash_card_preffix'.
    """

    from random import shuffle

    uniform_width_of_each_bar = int(FlashCards.flash_card_dims[0] /
                                    len(colors_seq))

    # the shifts in the column widths of each color bar inside the flash card
    shifts_to_reinforce_learning = list(
        range(int(-FlashCards.variation_in_width/2),
              int(FlashCards.variation_in_width/2) + 1)
    )

    shuffle(shifts_to_reinforce_learning)

    for variation in xrange(len(shifts_to_reinforce_learning)):
        dest_flash_card_fname = "{}_{}.png".format(dest_flash_card_preffix,
                                                   variation)
        column_width = (uniform_width_of_each_bar +
                        shifts_to_reinforce_learning[variation])

        generate_a_flash_card(colors_seq, column_width, dest_flash_card_fname)


def generate_a_flash_card(colors_seq, color_bar_width, dest_flash_card_fname):
    """Generate one flash card with the given colors seq, width of each color
    bar, to the given 'dest_flash_card_fname' PNG file."""

    from PIL import Image, ImageDraw

    a_flash_card = Image.new('RGB', FlashCards.flash_card_dims)
    draw = ImageDraw.Draw(a_flash_card)

    current_column = 0

    for color in colors_seq:
        # paint this bar on the flash card with the current color in
        # colors_seq
        next_col = current_column + color_bar_width
        draw.rectangle([(current_column, 0),
                        (next_col, FlashCards.flash_card_dims[1])],
                       outline=color, fill=color)
        current_column = next_col

    # paint the remainder bar in the flash card with the last color, if
    # any remainder bar was left because of the shift on the uniform width
    if current_column < FlashCards.flash_card_dims[0] - 1:
        draw.rectangle([(current_column, 0), FlashCards.flash_card_dims],
                       outline=colors_seq[-1], fill=colors_seq[-1])

    a_flash_card.save(dest_flash_card_fname, 'PNG')


if __name__ == '__main__':
    main()
