from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Convolution2D, Dense, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K

DIM = 84
NUM_ACTIONS = 3
LEARNING_SEQ_LEN = 4


def get_model_deepmind(dim=DIM, num_actions=NUM_ACTIONS):
    main_input = Input(shape=(dim, dim, LEARNING_SEQ_LEN))
    # input_shape = (-1, LEARNING_SEQ_LEN, dim, dim)
    input_shape = (dim, dim, LEARNING_SEQ_LEN)

    x = Convolution2D(16, 8, 8, input_shape=input_shape, activation='relu',
                      subsample=(4, 4), border_mode='same', dim_ordering='tf')(main_input)

    # x = Convolution2D(16, 8, 8, border_mode='same', activation='relu',
    #                   subsample=(4, 4), dim_ordering='tf')(x)

    x = Convolution2D(32, 4, 4, border_mode='same', activation='relu',
                      subsample=(2, 2), dim_ordering='tf')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_actions)(x)

    model = Model(input=main_input, output=x)
    model.compile(optimizer='adam', loss='mse')
    return model
