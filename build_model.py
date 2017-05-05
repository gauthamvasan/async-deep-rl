from keras.models import Model
from keras.layers import Conv2D, Input, Dense, Dropout, Flatten, MaxPooling2D
from keras import backend as K
import tensorflow as tf

def build_network(num_actions, history_length = 4, scaled_width = 84, scaled_height = 84):
    with tf.device("/cpu:0"):
        inputs = Input(shape=(history_length, scaled_width, scaled_height,))

        x = Conv2D(nb_filter=16, nb_col=8, nb_row=8,            # Num filters and its width and height resp
                         subsample=(4,4),                       # Stride of the conv filter
                         activation='relu',
                         border_mode='same',)(inputs)           # Padding

        x = Conv2D(nb_filter=32, nb_row=4, nb_col=4,
                         subsample=(2,2),
                         border_mode='same',
                         activation='relu')(x)

        x = Flatten()(x)
        x = (Dense(output_dim=256, activation='relu'))(x)
        q_values = Dense(output_dim=num_actions, activation='linear')(x)

        model = Model(input=inputs, output=q_values)

    return model

