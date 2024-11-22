import numpy as np
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D
from keras.layers import Lambda, Subtract
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import tensorflow as tf


def W_init(shape, dtype=None, name=None):
    """Initialize weights as in paper"""
    values = np.random.normal(loc=0, scale=1e-2, size=shape).astype(np.float32)
    return tf.Variable(values, name=name)


def b_init(shape, dtype=None, name=None):
    """Initialize bias as in paper"""
    values = np.random.normal(loc=0.5, scale=1e-2, size=shape).astype(np.float32)
    return tf.Variable(values, name=name)


input_shape = (100, 100, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)

# build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(
    Conv2D(
        64,
        (10, 10),
        activation="relu",
        input_shape=input_shape,
        kernel_initializer=W_init,
        kernel_regularizer=l2(2e-4),
    )
)
convnet.add(MaxPooling2D())
convnet.add(
    Conv2D(
        128,
        (7, 7),
        activation="relu",
        kernel_regularizer=l2(2e-4),
        kernel_initializer=W_init,
        bias_initializer=b_init,
    )
)
convnet.add(MaxPooling2D())
convnet.add(
    Conv2D(
        128,
        (4, 4),
        activation="relu",
        kernel_initializer=W_init,
        kernel_regularizer=l2(2e-4),
        bias_initializer=b_init,
    )
)
convnet.add(MaxPooling2D())
convnet.add(
    Conv2D(
        256,
        (4, 4),
        activation="relu",
        kernel_initializer=W_init,
        kernel_regularizer=l2(2e-4),
        bias_initializer=b_init,
    )
)
convnet.add(Flatten())
convnet.add(
    Dense(
        4096,
        activation="sigmoid",
        kernel_regularizer=l2(1e-3),
        kernel_initializer=W_init,
        bias_initializer=b_init,
    )
)

# encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

# merge two encoded inputs with the l1 distance between them
subtracted = Subtract()([encoded_l, encoded_r])
both = Lambda(lambda x: abs(x))(subtracted)
prediction = Dense(1, activation="sigmoid", bias_initializer=b_init)(both)
SiameseNet = Model(inputs=[left_input, right_input], outputs=prediction)

optimizer = Adam(0.00006)
SiameseNet.compile(loss="binary_crossentropy", optimizer=optimizer)
