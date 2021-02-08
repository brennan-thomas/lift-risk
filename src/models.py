from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Enable memory growth for any GPUs. This lets us use only the memory we need,
# and not use it all up in case other processes are using it
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# TODO FIX DEFAULTS


def deepconvlstm4class(shape, kernel=5, lr=0.0001, reg=0.001, dropout=0.5):
    """Construct a model using DeepConvLSTM architecture, with 4 output neurons
    
    Arguments:
    shape   -- shape of the input tensor ([750, 36] for the NIOSH dataset)
    kernel  -- size of the kernel in the Conv1D layers (default 5)
    lr      -- learning rate for Adam optimizer (default 0.0001)
    reg     -- regularization parameter (default 0.001)
    dropout -- dropout percentage (default 0.5)
    """

    inpt = Input(shape=shape)
    x = Conv1D(64, activation='tanh', kernel_size=kernel, padding='same')(inpt)
    x = Conv1D(64, activation='tanh', kernel_size=kernel, padding='same')(x)
    x = Conv1D(64, activation='tanh', kernel_size=kernel, padding='same')(x)
    x = Conv1D(64, activation='tanh', kernel_size=kernel, padding='same')(x)
    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    x = LSTM(128, activation='tanh', return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)

    opt = Adam(learning_rate=lr)
    model = Model(inputs=inpt, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def residual_block(in_shape, out_dim, kernel=5, reg=0.01):
    """Construct a residual block using convolutions and skip connections
    
    Arguments:
    in_shape    -- shape of the input tensor
    out_dim     -- size of the channel dimension in the output (i.e. number of Conv1D filters)
    kernel      -- size of the kernel in the Conv1D layers (default 5)
    reg         -- regularization parameter (default 0.01)
    
    """

    in_dim = in_shape[-1]
    inpt = Input(in_shape)
    skip = inpt
    
    # If the number of output channels is different than input channels, we need to transform the 
    # input dim to the same size so that they can be added together in the skip connection. 
    if in_dim != out_dim:
        skip = Conv1D(out_dim, activation=None, kernel_size=1, strides=1, use_bias=False, 
                      kernel_regularizer=tf.keras.regularizers.l2(l=reg))(skip)
    x = BatchNormalization()(inpt)
    x = relu(x)
    x = Conv1D(out_dim, activation=None, kernel_size=kernel, padding='same', use_bias=False, 
               kernel_regularizer=tf.keras.regularizers.l2(l=reg))(x)
    x = BatchNormalization()(x)
    x = relu(x)
    x = Conv1D(out_dim, activation=None, kernel_size=kernel, padding='same', 
               kernel_regularizer=tf.keras.regularizers.l2(l=reg))(x)
    x = x + skip

    model = Model(inputs=inpt, outputs=x)
    return model

def residual_4class_dense(shape, kernel=5, lr=0.001, reg=0.01, dropout=0.5):
    """Construct a model using a modified DeepConvLSTM architecture with residual blocks, with 4 output neurons
    
    Arguments:
    shape   -- shape of the input tensor ([750, 36] for the NIOSH dataset)
    kernel  -- size of the kernel in the Conv1D layers (default 5)
    lr      -- learning rate for Adam optimizer (default 0.0001)
    reg     -- regularization parameter (default 0.001)
    dropout -- dropout percentage (default 0.5)
    """

    inpt = Input(shape)
    x = residual_block(inpt.shape[1:], 128, kernel, reg=reg)(inpt)
    x = residual_block(x.shape[1:], 128, kernel, reg=reg)(x)
    x = residual_block(x.shape[1:], 128, kernel, reg=reg)(x)
    x = residual_block(x.shape[1:], 128, kernel, reg=reg)(x)
    x = residual_block(x.shape[1:], 128, kernel, reg=reg)(x)
    x = residual_block(x.shape[1:], 128, kernel, reg=reg)(x)
    x = Dropout(dropout)(x)

    x = LSTM(128, activation='tanh', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l=reg))(x)
    x = LSTM(128, activation='tanh', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l=reg))(x)
    x = Flatten()(x)
    x = Dense(512, activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l=reg))(x)
    x = BatchNormalization()(x)
    x = relu(x)
    x = Dropout(dropout)(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    
    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

