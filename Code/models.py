import tensorflow as tf
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Input, Activation, Concatenate, Flatten, Dropout, \
    MaxPooling1D, MaxPooling2D
from tensorflow.python.keras import regularizers, Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Lambda, Conv2D, Convolution2D, BatchNormalization
from tensorflow.python.keras.layers import SeparableConv2D, DepthwiseConv2D, ZeroPadding2D
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.python.client import device_lib
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow_model_optimization.python.core.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay, ConstantSparsity
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude, strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep, \
    PruningSummaries





def squeeze_net(shape, num_classes=6):
    input_img = Input(shape=shape)

    conv1 = Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_last")(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_last")(conv1)
    fire2_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format="channels_last")(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format="channels_last")(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format="channels_last")(fire2_squeeze)
    merge2 = Concatenate(axis=-1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format="channels_last")(merge2)
    fire3_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format="channels_last")(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format="channels_last")(fire3_squeeze)
    merge3 = Concatenate(axis=-1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format="channels_last")(merge3)
    fire4_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format="channels_last")(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format="channels_last")(fire4_squeeze)
    merge4 = Concatenate(axis=-1)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_last")(merge4)

    fire5_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze',
        data_format="channels_last")(maxpool4)
    fire5_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1',
        data_format="channels_last")(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2',
        data_format="channels_last")(fire5_squeeze)
    merge5 = Concatenate(axis=-1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format="channels_last")(merge5)
    fire6_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format="channels_last")(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format="channels_last")(fire6_squeeze)
    merge6 = Concatenate(axis=-1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze',
        data_format="channels_last")(merge6)
    fire7_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1',
        data_format="channels_last")(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2',
        data_format="channels_last")(fire7_squeeze)
    merge7 = Concatenate(axis=-1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze',
        data_format="channels_last")(merge7)
    fire8_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1',
        data_format="channels_last")(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2',
        data_format="channels_last")(fire8_squeeze)
    merge8 = Concatenate(axis=-1)([fire8_expand1, fire8_expand2])

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        data_format="channels_last")(merge8)
    fire9_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_squeeze',
        data_format="channels_last")(maxpool8)
    fire9_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand1',
        data_format="channels_last")(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand2',
        data_format="channels_last")(fire9_squeeze)
    merge9 = Concatenate(axis=-1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        num_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_last")(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D(data_format='channels_last')(conv10)
    softmax = Activation("softmax", name='softmax')(global_avgpool10)

    final_model = Model(inputs=input_img, outputs=softmax)

    return final_model



def sparcenet(shape, num_classes = 6):
    input_img = Input(shape=shape)
    maxpool1 = MaxPooling2D(
        pool_size=(1, 1), strides=1, name='maxpool1', data_format="channels_last")(input_img)
    conv1 = Convolution2D(
        96, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform',
        strides=1, padding='same', name='conv1', data_format="channels_last")(maxpool1)
    maxpool2 = MaxPooling2D(
        pool_size=(2, 2), strides=1, name='maxpool2', data_format="channels_last")(conv1)
    dense1 = Dense(32, kernel_initializer='glorot_uniform', name='dense1')(maxpool2)
    merge1 = Concatenate(axis=-1)([maxpool2, dense1])
    desne2 = Dense(num_classes, activation='softmax', name='dense2')(merge1)
    globalavg = GlobalAveragePooling2D(name='globalavg', data_format='channels_last')(desne2)
    final_model = Model(inputs=input_img, outputs=globalavg)
    return final_model


def squeeze_net_small(shape, num_classes=6):
    input_img = Input(shape=shape)

    conv1 = Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_last")(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_last")(conv1)
    fire2_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format="channels_last")(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format="channels_last")(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format="channels_last")(fire2_squeeze)
    merge2 = Concatenate(axis=-1)([fire2_expand1, fire2_expand2])

    # fire3_squeeze = Convolution2D(
    #     16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire3_squeeze',
    #     data_format="channels_last")(merge2)
    # fire3_expand1 = Convolution2D(
    #     64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire3_expand1',
    #     data_format="channels_last")(fire3_squeeze)
    # fire3_expand2 = Convolution2D(
    #     64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire3_expand2',
    #     data_format="channels_last")(fire3_squeeze)
    # merge3 = Concatenate(axis=-1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format="channels_last")(merge2)
    fire4_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format="channels_last")(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format="channels_last")(fire4_squeeze)
    merge4 = Concatenate(axis=-1)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_last")(merge4)

    # fire5_squeeze = Convolution2D(
    #     32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire5_squeeze',
    #     data_format="channels_last")(maxpool4)
    # fire5_expand1 = Convolution2D(
    #     128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire5_expand1',
    #     data_format="channels_last")(fire5_squeeze)
    # fire5_expand2 = Convolution2D(
    #     128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire5_expand2',
    #     data_format="channels_last")(fire5_squeeze)
    # merge5 = Concatenate(axis=-1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format="channels_last")(maxpool4)
    fire6_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format="channels_last")(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format="channels_last")(fire6_squeeze)
    merge6 = Concatenate(axis=-1)([fire6_expand1, fire6_expand2])

    # fire7_squeeze = Convolution2D(
    #     48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire7_squeeze',
    #     data_format="channels_last")(merge6)
    # fire7_expand1 = Convolution2D(
    #     192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire7_expand1',
    #     data_format="channels_last")(fire7_squeeze)
    # fire7_expand2 = Convolution2D(
    #     192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire7_expand2',
    #     data_format="channels_last")(fire7_squeeze)
    # merge7 = Concatenate(axis=-1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze',
        data_format="channels_last")(merge6)
    fire8_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1',
        data_format="channels_last")(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2',
        data_format="channels_last")(fire8_squeeze)
    merge8 = Concatenate(axis=-1)([fire8_expand1, fire8_expand2])

    # maxpool8 = MaxPooling2D(
    #     pool_size=(3, 3), strides=(2, 2), name='maxpool8',
    #     data_format="channels_last")(merge8)
    # fire9_squeeze = Convolution2D(
    #     64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire9_squeeze',
    #     data_format="channels_last")(maxpool8)
    # fire9_expand1 = Convolution2D(
    #     256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire9_expand1',
    #     data_format="channels_last")(fire9_squeeze)
    # fire9_expand2 = Convolution2D(
    #     256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire9_expand2',
    #     data_format="channels_last")(fire9_squeeze)
    # merge9 = Concatenate(axis=-1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge8)
    conv10 = Convolution2D(
        num_classes, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_last")(fire9_dropout)

    global_avgpool10 = GlobalAveragePooling2D(data_format='channels_last')(conv10)
    softmax = Activation("softmax", name='softmax')(global_avgpool10)

    final_model = Model(inputs=input_img, outputs=softmax)

    return final_model



def m3(shape, num_classes=6):
    print('Using Model M3')
    m = Sequential()
    m.add(Conv2D(256,
                 input_shape=shape,
                 kernel_size=80,
                 strides=2,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(MaxPooling2D(pool_size=2, strides=None))
    m.add(Conv2D(256,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=regularizers.l2(l=0.0001)))
    m.add(BatchNormalization())
    m.add(Activation('relu'))

    m.add(MaxPooling2D(pool_size=2, strides=None))
    m.add(GlobalAveragePooling2D())  # Same as GAP for 1D Conv Layer
    m.add(Dense(num_classes, activation='softmax'))
    return m



def Basline_Keras(shape,frames=128, bands=128, channels=1, num_classes=6,
                  conv_size=(5,5), conv_block='conv',
                  downsample_size=(4,2),
                  fully_connected=64,
                  n_stages=None, n_blocks_per_stage=None,
                  filters=24, kernels_growth=2,
                  dropout=0.5,
                  use_strides=False):
    from keras.regularizers import l2
    """
    Implements SB-CNN model from
    Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification
    Salamon and Bello, 2016.
    https://arxiv.org/pdf/1608.04363.pdf
    Based on https://gist.github.com/jaron/5b17c9f37f351780744aefc74f93d3ae
    but parameters are changed back to those of the original paper authors,
    and added Batch Normalization
    """
    Conv2 = SeparableConv2D if conv_block == 'depthwise_separable' else Convolution2D
    assert conv_block in ('conv', 'depthwise_separable')
    kernel = conv_size
    if use_strides:
        strides = downsample_size
        pool = (1, 1)
    else:
        strides = (1, 1)
        pool = downsample_size
    block1 = [
        Convolution2D(filters, kernel, padding='same', strides=strides,
                      input_shape=shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool),
        Activation('relu'),
    ]
    block2 = [
        Conv2(filters * kernels_growth, kernel, padding='same', strides=strides),
        BatchNormalization(),
        MaxPooling2D(pool_size=pool),
        Activation('relu'),
    ]

    block3 = [
        Conv2(filters * kernels_growth, kernel, padding='valid', strides=strides),
        BatchNormalization(),
        Activation('relu'),
    ]
    backend = [
        Flatten(),
        Dropout(dropout),
        Dense(fully_connected, kernel_regularizer=l2(0.001)),
        Activation('relu'),
        Dropout(dropout),
        Dense(num_classes, kernel_regularizer=l2(0.001)),
        Activation('softmax'),
    ]
    layers = block1 + block2 + block3 + backend
    model = Sequential(layers)
    model.summary()
    return model





def Stride_Keras(shape, frames=128, bands=128, channels=1, num_classes=6,
                    conv_size=(5, 5),
                    conv_block='conv',
                    downsample_size=(2, 2),
                    n_stages=3, n_blocks_per_stage=1,
                    filters=24, kernels_growth=1.5,
                    fully_connected=64,
                    dropout=0.5, l2=0.001):

        def add_common(x, name):
            x = BatchNormalization(name=name + '_bn')(x)
            x = Activation('relu', name=name + '_relu')(x)
            return x

        def conv(x, kernel, filters, downsample, name,
                 padding='same'):
            """Regular convolutional block"""
            x = Conv2D(filters, kernel, strides=downsample,
                       name=name, padding=padding)(x)
            return add_common(x, name)

        def conv_ds(x, kernel, filters, downsample, name,
                    padding='same'):
            """Depthwise Separable convolutional block
            (Depthwise->Pointwise)
            MobileNet style"""
            x = SeparableConv2D(filters, kernel, padding=padding, strides=downsample, name=name + '_ds')(x)
            return add_common(x, name=name + '_ds')

        def conv_bottleneck_ds(x, kernel, filters, downsample, name,
                               padding='same', bottleneck=0.5):
            """
            Bottleneck -> Depthwise Separable
            (Pointwise->Depthwise->Pointswise)
            MobileNetV2 style
            """
            if padding == 'valid':
                pad = ((0, kernel[0] // 2), (0, kernel[0] // 2))
                x = ZeroPadding2D(padding=pad, name=name + 'pad')(x)

            x = Conv2D(int(filters * bottleneck), (1, 1),
                       padding='same', strides=downsample,
                       name=name + '_pw')(x)
            add_common(x, name + '_pw')

            x = SeparableConv2D(filters, kernel,
                                padding=padding, strides=(1, 1),
                                name=name + '_ds')(x)
            return add_common(x, name + '_ds')

        def conv_effnet(x, kernel, filters, downsample, name,
                        bottleneck=0.5, strides=(1, 1), padding='same', bias=False):
            """Pointwise -> Spatially Separable conv&pooling
            Effnet style"""
            assert downsample[0] == downsample[1]
            downsample = downsample[0]
            assert kernel[0] == kernel[1]
            kernel = kernel[0]

            ch_in = int(filters * bottleneck)
            ch_out = filters

            if padding == 'valid':
                pad = ((0, kernel // 2), (0, kernel // 2))
                x = ZeroPadding2D(padding=pad, name=name + 'pad')(x)

            x = Conv2D(ch_in, (1, 1), strides=downsample,
                       padding=padding, use_bias=bias, name=name + 'pw')(x)
            x = add_common(x, name=name + 'pw')

            x = DepthwiseConv2D((1, kernel),
                                padding=padding, use_bias=bias, name=name + 'dwv')(x)
            x = add_common(x, name=name + 'dwv')

            x = DepthwiseConv2D((kernel, 1), padding='same',
                                use_bias=bias, name=name + 'dwh')(x)
            x = add_common(x, name=name + 'dwh')

            x = Conv2D(ch_out, (1, 1), padding=padding, use_bias=bias, name=name + 'rh')(x)
            return add_common(x, name=name + 'rh')

        block_types = {
            'conv': conv,
            'depthwise_separable': conv_ds,
            'bottleneck_ds': conv_bottleneck_ds,
            'effnet': conv_effnet,
        }

        def backend_dense1(x, n_classes, fc=64, regularization=0.001, dropout=0.5):
            from keras.regularizers import l2
            """
            SB-CNN style classification backend
            """
            x = Flatten()(x)
            x = Dropout(dropout)(x)
            x = Dense(fc, kernel_regularizer=l2(regularization))(x)
            x = Activation('relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(n_classes, kernel_regularizer=l2(regularization))(x)
            x = Activation('softmax')(x)
            return x

        #input = Input(shape=(bands, frames, channels))
        input = Input(shape)
        x = input

        block_no = 0
        for stage_no in range(0, n_stages):
            for b_no in range(0, n_blocks_per_stage):
                # last padding == valid
                padding = 'valid' if block_no == (n_stages * n_blocks_per_stage) - 1 else 'same'
                # downsample only one per stage
                downsample = downsample_size if b_no == 0 else (1, 1)
                # first convolution is standard
                conv_func = conv if block_no == 0 else block_types.get(conv_block)
                name = "conv{}".format(block_no)
                x = conv_func(x, conv_size, int(filters), downsample, name=name, padding=padding)

                block_no += 1
            filters = filters * kernels_growth
        x = backend_dense1(x, num_classes, fully_connected, regularization=l2)
        model = Model(input, x)
        model.summary()
        return model

# def m5(shape, num_classes=6):
#     print('Using Model M5')
#     m = Sequential()
#     m.add(Conv2D(128,
#                  input_shape=shape,
#                  kernel_size=80,
#                  strides=4,
#                  padding='same',
#                  kernel_initializer='glorot_uniform',
#                  kernel_regularizer=regularizers.l2(l=0.0001)))
#     m.add(BatchNormalization())
#     m.add(Activation('relu'))
#     m.add(MaxPooling2D(pool_size=4, strides=None))
#     m.add(Conv2D(128,
#                  kernel_size=3,
#                  strides=1,
#                  padding='same',
#                  kernel_initializer='glorot_uniform',
#                  kernel_regularizer=regularizers.l2(l=0.0001)))
#     m.add(BatchNormalization())
#     m.add(Activation('relu'))
#     m.add(MaxPooling2D(pool_size=4, strides=None))
#     m.add(Conv2D(256,
#                  kernel_size=3,
#                  strides=1,
#                  padding='same',
#                  kernel_initializer='glorot_uniform',
#                  kernel_regularizer=regularizers.l2(l=0.0001)))
#     m.add(BatchNormalization())
#     m.add(Activation('relu'))
#     m.add(MaxPooling2D(pool_size=4, strides=None))
#     m.add(Conv2D(512,
#                  kernel_size=3,
#                  strides=1,
#                  padding='same',
#                  kernel_initializer='glorot_uniform',
#                  kernel_regularizer=regularizers.l2(l=0.0001)))
#     m.add(BatchNormalization())
#     m.add(Activation('relu'))
#     m.add(MaxPooling2D(pool_size=4, strides=None))
#     m.add(GlobalAveragePooling2D())
#     m.add(Dense(num_classes, activation='softmax'))
#     return m


# def m11(num_classes=10):
#     print('Using Model M11')
#     m = Sequential()
#     m.add(Conv1D(64,
#                  input_shape=[AUDIO_LENGTH, 1],
#                  kernel_size=80,
#                  strides=4,
#                  padding='same',
#                  kernel_initializer='glorot_uniform',
#                  kernel_regularizer=regularizers.l2(l=0.0001)))
#     m.add(BatchNormalization())
#     m.add(Activation('relu'))
#     m.add(MaxPooling1D(pool_size=4, strides=None))

#     for i in range(2):
#         m.add(Conv1D(64,
#                      kernel_size=3,
#                      strides=1,
#                      padding='same',
#                      kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(l=0.0001)))
#         m.add(BatchNormalization())
#         m.add(Activation('relu'))
#     m.add(MaxPooling1D(pool_size=4, strides=None))

#     for i in range(2):
#         m.add(Conv1D(128,
#                      kernel_size=3,
#                      strides=1,
#                      padding='same',
#                      kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(l=0.0001)))
#         m.add(BatchNormalization())
#         m.add(Activation('relu'))
#     m.add(MaxPooling1D(pool_size=4, strides=None))

#     for i in range(3):
#         m.add(Conv1D(256,
#                      kernel_size=3,
#                      strides=1,
#                      padding='same',
#                      kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(l=0.0001)))
#         m.add(BatchNormalization())
#         m.add(Activation('relu'))
#     m.add(MaxPooling1D(pool_size=4, strides=None))

#     for i in range(2):
#         m.add(Conv1D(512,
#                      kernel_size=3,
#                      strides=1,
#                      padding='same',
#                      kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(l=0.0001)))
#         m.add(BatchNormalization())
#         m.add(Activation('relu'))

#     m.add(Lambda(lambda x: K.mean(x, axis=1))) # Same as GAP for 1D Conv Layer
#     m.add(Dense(num_classes, activation='softmax'))
#     return m


# def m_rec(num_classes=10):
#     from keras.layers.recurrent import LSTM
#     print('Using Model LSTM 1')
#     m = Sequential()
#     m.add(Conv1D(64,
#                  input_shape=[AUDIO_LENGTH, 1],
#                  kernel_size=80,
#                  strides=4,
#                  padding='same',
#                  kernel_initializer='glorot_uniform',
#                  kernel_regularizer=regularizers.l2(l=0.0001)))
#     m.add(BatchNormalization())
#     m.add(Activation('relu'))
#     m.add(MaxPooling1D(pool_size=4, strides=None))
#     m.add(LSTM(32,
#                kernel_regularizer=regularizers.l2(l=0.0001),
#                return_sequences=True,
#                dropout=0.2))
#     m.add(LSTM(32,
#                kernel_regularizer=regularizers.l2(l=0.0001),
#                return_sequences=False,
#                dropout=0.2))
#     m.add(Dense(32))
#     m.add(Dense(num_classes, activation='softmax'))
#     return m


# def m18(num_classes=10):
#     print('Using Model M18')
#     m = Sequential()
#     m.add(Conv1D(64,
#                  input_shape=[AUDIO_LENGTH, 1],
#                  kernel_size=80,
#                  strides=4,
#                  padding='same',
#                  kernel_initializer='glorot_uniform',
#                  kernel_regularizer=regularizers.l2(l=0.0001)))
#     m.add(BatchNormalization())
#     m.add(Activation('relu'))
#     m.add(MaxPooling1D(pool_size=4, strides=None))

#     for i in range(4):
#         m.add(Conv1D(64,
#                      kernel_size=3,
#                      strides=1,
#                      padding='same',
#                      kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(l=0.0001)))
#         m.add(BatchNormalization())
#         m.add(Activation('relu'))
#     m.add(MaxPooling1D(pool_size=4, strides=None))

#     for i in range(4):
#         m.add(Conv1D(128,
#                      kernel_size=3,
#                      strides=1,
#                      padding='same',
#                      kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(l=0.0001)))
#         m.add(BatchNormalization())
#         m.add(Activation('relu'))
#     m.add(MaxPooling1D(pool_size=4, strides=None))

#     for i in range(4):
#         m.add(Conv1D(256,
#                      kernel_size=3,
#                      strides=1,
#                      padding='same',
#                      kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(l=0.0001)))
#         m.add(BatchNormalization())
#         m.add(Activation('relu'))
#     m.add(MaxPooling1D(pool_size=4, strides=None))

#     for i in range(4):
#         m.add(Conv1D(512,
#                      kernel_size=3,
#                      strides=1,
#                      padding='same',
#                      kernel_initializer='glorot_uniform',
#                      kernel_regularizer=regularizers.l2(l=0.0001)))
#         m.add(BatchNormalization())
#         m.add(Activation('relu'))

#     m.add(Lambda(lambda x: K.mean(x, axis=1))) # Same as GAP for 1D Conv Layer
#     m.add(Dense(num_classes, activation='softmax'))
#     return m
