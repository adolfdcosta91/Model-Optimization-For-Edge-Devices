

import numpy as np

#from convert import to_tf_quant
from models import m3
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep, \
    PruningSummaries
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude, strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay, ConstantSparsity
from tensorflow_model_optimization.python.core.sparsity import keras as sparsity
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.tools import freeze_graph
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Conv1D, Lambda, Convolution1D, Conv2D, Convolution2D
from tensorflow.python.keras import regularizers, Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, Input, Activation, Concatenate, Flatten, Dropout, \
    MaxPooling1D, MaxPooling2D
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback


def pruning(epochs_prune, keras_file, training_data, batch_size, training_labels, validation_data, validation_labels, testing_data, testing_labels, Path):


    logdir = (Path + '/ThesisShrink/model/')

    loaded_model = tf.keras.models.load_model(keras_file)

    num_train_samples = training_data.shape[0]
    end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs_prune
    print('End Step:')
    print(end_step)

    new_pruning_params = {
        'pruning_schedule': PolynomialDecay(initial_sparsity=0.50,
                                            final_sparsity=0.80,
                                            begin_step=0,
                                            end_step=end_step,
                                            frequency=50)
    }

    new_pruned_model = prune_low_magnitude(loaded_model, **new_pruning_params)
    new_pruned_model.summary()

    new_pruned_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])

    callbacks = [
        UpdatePruningStep(),
        PruningSummaries(log_dir=logdir, profile_batch=batch_size),
        WandbCallback()
    ]

    new_pruned_model.fit(training_data, training_labels,
                         batch_size=batch_size,
                         epochs=epochs_prune,
                         verbose=1,
                         callbacks=callbacks,
                         validation_data=(validation_data, validation_labels))

    score = new_pruned_model.evaluate(testing_data, testing_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    final_model = strip_pruning(new_pruned_model)
    final_model.summary()

    return final_model
    # # tf.keras.models.save_model(final_model, "/home/ben/Desktop/ThesisShrink/model/prunedmodel/amplitude-model-prune.hdf5"
    # #                           include_optimizer=False)
    #
    # tf.keras.models.save_model(final_model, (Path + '/ThesisShrink/model/prunedmodel/amplitude-model-prune.hdf5'))
    #
    # tf.keras.models.save_model(final_model, wandb.run.dir + "/amplitude-model-prune.h5")
    #
    # # _, new_pruned_keras_file = tempfile.mkstemp('.hdf5')
    # # print('Saving pruned model to: ', new_pruned_keras_file)

