# THIS CODE WILL NOT RUN ON TENSORFLOW 2.0

import seaborn as sns
import pdb
import tensorflow.python.keras
from datetime import date
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math
import glob
import uuid
import shutil
from pprint import pprint
import time
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
from scipy.fftpack import dct
import scipy.io.wavfile
import numpy as np
import multiprocessing as mp
import librosa
import librosa.display
from Pruning import pruning
from convert import to_tf_quant
from models import m3, squeeze_net_small, sparcenet, Stride_Keras, Basline_Keras, squeeze_net

from convert import to_tf_quant
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
import h5py

import os

import wandb
from wandb.keras import WandbCallback
wandb.init(project="shrink")


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ------------------------------------------------------------------------------------------------------------------

# Check Gpu Available
print(device_lib.list_local_devices())


# ------------------------------------------------------------------------------------------------------------------

Path = str(os.getcwd())


#-------------------------------------------------------------------------------------------------------------------

# Setting Directory and Path for Computation

if not os.path.exists(Path +'/model/'):
    os.mkdir(Path + '/model/')
    os.mkdir(Path + '/model/tensorflowlitemodel/')
    os.mkdir(Path + '/model/prunedmodel')

if not os.path.exists(Path + '/amp-processed-data/'):
    os.mkdir(Path + '/amp-processed-data/')

RAW_DATA_DIR = (Path + '/Data/')
AMP_PROCESSED_DATA_DIR = (Path + '/amp-processed-data/')
MODEL_NAME = 'Audio_Recorder'

TextFile = open(Path + '/model/ShrinkLog.txt', "w+")
today = date.today()
initaltext = ("Model Optimization for Edge Devices Log File" + "\n" + str(today) + "\n" + ('*' * 100) + "\n")
TextFile.write(initaltext)


SAMPLE_RATE = 30000
AUDIO_LENGTH_MS = 1.0 # For 500ms write AUDIO_LENGTH_MS = 0.5 for 30ms = 0.030
AUDIO_LENGTH = int(SAMPLE_RATE * (AUDIO_LENGTH_MS))
SQRT_AUDIO_LENGTH = int(math.sqrt(AUDIO_LENGTH))
channel = 1
epochs = 100
epochs_prune = 25
batch_size = 32
verbose = 1
num_classes = 6
plot_number = 0

TextFile.write("Audio Length:" + str(AUDIO_LENGTH) + "\n" + "Sample Rate:" + str(SAMPLE_RATE) + "\n" + "Epochs:" + str(
    epochs) + "\n" + "Pruning Epochs:" + str(epochs_prune) + "\n" + "Batch Size:" + str(batch_size) + "\n"  + "Audio_Length_MS:" + str(AUDIO_LENGTH) + "\n"  + "SQRT_AUDIO_LENGTH:" + str(SQRT_AUDIO_LENGTH) + "\n" + (
    '*' * 100) + "\n")



# --------------------------------------------------------------------------------------------------------------------
# Function to get Labels

#path = location of data file
#labels = stores the names of the labels present in the directory
#label_indices = Stores the count of labels {In our case 0,1,2,3,4,5}
#to_categorical(label_indices) = generates a 6X6 matrix

def get_labels(path):

    labels = [i for i in sorted(os.listdir(path)) if i[0] != "."]
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)



# --------------------------------------------------------------------------------------------------------------------
# Function to generate .npy files

# label, input_path, output_path, tqdm_position = args values are given to label_to_amplitude_vecs when called from process_data_amplitude function
# wavfiles = [os.path.join(input_path, label, wavfile) concatinates the input path with label name and wave file
# wavfiles = [os.path.join(input_path, label, wavfile) for wavfile in os.listdir(os.path.join(input_path, label))] provies the actual wave file to wavefile varriable as per label in the directory

def label_to_amplitude_vecs(args) -> None:
    label, input_path, output_path, tqdm_position = args


    # Get all audio files for this label
    wavfiles = [os.path.join(input_path, label, wavfile) for wavfile in os.listdir(os.path.join(input_path, label))]

    # tqdm is amazing, so print all the things this way
    #print(" ", end="", flush=True)
    #twavs = tqdm(wavfiles)

    #vector=[] empty vector is intiated
    #audio_buf, _ = librosa.load(wavfile, mono=True, sr=SAMPLE_RATE)  puts wave fill in array
    #audio_buf = audio_buf.reshape(-1, 1)  makes one single horizontal array of wave file data
    #audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf) used of normalization     np.std calculates standard deviation
    #remaining_buf = audio_buf.copy()  makes a copy and stores in remaining_buf

    vectors = []
    for i, wavfile in enumerate(wavfiles):
        # Load the audio file; this also works for .flac files
        audio_buf, _ = librosa.load(wavfile, mono=True, sr=SAMPLE_RATE)
        audio_buf = audio_buf.reshape(-1, 1)
        audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
        remaining_buf = audio_buf.copy()

        #while loop is to re-arrange the data
        while remaining_buf.shape[0] > AUDIO_LENGTH:
            # Add the first AUDIO_LENGTH of the buffer as a new vector to train on
            new_buf = remaining_buf[: AUDIO_LENGTH]
            vectors.append(new_buf)

            # Remove 1/2 * AUDIO_LENGTH from the front of the buffer
            remaining_buf = remaining_buf[int(AUDIO_LENGTH / 2):]

        # Whatever is left, pad and stick in the training data
        remaining_buf = np.concatenate((remaining_buf, np.zeros(shape=(AUDIO_LENGTH - len(remaining_buf), 1))))
        vectors.append(remaining_buf)

        # Update tqdm
        # twavs.set_description("Label - '{}'".format(label))
        # twavs.refresh()
    np_vectors = np.array(vectors)
    np.save(os.path.join(output_path, label + '.npy'), np_vectors)
    print(" Processed Label - '{}'".format(label))
    return np_vectors.shape

# labels, _, _  only takes label parameter and ignores (label_indices, to_categorical(label_indices)) from get_labels function
#pool = mp.Pool() enables multi processing
#result = pool.map(label_to_amplitude_vecs,[(label, input_path, output_path, tqdm_position) Call the funtion label_to_amplitude_vecs and provides it with lable, input and output path

def process_data_amplitude(input_path, output_path):
    labels, _, _ = get_labels(input_path)
    pool = mp.Pool()
    result = pool.map(label_to_amplitude_vecs,
                      [(label, input_path, output_path, tqdm_position)
                      for tqdm_position, label in enumerate(labels)])
    pool.close()
    return result

# ----------------------------------------------------------------------------------------------------------------------
# Function to generate the .npy files
process_data_amplitude(RAW_DATA_DIR, AMP_PROCESSED_DATA_DIR)

# ---------------------------------------------------------------------------------------------------------------------
# Function call to generate labels

def get_train_test_val(data_dir, processed_data_dir, split_ratio: float):
    assert split_ratio < 1 and split_ratio > 0

    # Get available labels
    labels, indices, _ = get_labels(data_dir)

    # Getting first arrays
    X = np.load(processed_data_dir + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in tqdm(enumerate(labels[1:])):
        x = np.load(processed_data_dir + label + '.npy')
        print(processed_data_dir + label + '.npy')
        print(X.shape)
        print(x.shape)
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    print(X.shape)
    assert X.shape[0] == len(y)

    # Loading train set and test set
    training_data, testing_data, training_labels, testing_labels = train_test_split(X, y, test_size=(1 - split_ratio),
                                                                                    random_state=42, shuffle=True)
    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data,
                                                                                          training_labels,
                                                                                          test_size=(1 - split_ratio),
                                                                                          random_state=42, shuffle=True)
    training_labels = to_categorical(training_labels)
    testing_labels = to_categorical(testing_labels)
    validation_labels = to_categorical(validation_labels)
    return training_data, testing_data, validation_data, training_labels, testing_labels, validation_labels

# --------------------------------------------------------------------------------------------------------------------
# Function call to generate labels

(training_data, testing_data, validation_data, training_labels, testing_labels, validation_labels) = get_train_test_val(
    RAW_DATA_DIR, AMP_PROCESSED_DATA_DIR, 0.90)

print("Printing data shapes...")
print(training_data.shape)
print(training_labels.shape)
print(testing_data.shape)
print(testing_labels.shape)
print(validation_data.shape)
print(validation_labels.shape)

training_data = training_data[:, : int(math.sqrt(training_data.shape[1])) ** 2, :]
training_data = np.reshape(training_data, (training_data.shape[0], int(math.sqrt(training_data.shape[1])), int(math.sqrt(training_data.shape[1])), training_data.shape[2]))

testing_data = testing_data[:, : int(math.sqrt(testing_data.shape[1])) ** 2, :]
testing_data = np.reshape(testing_data, (testing_data.shape[0], int(math.sqrt(testing_data.shape[1])), int(math.sqrt(testing_data.shape[1])), testing_data.shape[2]))

validation_data = validation_data[:, : int(math.sqrt(validation_data.shape[1])) ** 2, :]
validation_data = np.reshape(validation_data, (validation_data.shape[0], int(math.sqrt(validation_data.shape[1])), int(math.sqrt(validation_data.shape[1])), validation_data.shape[2]))

print(training_data.shape)
print(training_labels.shape)
print(testing_data.shape)
print(testing_labels.shape)
print(validation_data.shape)
print(validation_labels.shape)

# ----------------------------------------------------------------------------------------------------------------
# Function to Train a model


def train(model, X_train, y_train, X_test, y_test):

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())
    
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
    mcp_save = ModelCheckpoint(Path + '/model/weights-{val_acc:.2f}.hdf5',
                               save_best_only=True, monitor='val_acc', mode='max')

    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose,
              validation_data=(X_test, y_test),
              callbacks=[reduce_lr, mcp_save, WandbCallback()],
              shuffle=True)



# --------------------------------------------------------------------------------------------------------------------

# Function to start model training


model = squeeze_net_small(shape=[SQRT_AUDIO_LENGTH, SQRT_AUDIO_LENGTH, 1], num_classes=6)

# # Quantization Awareness Training

sess = tf.compat.v1.keras.backend.get_session()
tf.contrib.quantize.create_training_graph(sess.graph)
sess.run(tf.global_variables_initializer())

train(model, training_data, training_labels, validation_data, validation_labels)

# Print the min max in fakequant

print("After Weight Adjustments")

for node in sess.graph.as_graph_def().node:
    if 'weights_quant/AssignMaxLast' in node.name \
            or 'weights_quant/AssignMinLast' in node.name:
        tensor = sess.graph.get_tensor_by_name(node.name + ':0')
        print('{} = {}'.format(node.name, sess.run(tensor)))


model.save(Path + '/model/amplitude-model.hdf5')
model.save(os.path.join(wandb.run.dir, "model.h5"))

# ---------------------------------------------------------------------------------------------------------------------

# Pruning the model


prunedmodel= pruning(epochs_prune, (Path + '/model/amplitude-model.hdf5'), training_data, batch_size, training_labels, validation_data, validation_labels, testing_data, testing_labels, Path)

tf.keras.models.save_model(prunedmodel, (Path + '/model/prunedmodel/amplitude-model-prune.hdf5'))




# --------------------------------------------------------------------------------------------------------------------

#Converting to TF Lite file_pathdel to Tf lite")

to_tf_quant((Path + '/model/prunedmodel/amplitude-model-prune.hdf5'), (Path + '/model/tensorflowlitemodel/amplitude-model-pruned-tensorflowlite.tflite'))



to_tf_quant((Path + '/model/amplitude-model.hdf5'), (Path + '/model/tensorflowlitemodel/amplitude-model-nonpruned-tensorflowlite.tflite'))

#----------------------------------------------------------------------------------------------------------------------

#Converting a Model to C-Array


os.system("xxd -i " + Path + "/model/tensorflowlitemodel/amplitude-model-nonpruned-tensorflowlite.tflite > " + Path + "/model/tensorflowlitemodel/nonpruned_carray.cc")

os.system("xxd -i " + Path + "/model/tensorflowlitemodel/amplitude-model-pruned-tensorflowlite.tflite > " + Path + "/model/tensorflowlitemodel/pruned_carray.cc")

#-----------------------------------------------------------------------------------------------------------------------

# Testing and Logging 


from test import test_tflite_model, test_reg_model, file_size

test_tflite_model((Path + '/model/tensorflowlitemodel/amplitude-model-pruned-tensorflowlite.tflite'), (Path + '/model/amp_lite-Pruned.pdf'), testing_labels, testing_data, 'Tf-lite Pruned', 'Pruned', TextFile, plot_number)

plot_number=plot_number+1

test_tflite_model((Path + '/model/tensorflowlitemodel/amplitude-model-nonpruned-tensorflowlite.tflite'), (Path + '/model/amp_lite-NonPruned.pdf'), testing_labels, testing_data, 'Tf-lite Non-Pruned', 'NonPruned', TextFile, plot_number)

plot_number=plot_number+1

test_reg_model ((Path + '/model/amplitude-model.hdf5'), (Path + '/model/amp_reg.pdf'), testing_labels, testing_data, TextFile, plot_number)
#--------------------------------------------------------------------------------------------------------------------------

print("Completed Run")


#---------------------------------------------------------------------------------------------------------------------------