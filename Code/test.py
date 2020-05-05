import seaborn as sns
import pdb
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
from wandb.keras import WandbCallback
import wandb
from convert import to_tf_quant
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
import h5py
import os

# ----------------------------------------------------------------------------------------------------------------------

# To Get File Size


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        num = file_info.st_size

    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return '%s %s' % (str(num), str(x) + "\n")
        num /= 1024.0


# ----------------------------------------------------------------------------------------------------------------------
# Testing TF Lite Model

def test_tflite_model(model_path, figpath, testing_labels, testing_data, title, tflite_type, TextFile, plot_number):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    sns.set()

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    num_correct = 0
    num_incorrect = 0
    classified = []
    actual = []
    num_total = len(testing_data)

    print("Testing with {} clips...".format(num_total))
    pprint(input_details)
    for i in tqdm(range(num_total)):
        # Test model on random input data.
        input_data = np.array([testing_data[i]], dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])

        output_class = np.where(output_data[0] == np.amax(output_data[0]))
        actual_class = np.where(testing_labels[i] == np.amax(testing_labels[i]))

        classified.append(output_class[0][0])
        actual.append(actual_class[0][0])

    conf_matrix = confusion_matrix(actual, classified)
    #     print(conf_matrix)

    # Create confusion matrix
    num_incorrect = 0
    for i in range(len(classified)):
        if classified[i] == actual[i]:
            num_correct += 1
        else:
            num_incorrect += 1

    import matplotlib
    import matplotlib.pyplot as plt

    print("Tensor Flow Lite Model Accuracy: {0:.3f}%".format(num_correct / num_total * 100))
    messagetextfile = (str(tflite_type) + "Tensor Flow Lite Model Accuracy: {0:.3f}%".format(num_correct / num_total * 100) + "\n")
    TextFile.write(messagetextfile)
    TextFile.write(file_size(model_path) + (('*' * 100) + "\n"))
    heatmap = sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='g', cbar=False, square=True)
    plot_number = heatmap.get_figure()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    heatmap.set_xticklabels(["KT", "CO", "FS", "KJ", "KN", "LA"])
    heatmap.set_yticklabels(["KT", "CO", "FS", "KJ", "KN", "LA"], rotation=0)
    plot_number.savefig(figpath, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------

# # Function to test the file accuracy(TF lite)
# print(" Pruned Model Testing")

# test_tflite_model(
#     "/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/tensorflowlitemodel/amplitude-model-tensorflowlite.tflite",
#     "/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/amp_lite.pdf", "Amplitude", "Pruned ")


# print("Non Pruned Model Testing")

# test_tflite_model(
#     "/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/tensorflowlitemodel/amplitude-model-nonpruned-tensorflowlite.tflite",
#     "/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/amp_lite.pdf", "Amplitude", "Non-Pruned ")


# --------------------------------------------------------------------------------------------------------------------
# Code to Generate Prediction Result

def test_reg_model(model_path, figpath, testing_labels, testing_data, TextFile, plot_number):
    print("entering")
    # Load TFLite model and allocate tensors.
    model_new = tf.keras.models.load_model(model_path)

    print("Loaded the model")

    num_correct = 0
    num_incorrect = 0
    classified = []
    actual = []
    num_total = len(testing_data)

    print("Testing with {} clips...".format(num_total))
    for i in tqdm(range(num_total)):
        output_data = model_new.predict(np.array([testing_data[i]]))

        output_class = np.where(output_data[0] == np.amax(output_data[0]))
        actual_class = np.where(testing_labels[i] == np.amax(testing_labels[i]))

        classified.append(output_class[0][0])
        actual.append(actual_class[0][0])
    #         print("Incorrect")
    #         print(output_class[0])
    #         print(actual_class[0])
    #         print(output_data[0])
    #         print(y_test[0])

    conf_matrix = confusion_matrix(actual, classified)
    # print(classified)
    # print(actual)
    print(conf_matrix)

    for i in range(len(classified)):
        if classified[i] == actual[i]:
            num_correct += 1
        else:
            num_incorrect += 1

    print("Accuracy Of Original Model: {0:.3f}%".format(num_correct / num_total * 100))
    messagetextfile = ("Accuracy Of Original Model: {0:.3f}%".format(num_correct / num_total * 100) + "\n")
    TextFile.write(messagetextfile)
    TextFile.write(file_size(model_path) + (('*' * 100) + "\n"))

    heatmap = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False, square=True)
    fig = heatmap.get_figure()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Original")
    plt.tight_layout()
    heatmap.set_xticklabels(["KT", "CO", "FS", "KJ", "KN", "LA"])
    heatmap.set_yticklabels(["KT", "CO", "FS", "KJ", "KN", "LA"], rotation=0)
    fig.savefig(figpath)
    plt.show()


# ---------------------------------------------------------------------------------------------------------------------

# test_reg_model("/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/amplitude-model.hdf5",
#                "/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/amp_reg.svg")

# ---------------------------------------------------------------------------------------------------------------------
