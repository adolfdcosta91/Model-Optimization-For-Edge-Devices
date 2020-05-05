import tensorflow as tf

def to_tf_quant(infile, outfile):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(infile)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tfmodel = converter.convert()
    open(outfile, "wb").write(tfmodel)

# ----------------------------------------------------------------------------------------------------------------------

# Converting from Tensor flow to Tensor flow lite

# frozen_model_name = "/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/prunedmodel/amplitude-model-prune.hdf5"
# non_pruned_model = "/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/amplitude-model.hdf5"
# tf_lite_model_name = "/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/tensorflowlitemodel/amplitude-model-tensorflowlite.tflite"
# tf_lite_model_name_non_pruned = "/home/ben/Desktop/AdolfThesisShrink/ThesisShrink/model/tensorflowlitemodel/amplitude-model-nonpruned-tensorflowlite.tflite"


# For Manually Using in this file


# print("About to convert!")
# to_tf_quant(frozen_model_name, tf_lite_model_name)
# to_tf_quant(non_pruned_model, tf_lite_model_name_non_pruned)
