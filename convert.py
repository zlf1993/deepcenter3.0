import tensorflow as tf
import sys
import os
ROOT_DIR = os.path.abspath("./")


# converter = tf.lite.TFLiteConverter.from_saved_model("./SAVE")
converter = tf.lite.TFLiteConverter.from_keras_model("./SAVE/weights.100-8.85.hdf5")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

