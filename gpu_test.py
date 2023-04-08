import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("GPU Available?:", tf.config.list_physical_devices('GPU'))