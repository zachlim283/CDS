import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

gpu_avail = "No" if len(tf.config.list_physical_devices('GPU')) == 0 else "Yes"
print(f"GPU Available?: {gpu_avail}")
