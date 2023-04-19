import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import keras.models
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn import metrics


# ===================================== Parameters =====================================
BATCH_SIZE = 32
model_path = 'Saved_Models/small_bert_L12_H768_1_ES'


# ======================================= Admin ========================================
gpu_avail = "No" if len(tf.config.list_physical_devices('GPU')) == 0 else "Yes"
print(f"GPU Available?: {gpu_avail}")

print("Preparing test set...")
with open("DialogueRNN_features/MELD_features/MELD_features_raw.pkl", 'rb') as f:
    de_pickled = pickle.load(f, encoding='latin1')
f.close()

videoIDs, videoSpeakers, _, videoText, \
    videoAudio, videoSentence, trainVid, \
    testVid, videoLabels = de_pickled

test_keys = [x for x in testVid]


# ================================== Prepare Datasets ==================================
# Test Dataset
test_text = [videoSentence[x] for x in test_keys]
test_labels = [videoLabels[x] for x in test_keys]

test_text_flat = np.array([item for sublist in test_text for item in sublist], dtype=object)
test_labels_flat = np.array([item for sublist in test_labels for item in sublist], dtype=int)

test_ds = tf.data.Dataset.from_tensor_slices((test_text_flat, test_labels_flat)).batch(BATCH_SIZE)
print("Complete!")


# =================================== Evaluate Model ===================================
loaded_model = keras.models.load_model(model_path)
# loss, accuracy = loaded_model.evaluate(test_ds, verbose=1)

raw_predictions = loaded_model.predict(test_ds, verbose=1)
predicted_labels = np.array([np.argmax(x) for x in raw_predictions])

print(metrics.classification_report(test_labels_flat, predicted_labels, digits=3))

for raw, pred, true in zip(raw_predictions.tolist(), predicted_labels.tolist(), test_labels_flat.tolist()):
    print(type(raw), type(pred), type(true))
    print(raw, pred, true)


with open(model_path + '/raw_predictions.txt', 'w') as f:
    pass

f.close()

# print(f'Loss: {loss}')
# print(f'Accuracy: {accuracy}')
