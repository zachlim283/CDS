import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import keras.models
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn import metrics
from sklearn.model_selection import train_test_split


# ===================================== Parameters =====================================
BATCH_SIZE = 32
MODEL_PATH = 'Saved_Models/combined_model_5'


# ======================================= Admin ========================================
gpu_avail = "No" if len(tf.config.list_physical_devices('GPU')) == 0 else "Yes"
print(f"GPU Available?: {gpu_avail}")

print("Preparing test dataset...")
with open("DialogueRNN_features/MELD_features/MELD_features_raw.pkl", 'rb') as f:
    de_pickled = pickle.load(f, encoding='latin1')
f.close()

videoIDs, videoSpeakers, _, videoText, \
    videoAudio, videoSentence, trainVid, \
    testVid, videoLabels = de_pickled

train_keys = [x for x in trainVid]
test_keys = [x for x in testVid]


# ================================== Prepare Datasets ==================================
# Text Data
test_text = [videoSentence[x] for x in test_keys]
test_labels = [videoLabels[x] for x in test_keys]

test_text_flat = np.array([item for sublist in test_text for item in sublist], dtype=object)
test_labels_flat = np.array([item for sublist in test_labels for item in sublist], dtype=int)

test_ds = tf.data.Dataset.from_tensor_slices((test_text_flat, test_labels_flat)).batch(BATCH_SIZE)

# Audio Data
def oneHot(trainLabels, valLabels, testLabels):
    # Calculate the total number of classes
    numOfClasses = np.max(trainLabels) + 1

    trainLabelOneHot = np.zeros((len(trainLabels), numOfClasses))
    valLabelOneHot = np.zeros((len(valLabels), numOfClasses))
    testLabelOneHot = np.zeros((len(testLabels), numOfClasses))

    for idx, label in enumerate(trainLabels):
        trainLabelOneHot[idx, int(label)] = 1.0
    for idx, label in enumerate(valLabels):
        valLabelOneHot[idx, int(label)] = 1.0
    for idx, label in enumerate(testLabels):
        testLabelOneHot[idx, int(label)] = 1.0

    return trainLabelOneHot, valLabelOneHot, testLabelOneHot


def flatten_audio_and_labels(audio, labels, train_keys):
    flattenAudio = [audio[x] for x in train_keys]
    flattenLabels = [labels[x] for x in train_keys]
    df_audio = np.array([item for sublist in flattenAudio for item in sublist], dtype=object)
    df_labels = np.array([item for sublist in flattenLabels for item in sublist], dtype=int)

    # flattenAudio = [y for x in audio.values() for y in x]
    # df_audio = np.array(flattenAudio)
    # flattenLabels = [y for x in labels.values() for y in x]
    # df_labels = np.array(flattenLabels)

    return df_audio, df_labels


X, y = flatten_audio_and_labels(videoAudio, videoLabels, train_keys)
X_test, y_test = flatten_audio_and_labels(videoAudio, videoLabels, test_keys)


# Split data into test train and val
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

audio_input_shape = (300,)
output_shape = (None, 1)
y_train, y_val, y_test = oneHot(y_train, y_val, y_test)

# Combined Dataset
speech_test_ds = tf.data.Dataset.from_tensor_slices(
    ({"audio": X_test.astype(np.float64), "text": test_text_flat}, y_test)).batch(BATCH_SIZE)

print("Datasets Generated!")


# =================================== Evaluate Model ===================================
print("Loading Model...")
loaded_model = keras.models.load_model(MODEL_PATH)

tf.keras.utils.plot_model(loaded_model,
                          show_shapes=True,
                          show_layer_activations=True,
                          to_file=MODEL_PATH + '/combined_model.png')

print("Running Evaluation...")
raw_predictions = loaded_model.predict(speech_test_ds, verbose=1)
predicted_labels = np.array([np.argmax(x) for x in raw_predictions])

print(metrics.classification_report(test_labels_flat, predicted_labels, digits=3))

df = pd.DataFrame(raw_predictions)
df['pred'] = predicted_labels
df['true'] = test_labels_flat
df['correct'] = np.where(df['pred'] == df['true'], df['true'], 3)
df['wrong'] = np.where(df['pred'] != df['true'], df['true'], 3)

fig1 = px.scatter_3d(df, x=0, y=1, z=2,
                     color='true', opacity=0.7, size_max=5,
                     color_continuous_scale=[(0.00, "rgb(0, 0, 102)"), (0.33, "rgb(0, 0, 102)"),
                                             (0.33, "rgb(204, 0, 0)"), (0.66, "rgb(204, 0, 0)"),
                                             (0.66, "rgb(255, 153, 0)"), (1.00, "rgb(255, 153, 0)")])

fig2 = px.scatter_3d(df, x=0, y=1, z=2,
                     color='correct', opacity=0.7, size_max=5,
                     color_continuous_scale=[(0.00, "rgb(0, 0, 102)"), (0.25, "rgb(0, 0, 102)"),
                                             (0.25, "rgb(204, 0, 0)"), (0.50, "rgb(204, 0, 0)"),
                                             (0.50, "rgb(255, 153, 0)"), (0.75, "rgb(255, 153, 0)"),
                                             (0.75, "rgb(179, 179, 179)"), (1.00, "rgb(179, 179, 179)")])

fig3 = px.scatter_3d(df, x=0, y=1, z=2,
                     color='wrong', opacity=0.7, size_max=5,
                     color_continuous_scale=[(0.00, "rgb(0, 0, 102)"), (0.25, "rgb(0, 0, 102)"),
                                             (0.25, "rgb(204, 0, 0)"), (0.50, "rgb(204, 0, 0)"),
                                             (0.50, "rgb(255, 153, 0)"), (0.75, "rgb(255, 153, 0)"),
                                             (0.75, "rgb(179, 179, 179)"), (1.00, "rgb(179, 179, 179)")])

fig1.show()
fig2.show()
fig3.show()
