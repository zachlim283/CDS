import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split


# ===================================== Parameters =====================================
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.00001
ES_PATIENCE = 5
LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
METRICS = [tf.metrics.SparseCategoricalAccuracy()]
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR)

CHECKPOINT_PATH = "Saved_Models/combined_model_vgg_1"

tf_hub_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2'
tf_hub_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'


# ======================================== Admin =======================================
gpu_avail = "No" if len(tf.config.list_physical_devices('GPU')) == 0 else "Yes"
print(f"GPU Available?: {gpu_avail}")

if os.path.isdir(CHECKPOINT_PATH):
    overwrite = input("Model already exists. Overwrite? (Y/N)")
    if overwrite.lower() != "y":
        print("Quitting...")
        exit()

    print("Continuing...")

print("Preparing dataset...")
with open("DialogueRNN_features/MELD_features/MELD_features_raw.pkl", 'rb') as f:
    de_pickled = pickle.load(f, encoding='latin1')
f.close()

videoIDs, videoSpeakers, _, videoText, \
    videoAudio, videoSentence, trainVid, \
    testVid, videoLabels = de_pickled

train_keys = [x for x in trainVid]
test_keys = [x for x in testVid]

# For VGGish audio features only
# val_keys = [x for x in range(1039, 1153)]


# ================================== Prepare Text Dataset ==================================
# Train + Val Datasets
tr_val_text = [videoSentence[x] for x in train_keys]
tr_val_labels = [videoLabels[x] for x in train_keys]

tr_val_text_flat = np.array([item for sublist in tr_val_text for item in sublist], dtype=object)
tr_val_labels_flat = np.array([item for sublist in tr_val_labels for item in sublist], dtype=int)

train_text_flat, val_text_flat, train_labels_flat, val_labels_flat = train_test_split(tr_val_text_flat,
                                                                                      tr_val_labels_flat,
                                                                                      random_state=0)

train_ds = tf.data.Dataset.from_tensor_slices((train_text_flat, train_labels_flat)).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((val_text_flat, val_labels_flat)).batch(BATCH_SIZE)

combined_ds = tf.data.Dataset.from_tensor_slices((tr_val_text_flat, tr_val_labels_flat)).batch(BATCH_SIZE)

# Test Dataset
test_text = [videoSentence[x] for x in test_keys]
test_labels = [videoLabels[x] for x in test_keys]

test_text_flat = np.array([item for sublist in test_text for item in sublist], dtype=object)
test_labels_flat = np.array([item for sublist in test_labels for item in sublist], dtype=int)

test_ds = tf.data.Dataset.from_tensor_slices((test_text_flat, test_labels_flat)).batch(BATCH_SIZE)


# =============================== Audio Data for LogRegress ============================
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

audio_input_shape = (42, 16,)
output_shape = (None, 1)
y_train, y_val, y_test = oneHot(y_train, y_val, y_test)


# ======================== Audio Data from VGGish Feature Embeddings ====================
# since data is occassionally corrupted, will be using just the subset that is clean!
def get_vggish_embeddings(filename):
    with open(filename, 'rb') as f:
        vggish_emb = pickle.load(f, encoding='ASCII')
    f.close()

    decoded_list = []
    value_table = vggish_emb.feature_lists.feature_list["audio_embedding"].feature
    for i in range(len(value_table)):
        decoded = np.frombuffer(value_table[i].bytes_list.value[0])
        decoded_list.append(decoded.tolist())

    return decoded_list

vggish_embeddings_train = []
vggish_train_keys = []

for dialogue_id in videoLabels.keys():
    for utterance_id in range(len(videoLabels[dialogue_id])):
        filename = f'./train_emb/dia{dialogue_id}_utt{utterance_id}.pickle'

        try:
            decoded_list = get_vggish_embeddings(filename)
            if len(decoded_list) == 0:
                continue
            vggish_embeddings_train.append(decoded_list)
            vggish_train_keys.append((dialogue_id, utterance_id))

        except:
            print(f"{filename} needs to be deleted")

# if empty or file not avail, skip!
# print(np.asarray(vggish_embeddings_train).shape) 

# ================================= get input embeddings ==================================

# use text_train, vgg_train, labels
# use text_val, vgg_val, val_labels
# use text_test, vgg_test, test_labels

vggish_embeddings_train = [(x + [[0]*16]*(42-len(x))) for x in vggish_embeddings_train] # add padding to ensure correct size
labels_flat = np.array([videoLabels[x[0]][x[1]] for x in vggish_train_keys])
text_flat = np.array([videoSentence[x[0]][x[1]] for x in vggish_train_keys])

# use traintestsplit to get val and test split
vgg_train_embedding = np.asarray(vggish_embeddings_train)
vgg_train2, vgg_test, labels2_flat, test_labels = train_test_split(vgg_train_embedding, labels_flat, random_state=0)
vgg_train, vgg_val, labels, val_labels = train_test_split(vgg_train2, labels2_flat, random_state=0)

text_train2, text_test, _, _ = train_test_split(text_flat, labels_flat, random_state=0)
text_train, text_val, _, _ = train_test_split(text_train2, labels2_flat, random_state=0)



# ============================== Create combined dataset ==================================

# # Datasets using LogRegress for Audio
# speech_train_ds = tf.data.Dataset.from_tensor_slices(
#     ({"audio": X_train.astype(np.float64), "text": train_text_flat}, y_train)).batch(BATCH_SIZE)
# speech_test_ds = tf.data.Dataset.from_tensor_slices(
#     ({"audio": X_test.astype(np.float64), "text": test_text_flat}, y_test)).batch(BATCH_SIZE)
# speech_val_ds = tf.data.Dataset.from_tensor_slices(
#     ({"audio": X_val.astype(np.float64), "text": val_text_flat}, y_val)).batch(BATCH_SIZE)

# Datasets using VGGish Embeddings for Audio
speech_train_ds = tf.data.Dataset.from_tensor_slices((
    {"audio": vgg_train.astype(np.float64), "text": text_train}, labels)).batch(BATCH_SIZE)
speech_test_ds = tf.data.Dataset.from_tensor_slices((
    {"audio": vgg_test.astype(np.float64), "text": text_test}, test_labels)).batch(BATCH_SIZE)
speech_val_ds = tf.data.Dataset.from_tensor_slices((
    {"audio": vgg_val.astype(np.float64), "text": text_val}, val_labels)).batch(BATCH_SIZE)

print("Datasets Generated!")
print("Building Models...")


# ================================ Bert model ==============================
def sentiment_classifier():
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Input')

    preprocessing_layer = hub.KerasLayer(tf_hub_preprocess, name='Preprocessing')
    encoder_inputs = preprocessing_layer(input_layer)
    encoder = hub.KerasLayer(tf_hub_encoder, trainable=True, name='BERT_Encoder')
    outputs = encoder(encoder_inputs)
    x = outputs['pooled_output']
    # x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_1')(x)
    # x = tf.keras.layers.Dense(256, activation=tf.keras.activations.selu, name='Selu')(x)
    # x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_2')(x)
    # x = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax, name='Classifier')(x)
    return tf.keras.Model(input_layer, x)


text_model = sentiment_classifier()


# ===================== Logistic regression for pretrained Audio features ===================
audio_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(audio_input_shape)),
    # tf.keras.layers.SimpleRNN(128, input_shape=(input_shape),return_sequences=True),
    # tf.keras.layers.Dense(64, activation=tf.keras.activations.selu, name='Selu'),
    # tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(rate=0.5),
    # tf.keras.layers.Dense(3, activation='softmax')
])


# =================================  Combined Model ==================================
def concatenated_model(text_model=text_model, audio_model=audio_model):
    # get encoders
    # get embedding projections:
    audio_input_shape = (42, 16,)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    audio_input = tf.keras.layers.Input(shape=audio_input_shape, name='audio')

    text_projections = text_model(text_input)
    audio_projections = audio_model(audio_input)

    # # Cross-attention (Luong-style).
    # query_value_attention_seq = tf.keras.layers.Attention(use_scale=True, dropout=0.2)(
    # [text_projections, audio_projections]
    # )

    # Concatenate features and classify
    concatenated = tf.keras.layers.Concatenate()([text_projections, audio_projections])
    # contextual = keras.layers.Concatenate()([concatenated, query_value_attention_seq])
    x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_1')(concatenated)
    x = tf.keras.layers.Dense(512, activation=tf.keras.activations.selu, name='Selu_1')(x)
    x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_2')(x)
    x = tf.keras.layers.Dense(256, activation=tf.keras.activations.selu, name='Selu_2')(x)
    x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_3')(x)
    x = tf.keras.layers.Dense(64, activation=tf.keras.activations.selu, name='Selu_3')(x)
    outputs = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax, name='Classifier')(x)

    return tf.keras.Model([text_input, audio_input], outputs)


# ====================================== Callbacks =====================================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=ES_PATIENCE,
        restore_best_weights=True
    )]

# ===================================== Train Model ====================================
print("Models Built!")
print("Starting Training...")
speech_model = concatenated_model(text_model, audio_model)

speech_model.compile(optimizer=OPTIMIZER,
                     loss=LOSS,
                     metrics=METRICS)

print(f'Training model with {tf_hub_encoder}')
history = speech_model.fit(speech_train_ds,
                           shuffle=True,
                           callbacks=callbacks,
                           epochs=EPOCHS,
                           verbose=1,
                           validation_data=speech_val_ds)

print("Training Complete!")


# =================================== Evaluate Model ===================================
tf.keras.utils.plot_model(speech_model,
                          show_shapes=True,
                          show_layer_activations=True,
                          to_file=CHECKPOINT_PATH + '/combined_model.png')

history_dict = history.history

acc = history_dict['categorical_accuracy']
val_acc = history_dict['val_categorical_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.savefig(CHECKPOINT_PATH + '/loss_graph.png')

loaded_model = tf.keras.models.load_model(CHECKPOINT_PATH)
loss, accuracy = loaded_model.evaluate(speech_test_ds)

predicted = np.array([np.argmax(x) for x in loaded_model.predict(speech_test_ds, verbose=1)])

with open(CHECKPOINT_PATH + '/modelsummary.txt', 'w') as f:
    loaded_model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write(f'\nModel trained using {tf_hub_encoder}\n')
    f.write(f'\n===== Parameters =====\n')
    f.write(f'Batch Size: {BATCH_SIZE}\nEpochs: {len(acc)}/{EPOCHS}\nLearning Rate: {LR}\n')
    f.write(f'\n====== Results =======\n')
    f.write(metrics.classification_report(test_labels_flat, predicted, digits=3))
    f.write(f"\nLoss: {loss}")
f.close()

print(f'Loss: {loss}')

plt.show()

# =================================== Visualisation ====================================
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix

# values for scatter plot will be obtained using softmax probability table
intermediate_layer_model = tf.keras.Model(inputs=speech_model.input,
                                          outputs=speech_model.get_layer('Classifier').output)
intermediate_output = intermediate_layer_model.predict(speech_test_ds)

df = pd.DataFrame(intermediate_output)
fig = px.scatter_3d(df, x=0,
                        y=1,
                        z=2)

fig.show()

# Confusion matrix
# predict_class = np.argmax(intermediate_output, axis=1)
# cm = confusion_matrix(y_test, predict_class)

# print(cm)
