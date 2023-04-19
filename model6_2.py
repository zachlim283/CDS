# ! py -m pip install tensorflow_hub
# ! py -m pip install tensorflow_text
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split

gpu_avail = "No" if len(tf.config.list_physical_devices('GPU')) == 0 else "Yes"
print(f"GPU Available?: {gpu_avail}")

print("Preparing dataset...")
with open("../multi_modal/DialogueRNN_features/MELD_features/MELD_features_raw.pkl", 'rb') as f:
    de_pickled = pickle.load(f, encoding='latin1')
f.close()

videoIDs, videoSpeakers, _, videoText, \
    videoAudio, videoSentence, trainVid, \
    testVid, videoLabels = de_pickled

train_keys = [x for x in trainVid]
test_keys = [x for x in testVid]
# ============================= **Zach's Dataset Cleaning** =========================
BATCH_SIZE = 32
# LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# METRICS = [tf.metrics.SparseCategoricalAccuracy()]
LOSS = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
METRICS = [tf.metrics.CategoricalAccuracy()]
EPOCHS = 50
LR = 0.0001

# ================================== Prepare Datasets ==================================
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
print("Complete!")

# ================================== Kevin's Datasets Cleaning ==================================
def oneHot(trainLabels, valLabels, testLabels):
	
	# Calculate the total number of classes
	numOfClasses = np.max(trainLabels)+1
	
	trainLabelOneHot = np.zeros((len(trainLabels),numOfClasses))
	valLabelOneHot = np.zeros((len(valLabels),numOfClasses))
	testLabelOneHot = np.zeros((len(testLabels),numOfClasses))

	for idx, label in enumerate(trainLabels):
		trainLabelOneHot[idx, int(label)]=1.0
	for idx, label in enumerate(valLabels):
		valLabelOneHot[idx, int(label)]=1.0
	for idx, label in enumerate(testLabels):
		testLabelOneHot[idx, int(label)]=1.0

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

# split_idx = int(X.shape[0] * 0.8)
# X_main, X_test = X[:split_idx], X[split_idx:]
# y_main, y_test = y[:split_idx], y[split_idx:]
X_test, y_test = flatten_audio_and_labels(videoAudio, videoLabels, test_keys)

# max_length = max([(X.shape[0]) for x in X])
# input_shape = (max_length, 74)
batch_size = 32
num_epochs = 30

# ================= Split data into test train and val ==================
X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0)

audio_input_shape = (300,)
output_shape = (None, 1)
y_train, y_val, y_test = oneHot(y_train, y_val, y_test)
# train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
# val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
# test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# ====================== Extract vggish embeddings from train_emb.zip ==============
# ** Changed here
from tensorflow.core.example import example_pb2
import numpy as np

def get_vggish_embeddings(filename):
    with open('./train_emb/dia0_utt0.pickle', 'rb') as f:
        vggish_emb = pickle.load(f, encoding='ASCII')
    f.close()
    # print(vggish_emb)

    decoded_list = []
    value_table = vggish_emb.feature_lists.feature_list["audio_embedding"].feature
    # print(value_table)
    for i in range(len(value_table)):
        decoded = np.frombuffer(value_table[i].bytes_list.value[0])
        decoded_list.append(decoded.tolist())

    # print(decoded_list)
    return decoded_list

vggish_embeddings_train = []
vggish_embeddings_test = []
for dialogue_id in videoLabels.keys():
    for utterance_id in range(len(videoLabels[dialogue_id])):
        filename = f'./train_emb/dia{dialogue_id}_utt{utterance_id}.pickle'
        try:
            decoded_list = get_vggish_embeddings(filename)
            if dialogue_id in train_keys:
                vggish_embeddings_train.append(decoded_list)
            elif dialogue_id in test_keys:
                vggish_embeddings_test.append(decoded_list)
        except:
            print(f"{filename} needs to be deleted")

# print(np.asarray(vggish_embeddings_train).shape)
# print(np.asarray(vggish_embeddings_test).shape)

print("completed!")

vgg_train = np.asarray(vggish_embeddings_train)
vgg_test = np.asarray(vggish_embeddings_test)
vgg_train, vgg_val, train_labels_flat, val_labels_flat = train_test_split(vgg_train,
                                                                        tr_val_labels_flat,
                                                                        random_state=0)


# ============================== Create combined dataset ==================================
# print(len(X_train))
# print(len(train_text_flat))
# print(len(train_labels_flat))
# print(len(y_train))

# ** changed here
# if using linear regressor, bcLSTM model
# speech_train_ds = tf.data.Dataset.from_tensor_slices(({"audio": X_train.astype(np.float64), "text": train_text_flat}, y_train)).batch(BATCH_SIZE)
# speech_test_ds = tf.data.Dataset.from_tensor_slices(({"audio":X_test.astype(np.float64), "text":test_text_flat}, y_test)).batch(BATCH_SIZE)
# speech_val_ds = tf.data.Dataset.from_tensor_slices(({"audio:":X_val.astype(np.float64), "text":val_text_flat}, y_val)).batch(BATCH_SIZE)

# if using vggish 
speech_train_ds = tf.data.Dataset.from_tensor_slices(({"audio": vgg_train.astype(np.float64), "text": train_text_flat}, y_train)).batch(BATCH_SIZE)
speech_test_ds = tf.data.Dataset.from_tensor_slices(({"audio":vgg_test.astype(np.float64), "text":test_text_flat}, y_test)).batch(BATCH_SIZE)
speech_val_ds = tf.data.Dataset.from_tensor_slices(({"audio:":vgg_val.astype(np.float64), "text":val_text_flat}, y_val)).batch(BATCH_SIZE)

print(X_train.astype(np.float64))

print(train_text_flat)
# ================================ Bert model ==============================
tf_hub_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2'
tf_hub_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

def sentiment_classifier():
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Input')
    
    preprocessing_layer = hub.KerasLayer(tf_hub_preprocess, name='Preprocessing')
    encoder_inputs = preprocessing_layer(input_layer)
    encoder = hub.KerasLayer(tf_hub_encoder, trainable=True, name='BERT_Encoder')
    outputs = encoder(encoder_inputs)
    x = outputs['pooled_output']
    x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_1')(x)
    x = tf.keras.layers.Dense(256, activation=tf.keras.activations.selu, name='Selu')(x)
    # x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_2')(x)
    # x = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax, name='Classifier')(x)
    return tf.keras.Model(input_layer, x)

# sc_model = sentiment_classifier()

text_model = sentiment_classifier()
print("Complete!")

# text_model = sentiment_classifier()

# ===================== Logistic regression for pretrained Audio features ===================
audio_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(audio_input_shape)),
    #tf.keras.layers.SimpleRNN(128, input_shape=(input_shape),return_sequences=True),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(rate=0.2)
    # tf.keras.layers.Dense(3, activation='softmax')
])
# =====================-===========  Combined Model ==================================

from keras.models import *
import keras

def concatenated_model(text_model=text_model, audio_model=audio_model):
    # get encoders
    # get embedding projections:
    # ** changed here
    # audio_input_shape = (300,)
    audio_input_shape = (5,16,)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    audio_input = tf.keras.layers.Input(shape=(audio_input_shape), name='audio')

    text_projections = text_model(text_input)
    audio_projections = audio_model(audio_input)

    # # Cross-attention (Luong-style).
    # query_value_attention_seq = tf.keras.layers.Attention(use_scale=True, dropout=0.2)(
    # [text_projections, audio_projections]
    # )

    # concatenate, add more layers and classify
    concatenated = tf.keras.layers.Concatenate()([text_projections, audio_projections])
    # contextual = keras.layers.Concatenate()([concatenated, query_value_attention_seq])
    x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_1')(concatenated)
    x = tf.keras.layers.Dense(256, activation=tf.keras.activations.selu, name='Selu')(x)
    x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_2')(x)
    outputs = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax, name='Classifier')(x)
    
    return keras.Model([text_input, audio_input], outputs)

# ====================================== Callbacks =====================================
checkpoint_path = "Saved_Models/combined_model"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )]
# ===================================== Train Model ====================================
print("Starting Training...")
speech_model = concatenated_model(text_model, audio_model)
keras.utils.plot_model(speech_model, show_shapes=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

speech_model.compile(optimizer=optimizer,
                 loss=LOSS,
                 metrics=METRICS)

print(f'Training model with {tf_hub_encoder}')
history = speech_model.fit(speech_train_ds,
                       shuffle=True,
                       callbacks=callbacks,
                       epochs=EPOCHS,
                       verbose=1,
                       validation_data=speech_val_ds)
print("Complete!")

loaded_model = keras.models.load_model(checkpoint_path)
loss, accuracy = loaded_model.evaluate(speech_test_ds)
predicted = np.array([np.argmax(x) for x in loaded_model.predict(speech_test_ds, verbose=1)])

# =============================== visualisation =================================
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
from keras import backend as K

# values for scatter plot will be obtained using softmax probability table
intermediate_layer_model = Model(inputs=speech_model.input,
                                 outputs=speech_model.input.get_layer('Classifier').output)
intermediate_output = intermediate_layer_model.predict(speech_test_ds)

df = pd.DataFrame(intermediate_output)
fig = px.scatter_3d(df, x = '0',
                        y = '1',
                        z = '2',
                        color = 'sentiments')

fig.show()
# summary will be done by extracting the confusion matrix
predict_class = np.argmax(intermediate_output, axis=1)
cm =confusion_matrix(y_test, predict_class)

print(cm)
