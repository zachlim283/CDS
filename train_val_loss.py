import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

# ===================================== Parameters =====================================
BATCH_SIZE = 32
LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
METRICS = [tf.metrics.SparseCategoricalAccuracy()]
EPOCHS = 50
LR = 0.0001

tf_hub_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/2'
tf_hub_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

checkpoint_path = "Saved_Models/small_bert_L12_H768_1_ES"

# ======================================= Admin ========================================
gpu_avail = "No" if len(tf.config.list_physical_devices('GPU')) == 0 else "Yes"
print(f"GPU Available?: {gpu_avail}")

print("Preparing dataset...")
with open("DialogueRNN_features/MELD_features/MELD_features_raw.pkl", 'rb') as f:
    de_pickled = pickle.load(f, encoding='latin1')
f.close()

videoIDs, videoSpeakers, _, videoText, \
    videoAudio, videoSentence, trainVid, \
    testVid, videoLabels = de_pickled

train_keys = [x for x in trainVid]
test_keys = [x for x in testVid]

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

# ===================================== Build Model ====================================
print("Building Model...")


def sentiment_classifier():
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Input')
    preprocessing_layer = hub.KerasLayer(tf_hub_preprocess, name='Preprocessing')
    encoder_inputs = preprocessing_layer(input_layer)
    encoder = hub.KerasLayer(tf_hub_encoder, trainable=True, name='BERT_Encoder')
    outputs = encoder(encoder_inputs)
    x = outputs['pooled_output']
    x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_1')(x)
    x = tf.keras.layers.Dense(256, activation=tf.keras.activations.selu, name='Selu')(x)
    x = tf.keras.layers.Dropout(0.5, name='Dropout_0.5_2')(x)
    x = tf.keras.layers.Dense(3, activation=tf.keras.activations.softmax, name='Classifier')(x)
    return tf.keras.Model(input_layer, x)


sc_model = sentiment_classifier()
print("Complete!")


# ====================================== Callbacks =====================================
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
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

sc_model.compile(optimizer=optimizer,
                 loss=LOSS,
                 metrics=METRICS)

print(f'Training model with {tf_hub_encoder}')
history = sc_model.fit(x=tr_val_text_flat,
                       y=tr_val_labels_flat,
                       validation_split=0.2,
                       shuffle=True,
                       callbacks=callbacks,
                       epochs=EPOCHS,
                       verbose=1)
print("Complete!")

# =================================== Evaluate Model ===================================
history_dict = history.history

acc = history_dict['sparse_categorical_accuracy']
val_acc = history_dict['val_sparse_categorical_accuracy']
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

plt.savefig(checkpoint_path + '/loss_graph.png')

loaded_model = keras.models.load_model(checkpoint_path)
loss, accuracy = loaded_model.evaluate(test_ds)

predicted = np.array([np.argmax(x) for x in loaded_model.predict(test_ds, verbose=1)])

with open(checkpoint_path + '/modelsummary.txt', 'w') as f:
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
