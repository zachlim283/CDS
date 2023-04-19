from tensorflow.core.example import example_pb2
import numpy as np


# ================================= get vggish embeddings ===================
def get_vggish_embeddings(filename):
    with open(filename, 'rb') as f:
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
# vggish_embeddings_test = []
vggish_train_keys = []
# vggish_test_keys = []
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

print(np.asarray(vggish_embeddings_train).shape)
# print(np.asarray(vggish_embeddings_test).shape)

print("completed!")


# ================================= get input embeddings ==================================

labels_flat = np.array([videoLabels[x[0]][x[1]] for x in vggish_train_keys])
text_flat = np.array([videoText[x[0]][x[1]] for x in vggish_train_keys])

# use traintestsplit to get val and test split
vgg_train_embedding = np.asarray(vggish_embeddings_train)
vgg_train2, vgg_test, labels2_flat, test_labels = train_test_split(vgg_train_embedding, labels_flat,random_state=0)
vgg_train, vgg_val, labels, val_labels = train_test_split(vgg_train2,labels2_flat,random_state=0)

text_train2, text_test, _, _ = train_test_split(text_flat, labels_flat,random_state=0)
text_train, text_val, _, _ = train_test_split(text_train2,labels2_flat,random_state=0)

print(len(text_train), len(vgg_train))

# use text_train, vgg_train, labels
# use text_val, vgg_val, val_labels
# use text_test, vgg_test, test_labels

# ===================================== convert to dataset ==============================
# if using linear regressor, bcLSTM model
# speech_train_ds = tf.data.Dataset.from_tensor_slices(({"audio": X_train.astype(np.float64), "text": train_text_flat}, y_train)).batch(BATCH_SIZE)
# speech_test_ds = tf.data.Dataset.from_tensor_slices(({"audio":X_test.astype(np.float64), "text":test_text_flat}, y_test)).batch(BATCH_SIZE)
# speech_val_ds = tf.data.Dataset.from_tensor_slices(({"audio:":X_val.astype(np.float64), "text":val_text_flat}, y_val)).batch(BATCH_SIZE)

# if using vggish 
speech_train_ds = tf.data.Dataset.from_tensor_slices(({"audio": vgg_train.astype(np.float64), "text": text_train}, labels)).batch(BATCH_SIZE)
speech_test_ds = tf.data.Dataset.from_tensor_slices(({"audio":vgg_test.astype(np.float64), "text":text_test}, test_labels)).batch(BATCH_SIZE)
speech_val_ds = tf.data.Dataset.from_tensor_slices(({"audio:":vgg_val.astype(np.float64), "text":text_val}, val_labels)).batch(BATCH_SIZE)
