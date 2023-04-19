import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import pickle


# with open("DialogueRNN_features/MELD_features/MELD_features_raw.pkl", 'rb') as f:
#     de_pickled = pickle.load(f, encoding='latin1')
# f.close()
#
# videoIDs, videoSpeakers, _, videoText, \
#     videoAudio, videoSentence, trainVid, \
#     testVid, videoLabels = de_pickled
#
# print(testVid)
# print(trainVid)

# for files in os.listdir("dev_features"):
#     filename_old = files.split(".")[0].split("_")
#
#     dia_no_old = int(filename_old[0][3:])
#     utt_no = filename_old[1][3:]
#
#     filename_new = f"dia{str(dia_no_old + 1039)}_utt{utt_no}.pickle"
#
#     filepath_old = f"/home/zach/PycharmProjects/CDS/dev_features/{files}"
#     filepath_new = f"/home/zach/PycharmProjects/CDS/dev_emb/{filename_new}"
#
#     shutil.copy(filepath_old, filepath_new)


# for files in os.listdir("test_features"):
#     filename_old = files.split(".")[0].split("_")
#
#     dia_no_old = int(filename_old[0][3:])
#     utt_no = filename_old[1][3:]
#
#     filename_new = f"dia{str(dia_no_old + 1153)}_utt{utt_no}.pickle"
#
#     filepath_old = f"/home/zach/PycharmProjects/CDS/test_features/{files}"
#     filepath_new = f"/home/zach/PycharmProjects/CDS/test_emb/{filename_new}"
#
#     shutil.copy(filepath_old, filepath_new)

# test_1 = open("/home/zach/PycharmProjects/CDS/combined_emb/dia0_utt0.pickle", mode='rb')
# print(pickle.load(test_1))

# test_2 = open("/home/zach/PycharmProjects/CDS/combined_emb/dia1432_utt15.pickle", mode='rb')
# for k, v in pickle.load(test_2).items():
#     print(k, v)


val_keys = [x for x in range(1039, 1153)]
print(val_keys)