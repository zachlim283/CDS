import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import pickle


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