from numpy.core.fromnumeric import ptp
from tensorflow.keras import models
from face_detect import Face_detect
import numpy as np
from tensorflow.keras.utils import to_categorical
from vgg16_model import vgg16_model
from tensorflow .keras.models import load_model

# fd=Face_detect(["Đức","HĐức","Hiếu","Hùng","Kiên","Linh","Quân","Tân","Thắng","Trường","Tuấn","Vân","Việt Đức","Xuân Anh"])
# fd.run_crop_face()

import glob
import random
import matplotlib.pyplot as plt

# get x_train and y_train
# filelist = glob.glob('Images_face_train/*.jpg')
# x_train = np.array([np.array(plt.imread(fname)) for fname in filelist])

# filelist_test = glob.glob('Images_face_test/*.jpg')
# x_test = np.array([np.array(plt.imread(fname)) for fname in filelist_test])

# f = open("y_train.txt", "r")
# y = f.read()
# y_train = np.array(y.split(" "))


#shulfer

# x_train1 = []
# y_train1 = []
# index_shuf = list(range(len(y_train)))
# random.shuffle(index_shuf)
# for i in index_shuf:
#     x_train1.append(x_train[i])
#     y_train1.append(y_train[i])

# f = open("y_test.txt", "r")
# y = f.read()
# y_test = np.array(y.split(" "))

# x_train = np.array(x_train1)
# y_train = np.array(y_train1)

# y_train=to_categorical(y_train,14)
# y_test=to_categorical(y_test,14)


# vgg=vgg16_model()
# vgg.fine_tune()
# H=vgg.train(x_train,y_train,x_test,y_test)

from run_model import Run_model

# run_model=Run_model()
# run_model.run('Đức')
# from tensorflow.keras.applications.vgg16 import preprocess_input

model = load_model('Save_model/facenet.h5')
model.summary()

# x_train=preprocess_input(x_train)


# for i in range(0,50):

#     test=x_train[i]
#     test=np.expand_dims(test,axis=-1)
#     test=test.reshape(1,224,224,3)
#     y_hat=model.predict(test)
#     print(np.argmax(y_hat))
#     print(y_train[i])
#     print('---------------')
