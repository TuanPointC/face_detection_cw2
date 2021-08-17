import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from numpy.linalg import norm
import pickle

class Run_model():

    def __init__(self):
        pass

    def get_embedding(model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    def run(self,path_video):

        #cap = cv2.VideoCapture('Video/'+str(path_video)+'.mp4')
        cap = cv2.VideoCapture(0)

        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        name=["Duc","HDuc","Hieu","Hung","Kien","Linh","Quan","Tan","Thang","Truong","Tuan","Van","Viet Duc","Xuan Anh"]
        facenet= load_model('Save_model/facenet.h5')
        svm= 'Save_model/model.sav'
        with open(svm, 'rb') as file:  
            svm_model = pickle.load(file)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    if (w<150 or h<150): continue
                    faces = frame[y:y + h, x:x + w]
                    faces=cv2.resize(faces,dsize=(160,160))

                    y_pre=self.get_embedding(facenet,faces)
                    y_pre=y_pre/norm(y_pre)
                    y_pre=y_pre.reshape(1,-1)
                    y_hat=svm_model.predict(faces)
                    y_hat=name[np.argmax(y_hat)]

                    cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0),thickness=4)
                    if(y_hat<0.5):
                        frame =cv2.putText(frame, 'unknown', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)
                    else:
                        frame =cv2.putText(frame, y_hat, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)

                cv2.imshow('video',frame)
                key = cv2.waitKey(10)
                #if Esc key is press then break out of the loop 
                if key == 27: #The Esc key
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()




