import cv2
import numpy as np
class Face_detect():
    name_video=[]
    def __init__(self,name_video):
        self.name_video=name_video


    def crop_face(self,path_video,index):
        if (index<10):
            index2='0'+str(index)
        else:
            index2=str(index)

        print(path_video)
        cap = cv2.VideoCapture('Video/'+str(path_video)+'.mp4')

        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        i=0
        f = open("y_train.txt", "a")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    faces = frame[y:y + h, x:x + w]
                    if(h<150 or w<150): continue
                    if(i%5==0):
                        faces=cv2.resize(faces,dsize=(224,224))
                        cv2.imwrite('Images_face_train/'+index2+'_face'+str(i//5)+'.jpg', faces)
                        f.write(str(index)+' ')
                    i+=1

            # Break the loop
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        f.close()


    def run_crop_face(self):
        for i in range(0,len(self.name_video)):
            name=self.name_video[i]
            self.crop_face(name,i)



