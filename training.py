import os
import cv2 
import numpy as np 
from PIL import Image
recognizer=cv2.face.LBPHFaceRecognizer_create() 
path='C:/Users/SURAJ/Desktop/project face detection/data'
def getimagesWithID(path):
    imagesPaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagesPaths:
        faceImg=Image.open(imagePath).convert('1')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print (ID)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return IDs, faces

Ids, faces=getimagesWithID(path)
recognizer.train(faces,np.array(Ids)) 
recognizer.save('C:/Users/SURAJ/Desktop/project face detection/trainingdata.yml')
cv2.destroyAllWindows()       