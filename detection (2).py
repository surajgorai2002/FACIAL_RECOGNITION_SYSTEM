import cv2
from datetime import datetime
face_cascade = cv2.CascadeClassifier("C:\\python\\Lib\\site-packages\\cv2\\data\haarcascade_frontalface_default.xml")
video=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create() 
rec.read('C:/Users/SURAJ/Desktop/project face detection/recognizer/trainingdata.yml')
id=0
font=cv2.FONT_HERSHEY_SIMPLEX
def attendance(id):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        idList = []
        for line in myDataList:
            entry = line.split(',')
            idList.append(entry[0])
        if id not in idList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{id},{tStr},{dStr}')

while True:
    
    check, frame= video.read()
    gray_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
      cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),3)
      id,conf=rec.predict(gray_img[y:y+h,x:x+w])
      if(id==1):
        id="Suraj Gorai"
      cv2.putText(frame,str(id),(x,y+h),font,1,(0,0,255),5);
      attendance(id)
    cv2.imshow("capture", frame)
    key=cv2.waitKey(1)
    if key== ord('q'):
       break

video.release()

cv2.destroyAllWindows()