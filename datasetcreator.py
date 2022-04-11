import cv2
face_cascade = cv2.CascadeClassifier("C:\\python\\Lib\\site-packages\\cv2\\data\haarcascade_frontalface_default.xml")
video=cv2.VideoCapture(0)
id=input('enter user id')
samplenum=0;

while True:
    
    check, frame= video.read()
    gray_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
      samplenum=samplenum+1;
      file_path="C:/Users/SURAJ/Desktop/project face detection/data/user."+str(id)+"."+str(samplenum)+".jpg"
      cv2.imwrite(file_path,gray_img[y:y+h,x:x+w])
      cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),3)
      cv2.waitKey(100)
    cv2.imshow("capture", frame)
    cv2.waitKey(1)
    if samplenum>20:
       break

video.release()
cv2.destroyAllWindows()