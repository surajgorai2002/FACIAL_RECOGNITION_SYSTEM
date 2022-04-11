import cv2
face_cascade = cv2.CascadeClassifier("C:\\python\\Lib\\site-packages\\cv2\\data\haarcascade_frontalface_default.xml")
video=cv2.VideoCapture(0)

while True:
    
    check, frame= video.read()
    gray_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
      cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),3)
    cv2.imshow("capture", frame)
    key=cv2.waitKey(1)
    if key== ord('q'):
       break

video.release()

cv2.destroyAllWindows()