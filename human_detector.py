import cv2
video =cv2.VideoCapture('in.avi')
human_cascade =cv2.CascadeClassifier('haarcascade_fullbody (1).xml')
while True:
    check,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    human = human_cascade.detectMultiScale(gray,1.9,1)
    for (x,y,w,h) in human:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('human_detection',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
video.release()
cv2.destroyAllWindows()