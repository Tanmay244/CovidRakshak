import cv2
import numpy as np
import os
import time
from datetime import datetime
from spreadsheetsFR import detectPatient

def livestream(names,status,l):

    c=1
      
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')   #load trained model
    cascadePath = "trainer/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #initiate id counter, the number of persons you want to include
    id = 1 

    time=['']
    date=['']
    for x in range(0,l):
        time.append('1')
        date.append('1')

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    set = {-1,0}

    while True:

        ret, img =cam.read()

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                pid = id
                set.update([pid])
                now = datetime.now()
                date[id] = now.strftime("%d/%m/%Y")
                time[id] = now.strftime("%H:%M:%S")
                
                confidence = "  {0}%".format(round(100 - confidence))


            if(str(status[id]=='Positive')):
                cv2.putText(img, str(status[id]), (x+5,y-30), font, 1, (0,0,255), 2)
                cv2.putText(img, str(names[id]), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            elif(str(status[id]=='At risk')):
                cv2.putText(img, str(status[id]), (x+5,y-30), font, 1, (0,0,255), 2)
                cv2.putText(img, str(names[id]), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            else:
                id1 = "Unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                cv2.putText(img, id1, (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.imshow('COVID Patient Tracking Live Feed',img) 

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    set.remove(0)
    set.remove(-1)

    for pid in set:
        detectPatient(pid,names[pid],date[pid],time[pid],status[pid])

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
