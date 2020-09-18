from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime

print("[INFO] Importing modules completed!...")     

proto_txt_path = 'trainer/deploy.prototxt'
model_path = 'trainer/res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

mask_detector = load_model('trainer/mask_detector.model')
print("[INFO] Starting live video feed, Please wait...")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=700) 
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))

    face_detector.setInput(blob)
    detections = face_detector.forward()
 
    faces = []
    bbox = []
    results = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            bbox.append((startX, startY, endX, endY))

    if len(faces) > 0:
        results = mask_detector.predict(faces)

    for (face_box, result) in zip(bbox, results):
        (startX, startY, endX, endY) = face_box
        (mask, withoutMask) = result

        label = ""
        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, "Press 'Esc' to Exit", (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        if(label=="No Mask"):
            cv2.putText(frame, "PLEASE WEAR A MASK!", (5,70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Mask Detection Live Feed", frame)
        
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break


#clean up code
print("[INFO] Live stream has ended... ")
cap.release()
cv2.destroyAllWindows()


















        
        
