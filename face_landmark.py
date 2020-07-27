import cv2
import dlib
import numpy as np
import os

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("D:/Projects/Python Projects/face_recognition/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

mood = input("Enter the expression name: ")

frames = []
outputs = []

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        #print(landmarks.parts())
        above_nose = landmarks.parts()[27]
        #print(above_nose.x, above_nose.y)
        
        expression = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks.parts()[17:]])
        
        
    #print(faces)
    if ret:
        cv2.imshow("Faces", frame)
    
    key = cv2.waitKey(1)
    
    if key == ord("q"):
        break
    elif key == ord("c"):
        frames.append(expression.flatten())
        outputs.append([mood])
        
X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, X])

f_name = "face_mood.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)

cap.release()
cv2.destroyAllWindows()