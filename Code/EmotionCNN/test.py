# Emotion detection using fer2013 kaggle dataset
# the dataset was edited and 2 emotions were removed
# very simple example

# NOTE I plan on documenting more of this code.
# this was hacked together from other examples.

""" 
    WRITTEN by JP Aldama
    Quaxis Corporation for Research/Innovation (c) 2020
    https://github.com/sysad-aldama/
    quaxiscorp@gmail.com
"""    

from keras.models import load_model
from time import sleep 
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

# face classifier. Use absolute path
face_classifier = cv2.CascadeClassifier(
                  '/Users/jpaldama/bin/python/thebig4/machine_learning/'+
                  'emodetect/emo/haarcascade_frontalface_default.xml')

# classifier (our machine learning model)
classifier = load_model(
             '/Users/jpaldama/bin/python/thebig4/machine_learning/emodetect/emo/'+
             'Emotion_little_vgg.h5')

# our emotion labels
# NOTE Disgusted and Scared were removed
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']   

# where 0 is your default webcam on your machine. 
# if you have extra cameras you would like to use
# change 0 to 1 to use an external camera 
# refer to cv2 documentation to learn which number 
# corresponds with cameras 
# use python help(cv2) 

# I am using integrated webcam so its a 0
capture = cv2.VideoCapture(0) 

while True:
    # grab single frame, returns True
    ret, frame = capture.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3,5)
    
    #Draw rectangle on face detected
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) !=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            # make prediction on the ROI, then lookup the class 
            predictions= classifier.predict(roi)[0]
            label = class_labels[predictions.argmax()]
            label_position =(x,y)
            cv2.putText(frame, 
                        label,
                        label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else: 
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
    #show the text on the output window
    cv2.imshow('Emotion Detector', frame)

    # q key for exit
    # cv2.waitKey(0) is for images and first frame will show on window
    # cv2.waitKey(1) displays all frames and waits for certian key 
    # to be pressed 'q' 0xFF = 255 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Display exit message
        print('[EXITING] Quaxis Corporation for Research/Innovation'+
              '(c) 2020')
        break
    
# Close webcam and close main program
capture.release()
cv2.destroyAllWindows()

