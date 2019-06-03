#USAGE
#crop_faces.py -i input_images_directory -o output_images_directory


#!/usr/bin/env python
# coding: utf-8

# In[1]:

from imutils import paths
import numpy as np
import argparse
import cv2
import os
import shutil

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,	help="path to pre-trained model")
# ap.add_argument("-l", "--labels", required=True, help="path to class labels")
ap.add_argument("-i", "--input", required=True, help="path to directory containing input images")
ap.add_argument("-o", "--output", required=True, help="path to directory to store output images")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

outputPath = args["output"]

try:
	shutil.rmtree(outputPath)
except:
	pass

os.mkdir(outputPath)
imagePaths = list(paths.list_images(args["input"]))

# In[2]:


face_classifier = cv2.CascadeClassifier("Haarcascades/haarcascade_frontalface_default.xml")


# In[3]:


def face_detector(img,size=0.5):
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale (gray, 1.3, 5)
    if faces is ():
        return img
    for (x, y, w, h) in faces:
        x = x - 10
        w = w + 10
        y = y - 10
        h = h + 10
        cv2.rectangle (img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y: y+h, x: x+w]
        roi_color = img[y: y+h, x: x+w]
    # return img
    return roi_color


# In[4]:


# cap = cv2.VideoCapture (0)
for imagePath in imagePaths:
	# ret, frame = cap.read ()
	frame = cv2.imread(imagePath)
	result = face_detector(frame)
	result= cv2.resize(result,(500,500))
	cv2.imshow ("Our Face Extractor", result)
	filename = imagePath.split(os.path.sep)[-1]
	outputPath = os.path.sep.join([args["output"], filename])
	cv2.imwrite(outputPath,result)
	cv2.imshow("Original",frame)
	cv2.waitKey(0)
	# if cv2.waitKey (1) == 13: #13 is the Enter Key
        # break


# In[5]:


# cap.release ()
cv2.destroyAllWindows ()
