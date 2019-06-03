#USAGE
# python labeler.py -i input_image_folder -o output_image_folder -l label_file -t (Gender|Age) -c csv_file_to_store_data

from imutils import paths
import numpy as np
import argparse
import cv2
import os
import shutil
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,	help="path to pre-trained model")
ap.add_argument("-t", "--class_type", required=True, help="class type")
ap.add_argument("-l", "--labels", required=True, help="path to class labels")
ap.add_argument("-c", "--csv_file", required=True, help="path to image attributes csv file")
ap.add_argument("-i", "--input", required=True, help="path to directory containing input images")
ap.add_argument("-o", "--output", required=True, help="path to directory to store output images")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

LABELS = open(args["labels"]).read().strip().split("\n")
LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}

outputPath = args["output"]

try:
	shutil.rmtree(outputPath)
except:
	pass

os.mkdir(outputPath)
imagePaths = list(paths.list_images(args["input"]))

imagesLabels=[]

for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	cv2.imshow("Image",image)
	while(1):
		response = cv2.waitKey(33)
		if(response>47 and response<58):
			labelcode = response-48
			label = LABELS[labelcode]
			break
	temp_image = image.copy()
	temp_image[:,:]=(0,255,0)
	textsize = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX , 1, 2)[0]
	textX = int((image.shape[1] - textsize[0]) / 2)
	textY = int((image.shape[0] + textsize[1]) / 2)
	cv2.putText(temp_image, label, (textX, textY),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
	temp_image = cv2.addWeighted(image,0.5,temp_image,0.5,0)
	cv2.imshow("Image",temp_image)

	if args["class_type"]=='Gender' or args["class_type"]=='gender':
		imageAttr = {'Image address': imagePath, 'Gender': label, 'Age': "some age"}
		imagesLabels.append(imageAttr)
	else:
		imageAttr = {'Image address': imagePath, 'Gender': "some gender", 'Age': label}
		imagesLabels.append(imageAttr)

	cv2.waitKey(200)

print(imagesLabels)
import csv
csv_columns=["Image address","Gender","Age"]
with open(args["csv_file"],'w') as csvfile:
	writer = csv.DictWriter(csvfile,fieldnames=csv_columns)
	writer.writeheader()
	for data in imagesLabels:
		writer.writerow(data)
print("data written")
