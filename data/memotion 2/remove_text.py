# https://youtu.be/3RNPJbUHZKs
"""
Remove text from images

"""

import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
import os

#General Approach.....
#Use keras OCR to detect text, define a mask around the text, and inpaint the
#masked regions to remove the text.
#To apply the mask we need to provide the coordinates of the starting and 
#the ending points of the line, and the thickness of the line

#The start point will be the mid-point between the top-left corner and 
#the bottom-left corner of the box. 
#the end point will be the mid-point between the top-right corner and the bottom-right corner.
#The following function does exactly that.
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

#Main function that detects text and inpaints. 
#Inputs are the image path and kreas_ocr pipeline
def inpaint_text(img_path, pipeline):
    inpainted_img = None
    # read the image 
    img = keras_ocr.tools.read(img_path) 
    
    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples. 
    prediction_groups = pipeline.recognize([img])
    
    #Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    if len(prediction_groups[0]) > 0:
        for box in prediction_groups[0]:
            x0, y0 = box[1][0]
            x1, y1 = box[1][1] 
            x2, y2 = box[1][2]
            x3, y3 = box[1][3] 
            
            x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
            x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
            
            #For the line thickness, we will calculate the length of the line between 
            #the top-left corner and the bottom-left corner.
            thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
            
            #Define the line and inpaint
            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
            thickness)
            inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
    else:
        print("No text detected")
        inpainted_img = img

    return(inpainted_img)

# compress the images
train_path = "./image folder/train_images/"
removed_train_path = "./image folder/removed_train_images/"

def removed_text(input_path, output_path):
    # iterate through all the files in the directory
    i = 0
    for i in range(1, 7001):
        filename = str(i) + ".jpg"
        path = os.path.join(input_path, filename)
        output = os.path.join(output_path, filename)
        # judge if the file is already compressed
        if os.path.exists(output):
            continue
        # remove the text in the images (for testing)
        pipeline = keras_ocr.pipeline.Pipeline()

        img_text_removed = inpaint_text(path, pipeline)

        cv2.imwrite(output, cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))
        print("Text on imgae {} has been removed".format(i))
        

removed_text(train_path, removed_train_path)