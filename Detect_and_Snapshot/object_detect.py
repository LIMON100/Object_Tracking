# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:03:24 2021

@author: limon
"""

import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import time

from imutils.video import FileVideoStream
from imutils.video import FPS

def main():

    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

    classes = []
    with open("./data/labels/coco.names", "r") as f:
        classes = f.read().splitlines()

    #cap = cv2.VideoCapture('traffic1.mkv')

    cap = FileVideoStream("traffic1.mkv").start()
    time.sleep(0.1)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (500, 3))

    counter = []
    current_count = 0
    result = []

    directory1 = "/home/ecl/Downloads/Limon/Object_Tracking/imgzmq/dataset/"
    c = 0

    fps = FPS().start()

    while cap.more():

        img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        current_count = 0
        if len(indexes)>0:
            
            for i in indexes.flatten():

                c +=1
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                
                #cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
                #cv2.putText(img, label, (x, y + 20), font, 2, (255, 255, 255), 2)
                #cv2.imwrite(directory1 + f"image{c}.jpg", img[y:y+h, x:x+w])
                #cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
                #cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)

                #cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
                #cv2.line(img, (220, 456), (1000, 456), (0, 0, 255), 2)
                #cv2.line(img, (220, 440), (1000, 440), (0, 255, 0), 1)
                #cv2.line(img, (220, 460), (1000, 460), (0, 255, 0), 1)

                cv2.line(img, (92, 456), (1135, 456), (0, 0, 255), 2)
                cv2.line(img, (92, 460), (1135, 460), (0, 255, 0), 1)

                count = 0
                #center_y1 = int(((y) + (h))/2)
                x_mid = int((x + (x + w)) / 2)
                y_mid = int((y + (y + h)) / 2)
                #if center_y1 <= int(3*height/6+height/20) and center_y1 >= int(3*height/6-height/20):
                if y_mid >= 456 and y_mid <= 460:
                    print("Inside line...........")            
                    #print("Inside count..........")
                    current_count += 1

                    directory = r'/home/ecl/Downloads/Limon/Object_Tracking/imgzmq/dataset'
                    for filename in os.listdir(directory):
                        if filename.endswith(".jpg") or filename.endswith(".png"):
                            a1 = os.path.join(directory, filename)
                            b = int(re.search(r'\d+', a1).group())
                            result.append(b)
                        else:
                            continue
                            
                        
                        b1 = max(result) + 1
                        #count = 0

                        while(True):
                            print("inside save picture.............")
                            count += 1
                            #image[box.ymin:box.ymax, box.xmin:box.xmax]
                            #snew_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                            new_img = img[y:y+h , x:x+w]
                            #new_img = new_img[int(bbox[1]):(int(bbox[1])+int(bbox[3])), int(bbox[0]):(int(bbox[0])+int(bbox[2]))]
                            new_img = cv2.resize(new_img, (240, 240), interpolation = cv2.INTER_NEAREST)
                            #new_img = img[track.track_id]

                            cv2.imwrite(directory1 + f"image{b1}.jpg", new_img)
                                        
                            if count >= 1:
                                break
                

        #cv2.putText(img, "Total Vehicle Count: " + str(current_count), (0,130), 0, 1, (0,0,255), 2)
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)

        if cap.Q.qsize() < 2:
            time.sleep(0.001)

        fps.update()
        if key==27:
            break

    #cap.release()
    cv2.destroyAllWindows()
    cap.stop()


if __name__ == '__main__':

    try:
        main()
    except AttributeError:
        pass
