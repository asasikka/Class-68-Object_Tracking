import numpy as np
import cv2
import math

confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)


video = cv2.VideoCapture("bb2.mp4")
state="play"

# Load OpenCv Tracker

tracker=cv2.legacy.TrackerCSRT_create()
detected=False

xcords=[]
ycords=[]

def drawBox(image,bbox):
    x=int(bbox[0])
    y=int(bbox[1])
    w=int(bbox[2])
    h=int(bbox[3])

    # print(x,y,w,h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
    cv2.putText(image,'Tracking', (75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

def goalTracker(image,bbox):
    x=int(bbox[0])
    y=int(bbox[1])
    w=int(bbox[2])
    h=int(bbox[3])

    c1=x+int(w/2)
    c2=y+int(h/2)

    xcords.append(c1)
    ycords.append(c2)

    cv2.circle(image,(c1,c2),2,(0,0,255),3)

    goalX=505 
    goalY=125
    cv2.circle(image,(goalX,goalY),2,(0,255,0),5)

    for i in range(len(xcords)):
         cv2.circle(image,(xcords[i],ycords[i]),2,(0,0,255),3)

    distance=math.sqrt((c1-goalX)**2+(c2-goalY))
    print("Distance: ", distance)

    if(distance<=80):
        cv2.putText(image,"Goal Reached", (20,200), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

while True:
    if state == "play":

        check, image = video.read()

        if (check):
            image = cv2.resize(image, (0, 0), fx=1, fy=1)

            dimensions = image.shape[:2]
            H, W = dimensions

            if(detected==False):


                blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416))
                yoloNetwork.setInput(blob)

                layerName = yoloNetwork.getUnconnectedOutLayersNames()
                layerOutputs = yoloNetwork.forward(layerName)

                boxes = []
                confidences = []
                classIds = []

                for output in layerOutputs:
                    for detection in output:
                        scores = detection[5:]
                        classId = np.argmax(scores)
                        confidence = scores[classId]

                        if confidence > confidenceThreshold:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY,  width, height) = box.astype('int')
                            x = int(centerX - (width/2))
                            y = int(centerY - (height/2))

                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIds.append(classId)

                indexes = cv2.dnn.NMSBoxes(
                    boxes, confidences, confidenceThreshold, NMSThreshold)

                font = cv2.FONT_HERSHEY_SIMPLEX
                for i in range(len(boxes)):
                    if i in indexes:
                        if labels[classIds[i]] == "sports ball":
                            # Write condition to detect the sports ball in the image
                            # print(i)

                            if i%2 == 0:
                                color=(0,0,255)
                            else:
                                color=(0,255,0)
                            x, y, w, h = boxes[i]

                            # Change the color of the box and label for every frame
                        


                            # Draw bounding box and label on image
                            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                            # Draw the label above the box
                            label=labels[classIds[i]]
                            cv2.putText(image, label, (x,y-8), font,0.7,color,2)
                        


                        # Write condition to detect the person in the image
                        # if labels[classIds[i]] == "person":
                    

                        #     # Change the color of the box and label for every
                            
                        #     if i%2 == 0:
                        #         color=(0,0,255)
                        #     else:
                        #         color=(0,255,0)
                        #     x, y, w, h = boxes[i]
                            

                        #     # Draw bounding box and label on image
                        #     cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                        #     # Draw the label above the box
                        #     label=labels[classIds[i]]
                        #     cv2.putText(image, label, (x,y-8), font,0.7,color,2)
                            detected=True
                            tracker.init(image,boxes[i])

            else:
                trackerInfo = tracker.update(image)
                # print(trackerInfo)
                success=trackerInfo[0]
                bbox=trackerInfo[1]

                if success:
                    drawBox(image, bbox)
                else:
                    cv2.putText(image,'Lost', (75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                goalTracker(image,bbox)

            cv2.imshow('Image', image)
            cv2.waitKey(1)

    key=cv2.waitKey(1)
    if key == 32:
        # print("stopped")
        break
    # p key
    if key == 112:
        # print("pause")
        state="pause"
    # 1 key
    if key == 108:
        # print("play")  
        state="play"
        