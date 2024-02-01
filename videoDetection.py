import numpy as np
import cv2

confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)


video = cv2.VideoCapture("bb2.mp4")
state="play"


while True:
    if state == "play":

        check, image = video.read()

        if (check):
            image = cv2.resize(image, (0, 0), fx=1, fy=1)

            dimensions = image.shape[:2]
            H, W = dimensions

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
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                        # Draw the label above the box
                        label=labels[classIds[i]]
                        cv2.putText(image, label, (x,y-8), font,0.7,color,2)
                    


                    # Write condition to detect the person in the image
                    if labels[classIds[i]] == "person":
                

                        # Change the color of the box and label for every
                        
                        if i%2 == 0:
                            color=(0,0,255)
                        else:
                            color=(0,255,0)
                        x, y, w, h = boxes[i]
                        

                        # Draw bounding box and label on image
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                        # Draw the label above the box
                        label=labels[classIds[i]]
                        cv2.putText(image, label, (x,y-8), font,0.7,color,2)

                    

            cv2.imshow('Image', image)
            cv2.waitKey(1)

    key=cv2.waitKey(1)
    if key == 32:
        print("stopped")
        break
    # p key
    if key == 112:
        print("pause")
        state="pause"
    # 1 key
    if key == 108:
        print("play")  
        state="play"
        