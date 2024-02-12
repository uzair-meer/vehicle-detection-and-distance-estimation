import cv2 as cv
import numpy as np


#average heights of classes
averageHeights = {
    #class name : average height in feet
    'person': 5.5,
    'car': 5,
    'bicycle': 4.5,
    'motorbike': 4,
    'bus': 12.5,
    'truck': 12.5,
    # 'dog': 2
}

allDistances = []

def calculateDistance(className, h):
    #! di = H*F/hi  -> formula of similar triangles
    H = averageHeights[className]
    # F = 2.5 inch and there are 100 pixels per inch so F = 250
    F = 250
    distance = (H*F)/h

    return distance

def createBox(img, classes, detectedClassId, x, y, w, h, allDistances):
    text = str(classes[detectedClassId])

    #bgr
    #default color
    color = (20,50,255)
    if detectedClassId == 0:
        color = (255, 0, 0)
    elif detectedClassId == 2:
        color = (0, 255, 0)
    elif detectedClassId == 7:
        color = (12, 100, 40)
    elif detectedClassId == 5:
        color = (120, 15, 99)
    elif detectedClassId == 3:
        color = (88, 52, 120)
    elif detectedClassId == 1:
        color = (50, 120, 140)
    
    distance = calculateDistance(text, h)

    #storing distances and coordinates for nearest object classification
    allDistances.append((distance, detectedClassId, x, y, w, h))

    distText =  str(round(distance, 2)) + 'ft'
    (textWidth, textHeight), _ = cv.getTextSize(distText, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    #bounding box
    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
    #distance box
    cv.rectangle(img, (x, y - 20), (x + textWidth, y), color, -1)
    #distance value
    cv.putText(img, distText, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    #class name
    cv.putText(img, text, (x, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def identifyNearestObject(img, allDistances):
    #sort all the values by their distance value
    allDistances.sort(key=lambda val:val[0])
    minDistance = allDistances[0]

    #marking the red color to the nearest object
    color = (0, 0, 255)
    cv.rectangle(img, (minDistance[2], minDistance[3]), (minDistance[2] + minDistance[4], 
    minDistance[3] + minDistance[5]), color, 2)

def yoloDetection(image):
    imgWidth = image.shape[1]
    imgHeight = image.shape[0]

    allDistances = []

    classes = []  # objects yolo can detect

    # read all object names from file
    with open('./yolov3-spp/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # create network from weights and config file
    yolo = cv.dnn.readNet('./yolov3-spp/yolov3-spp.weights', './yolov3-spp/yolov3-spp.cfg')

    # preprocessing input image
        # blob returns 4-dimensional array
        # @param1 image
        # @param2 scale factor -> multiplies our image by scale must be in 0-1
        # @param3 size -> target size in which image would become mostly used values are 224x224 or 229x229
        # @param4 mean subtract the value from each channel RGB must be 3-item tuple this is done to normalize our pixel Value take input as RGB
        # @param5 swapRB -> open cv by default read img in BGR format we can manupulate it with mean param by default its True
        # @param6 crop-> if we want to crop image
    blob = cv.dnn.blobFromImage(image, 1/255, (224, 224), (0, 0, 0), True, crop=False)
    yolo.setInput(blob)


    #YoloV3 has 3 output layers in our case these are 89, 101, 113
    #getLayerNames() 
        #get all the names of all layers of the network
    #getUnconnectedOutLayer 
        #gets the index of output layers
    #forward()
        # has our predictions, each prediction is list of floating values

    layerNames = yolo.getLayerNames()
    outputLayers = [layerNames[i - 1] for i in yolo.getUnconnectedOutLayers()]
    outputsByYolo = yolo.forward(outputLayers) 


    detectedClassIds = []
    confidences = [] # Probability
    boxes = []
    confidenceThreshold = 0.25
    

    for out in outputsByYolo:
        for detection in out:
            scores = detection[5:]
            detectedClassId = np.argmax(scores)  # indices of maximum score
            confidence = scores[detectedClassId]  # value of maximum score
            if confidence > confidenceThreshold:
                # print(detection[:4])
                # cv.circle(image, (int(detection[0]), int(detection[1])), 2, (255, 0, 0), 2)
                centerX = int(detection[0] * imgWidth)
                centerY = int(detection[1] * imgHeight)
                w = int(detection[2] * imgWidth)
                h = int(detection[3] * imgHeight)
                #cordinates of top left bounding box
                x = centerX - w / 2
                y = centerY - h / 2

                detectedClassIds.append(detectedClassId) #id of maximum scored class
                confidences.append(float(confidence)) #value | probability of maximum scored class value // typecast to float cuz its value is in <class 'numpy.float32'> 
                boxes.append([x, y, w, h]) # all detected boxes 

    

    # The process of NMS (non maximum suppression)

    nmsThreshold = 0.4
    indices = cv.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold)

    #create bounding boxes
    for i in indices:
        box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        # print(box)

        # create box around detected objects
        #only create box if only these ids are there 0 is person 2 is car
        if detectedClassIds[i] == 2 or detectedClassIds[i] == 0 or detectedClassIds[i] == 7 or detectedClassIds[i] == 5 or detectedClassIds[i] == 3 or detectedClassIds[i] == 1 :
            createBox(image, classes, detectedClassIds[i], round(x), round(y), round(w), round(h), allDistances)

    #identification of nearest object
    identifyNearestObject(image, allDistances)
    cv.imshow("object detection", image)


def runOnImage(imgPath):
    image = cv.imread(imgPath)
    yoloDetection(image)
    cv.waitKey()

def runOnVideo(videoPath):
    video = cv.VideoCapture(videoPath)

    while True:
        isTrue, frame = video.read()
        if not isTrue:
            break

        yoloDetection(frame)

        if cv.waitKey(1) & 0xFF==ord('k'):
            break

    video.release()
    cv.destroyAllWindows()

#! ==========================RUN ONE OF THE FOLLOWING FUNCTIONS==========================================

#? For Image
runOnImage('./images/img2.png')

#? For Video
# runOnVideo('./videos/video2.mp4')