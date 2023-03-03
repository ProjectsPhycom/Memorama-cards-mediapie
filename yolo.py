# YOLO object detection
import cv2
import numpy as np
import time

# --------------- READ DNN MODEL ---------------
config = "model/yolov3.cfg"
weights = "model/yolov3.weights"
LABELS = open("model/coco.names").read().split("\n")
# print(LABELS, len(LABELS))
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
# print("colors.shape:", colors.shape)

# Load model
net = cv2.dnn.readNetFromDarknet(config, weights)
# --------------- READ THE IMAGE AND PREPROCESSING ---------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape
    frame_resized = cv2.resize(frame, (416, 416))

    # Create a blob
    blob = cv2.dnn.blobFromImage(frame_resized,
                                 1 / 255.0,
                                 (416, 416),
                                 swapRB=True,
                                 crop=False)

    # --------------- DETECTIONS AND PREDICTIONS ---------------
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    #print("ln:", ln)
    net.setInput(blob)  # Entrada de la red
    outputs = net.forward(ln)  # propagación
    #print("outputs:", outputs)

    boxes = []
    confidences = []
    classIDs = []
    for output in outputs:
        for detection in output:
            # print("detection:", detection)
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.45:
                # print("detection:", detection)
                # print("classID:", classID)
                box = detection[:4] * np.array([width, height, width, height])
                (x_center, y_center, w, h) = box.astype("int")
                x = int(x_center - (w / 2))
                y = int(y_center - (h / 2))
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    print("idx:", idx)
    if len(idx) > 0:
        for i in idx:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = colors[classIDs[i]].tolist()
            text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
            thikness = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thikness)
            cv2.rectangle(frame, (x, y), (int(x + len(text)*9.5), y + 22), color, -1)
            cv2.putText(frame, text, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), thikness)

    cv2.imshow("Google Proyect", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
