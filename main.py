import cv2
import numpy as np

colors = []
for i in range(81):
    random_c = np.random.randint(256, size=3)
    colors.append((int(random_c[0]), int(random_c[1]), int(random_c[2])))


net = cv2.dnn.readNet("yolov2.weights", "yolo2.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

classes = []
with open("classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


def image(str):

    img = cv2.imread(str)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    cv2.putText(img, "aitu 2022. Written by Ashim Yerbol & Usenkhanov Daniyar. CS-2121", (20, 20), font, 1, (255,0,0), 1)
    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video(vid):
    cap = cv2.VideoCapture(vid)

    while True:
        succ, img = cap.read()

        (class_ids, scores, bboxes) = model.detect(img, confThreshold=0.3, nmsThreshold=.4)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]
            color = colors[class_id]

            if class_name in classes:
                cv2.putText(img, class_name + " " + str(score), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        cv2.putText(img, "aitu 2022. Written by Ashim Yerbol & Usenkhanov Daniyar. CS-2121", (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0), 1)

        cv2.imshow("Object Recognition", img)
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()    

def live(vid):
    cap = cv2.VideoCapture(vid)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        succ, img = cap.read()

        (class_ids, scores, bboxes) = model.detect(img, confThreshold=0.3, nmsThreshold=.4)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]
            color = colors[class_id]

            if class_name in classes:
                cv2.putText(img, class_name + " " + str(score), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        cv2.putText(img, "aitu 2022. Written by Ashim Yerbol & Usenkhanov Daniyar. CS-2121", (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,0,0), 1)

        cv2.imshow("Object Recognition", img)
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

print("Hello! This is the code for recognition and tracking objects in images, videos or even live webcam.\nWe have 3 different modes. There are:\n1.Image\n2.Video\n3.Live\n0.Exit\nPlease, select mode: ", end="")
n = int(input())

while n != 0:
    if n == 1:
        image(input("full directory to image: "))
    elif n == 2:
        video(input("full directory to video: "))
    elif n == 3:
        live(0)
    print("1.Image\n2.Video\n3.Live\n0.Exit\nPlease, select mode: ", end="")
    n = int(input())
    
print("Salamaleikum")