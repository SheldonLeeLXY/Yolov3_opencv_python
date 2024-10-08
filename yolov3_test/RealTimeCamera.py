import cv2
import numpy as np

# 加载 YOLO 模型
net = cv2.dnn.readNet("../yolov3.weights", "../yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 读取类别名称
with open("../coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 打开摄像头
cap = cv2.VideoCapture(61)  # 0 表示系统的默认摄像头

while True:
    # 从摄像头读取帧
    ret, frame = cap.read()

    if not ret:
        print("无法获取视频帧")
        break

    height, width, channels = frame.shape

    # 图像预处理
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 显示信息 on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # 物体检测
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 矩形框的坐标
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 去除多余的框（非极大值抑制）
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 绘制结果
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    # 显示实时视频和检测结果
    cv2.imshow("Camera", frame)

    # 按 'q' 退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和窗口
cap.release()
cv2.destroyAllWindows()
