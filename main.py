import cv2
import numpy as np
import os

# Load YOLO
weights_path = r"C:\Users\hijah\PycharmProjects\pythonProject\DataSet Info\Weight FIle For Yolov3.weights"
config_path = r"C:\Users\hijah\PycharmProjects\pythonProject\DataSet Info\yolov3_t.cfg"
net = cv2.dnn.readNet(weights_path, config_path)

# Load class labels
classes_path = r"C:\Users\hijah\PycharmProjects\pythonProject\DataSet Info\dataset info.txt"
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set the minimum confidence threshold
confidence_threshold = 0.5

# Get image file path from user
image_path = input("Enter the path to the image file: ")
if not os.path.isfile(image_path):
    print("Failed to load the image")
    exit()

# Load image
image = cv2.imread(image_path)

# Get image dimensions
height, width, _ = image.shape

# Create blob from image             #
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set input for the network
net.setInput(blob)

# Run forward pass through the network
output_layers_names = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers_names)

# Loop over each output layer
for output in layer_outputs:
    # Loop over each detection
    for detection in output:
        # Get class scores and class ID
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter detections by confidence threshold and class ID
        if confidence > confidence_threshold and class_id == 0:

            # Calculate bounding box coordinates
            box = detection[0:4] * np.array([width, height, width, height])
            x, y, w, h = map(int, box)

            # Draw bounding box and label                              #(B-G-R)  #thickness
            cv2.rectangle(image, (x-w//2, y-h//2), (x+w//2, y+h//2), (255, 0, 0), 2)
            cv2.putText(image, "Gun", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Apply non-maximum suppression to remove overlapping bounding boxes
indices = cv2.dnn.NMSBoxes([(x-w//2, y-h//2, w, h)], [confidence], confidence_threshold, 0.4)

# Loop over the selected indices after NMS
for i in indices:
    i = i[0]
    x, y, w, h = x, y, w, h = x-w//2, y-h//2, w, h
    cv2.rectangle(image, (x, y), (x+w, y+h), (200, 600, 0), 2)
    cv2.putText(image, "Gun", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


#display the image
cv2.imshow("Gun Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()