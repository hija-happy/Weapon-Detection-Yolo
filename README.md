# Weapon Detection using YOLOv3

This Python script is designed to detect weapons in images using a pre-trained YOLO (You Only Look Once) object detection model. 

## How to Use

**Input:**  
Provide the path to an image containing objects, including possibly a weapon.

**Output:**  
If the script detects a weapon in the image with confidence above a certain threshold, it will draw a bounding box around the detected weapon.

## How It Works

* **Loading YOLO Model:**  
  The script loads a pre-trained YOLO model, which has learned to recognize various objects including weapons.

* **Setting Confidence Threshold:**  
  It sets a minimum confidence level for detecting objects. Only objects with confidence above this threshold will be considered valid detections.

* **Processing Image:**  
  Load the specified image using the provided path.

* **Detecting Weapons:**  
  The script analyzes the image using the YOLO model to identify objects, focusing on detecting weapons based on learned patterns.

* **Drawing Bounding Box:**  
  If a weapon is detected with sufficient confidence, the script draws a box around the detected weapon on the image.

## Steps

1. Run the script.
2. When prompted, enter the path to the image you want to analyze.
3. The script will process the image and display it with a bounding box around any detected weapons.

*Note:*  
Make sure to have the necessary YOLO model files (`yolov3.weights`, `yolov3_t.cfg`) and class labels (`dataset info.txt`) in the specified paths for the script to work correctly.

![ScreenShot](https://github.com/hija-happy/Weapon-Detection-Yolo/assets/116438494/d4b452f7-7496-471e-a857-d7d423665c29)
