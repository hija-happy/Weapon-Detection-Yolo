This Python script is designed to detect guns in images using a pre-trained YOLO (You Only Look Once) object detection model. 
Here's how you can use it:

INPUT: Provide the path to an image containing objects, including possibly a gun.
OUTPUT: If the script detects a gun in the image with confidence above a certain threshold, it will draw a bounding box around the detected gun.

HOW IT WORKS:

Loading YOLO Model:
The script loads a pre-trained YOLO model, which has learned to recognize various objects including guns.
Setting Confidence Threshold:
It sets a minimum confidence level for detecting objects. Only objects with confidence above this threshold will be considered valid detections.
Processing Image:
Load the specified image using the provided path.
Detecting Guns:
The script analyzes the image using the YOLO model to identify objects, focusing on detecting guns based on learned patterns.
Drawing Bounding Box:
If a gun is detected with sufficient confidence, the script draws a box around the detected gun on the image.
Steps:

Run the script.
When prompted, enter the path to the image you want to analyze.
The script will process the image and display it with a bounding box around any detected guns.

Note: Make sure to have the necessary YOLO model files (Yolov3.weights, yolov3_t.cfg) and class labels (dataset info.txt) in the specified paths for the script to work correctly.

![ScreenShot](https://github.com/hija-happy/Weapon-Detection-Yolo/assets/116438494/d4b452f7-7496-471e-a857-d7d423665c29)
