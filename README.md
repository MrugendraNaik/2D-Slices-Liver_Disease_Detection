# 2D-Slices-Liver_Disease_Detection

This is a project I was working on as practice, also i wanted to get into this domian of AIML.
The goal was To detect whether a liver CT scan slice is healthy or has a disease (tumor) using image classification.

1. I used Decathalon Liver [Task_03_Liver] Dataset on their site.
   
2. Preprocessing 
Takes 3D .nii.gz CT images and segmentation labels from Task03_Liver dataset. It then extracts the middle axial slice from each 3D scan. Then it normalizes the slice to [0, 255].
After that it checks if the label mask has tumor (label == 2). After all this saves the 2D slice as a .jpg into data folder which it creates.

3. Training Model
ResNet-18 pretrained on ImageNet was used with final layer changed to 2 classes: healthy vs disease. Also crossEntropyLoss was used. trained for multiple epochs on 2D .jpg slices.

4.Testing & Visualization
Takes input of the Test .jpg slices. The Output predicts "Healthy" or "Diseased". Displays the input image alongside the model's prediction. Also plots a line graph of all predictions with markers.



"One issue I faced because this project is something i started working on recently, is that the model is predicting healthy liver as unhealthy and unhealthy as healthy, which is the wrong output"
