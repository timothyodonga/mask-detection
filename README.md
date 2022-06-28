## Mask detection Classifier

This code repo includes code to run a standard CNN model applied to detect a person wearing a mask in an image giving the output probability of the person in the image wearing a mask.
<br> It also includes code to apply visual explainability methods like SmoothGradCAM, to visualize the region of interest that
the CNN model attributes most importance when making the prediction.

To run the code in train mode, change the mode in config.yaml to True.
<br>To run the code in test/inference mode, change the mode in config.yaml to False

To run the code in train or test mode, run the following command
<br>`python main.py`

The type of model selected, hyperparameters can be changed in the config.yaml file.

This repository also includes a jupyter notebook where one can interactively see the heat map of where the CNN model focuses on for an image.
