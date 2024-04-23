# Urban Road Expansion Detection with UNet Model

---

## Introduction
This project focuses on the detection of urban road expansion using deep learning techniques, particularly a UNet model trained on satellite imagery. The model is capable of classifying images into three classes: None, Buildings, and Roads. The objective is to monitor urban growth patterns and assess infrastructure development through accurate classification of land cover types.

## Model Architecture
The UNet model architecture is employed for this task due to its effectiveness in semantic segmentation tasks. It consists of an encoder-decoder network with skip connections, allowing for the precise localization of features. The model is trained on 1024x1024 input images with three classes, achieving an accuracy of 0.89 and an Intersection over Union (IoU) of 0.88.

## Data Preparation
The training data consist of tiled images formed together to cover a larger area. The dataset contains images labeled with three classes: None, Buildings, and Roads. Each image is accompanied by its corresponding mask indicating the class labels. The model is trained for 50 epochs using this labeled dataset.

## Model Training
For detailed information on how the model was trained, please refer to the `train_model.ipynb` notebook.

## Directory Structure
The project directory follows the structure below:
- SampleData/
  - Input/       
    - sample_input1.jpg
    - sample_input2.jpg
  - Output/      
    - sample_output1.jpg
    - sample_output2.jpg
- Models/
  - model.onnx   
- main.py         
- requirements.txt  

## Usage
1. **Training:** The UNet model is trained using the provided training data. Run the training script with the appropriate parameters to start the training process.

2. **Inference:** Once the model is trained, you can use the `main.py` script to perform inference on new images. Provide the path to the input images, and the script will generate predictions in the output folder.

## Sample Images
Here are some output images:

### Sample Output
![Sample Output 1](/Runtime/output.png)

## Hackathon Participation
This project participated in the IEEE GRSS Young Professional Hackathon for Earth Observation Onboard Edge AI Model Grand Challenge organized by EEE GRSS Bangalore Section and Let's Talk Spatial in collaboration with SkyServe. We are proud to announce that our team won the second prize in this hackathon!

## Conclusion
The trained UNet model serves as a powerful tool for detecting urban road expansion, enabling effective monitoring of urbanization processes. By leveraging deep learning techniques, this project contributes to urban planning initiatives and facilitates decision-making for sustainable development.

---
