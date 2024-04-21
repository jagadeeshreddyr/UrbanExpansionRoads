import onnx
import numpy as np
import onnxruntime as ort
import cv2
import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt

with open('Utils/synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]


model_path = 'Models/model.onnx'
model_path = 'Models/quantized_model (1).onnx'
model = onnx.load(model_path)
img_directory = 'SampleData/Input/'
session = ort.InferenceSession(model.SerializeToString())




def read_image(x, image_h = 1024, image_w = 1024):


    """ Image """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (image_h, image_w))
    x = x/255.0
    x = x.astype(np.float32)


    return x


def predict(path):


    sample_images = glob.glob(path)


    for image_path in sample_images:


        base_name = Path(image_path).stem

        img = read_image(image_path)

        img = np.expand_dims(img, axis=0)

        img = img.reshape(1, 1024,1024,3)

        ort_inputs = {session.get_inputs()[0].name: img.astype(np.float32)} 

        pred = session.run(None, ort_inputs)[0]

        predicted_im = pred[0,...]   
        norm = np.zeros((predicted_im.shape[0],predicted_im.shape[1]))
        norm_prediction = cv2.normalize(predicted_im,norm,0,255,cv2.NORM_MINMAX)
        norm_prediction = norm_prediction.astype('uint8')

        out_name=os.path.join('Runtime',base_name)+"-out.tif"
        cv2.imwrite(out_name,predicted_im)




if __name__ == "__main__":


    dirname = os.path.dirname(os.path.abspath(__file__))

    os.chdir(dirname)

    sample_path = 'SampleData\\Input\\*.tif'
    predict(sample_path)