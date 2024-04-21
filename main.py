import onnx
import numpy as np
import onnxruntime as ort
import cv2
import os
import glob
from pathlib import Path
import tifffile
import rasterio

from Utils.split_merge import split_image, merge_image


with open('Utils/synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]


model_path = 'Models/model.onnx'
model = onnx.load(model_path)
img_directory = 'SampleData/Input/'
session = ort.InferenceSession(model.SerializeToString())



def read_image(x, image_h = 1024, image_w = 1024):

    """ Image """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)

    return x


def predict(path):


    sample_images = glob.glob(path)


    for image_path in sample_images:


        base_name = Path(image_path).stem

        img_1 = read_image(image_path)

        img_list = split_image(img_1, (1024,1024))

        img_out = []

        for img in img_list:

            img = np.expand_dims(img, axis=0)

            img = img.reshape(1, 1024,1024,3)

            ort_inputs = {session.get_inputs()[0].name: img.astype(np.float32)} 

            pred = session.run(None, ort_inputs)[0]

            predicted_im = pred[0,...] 

            img_out.append(predicted_im)  

        final_image = merge_image(img_out, img_1.shape)

        out_name=os.path.join('Runtime',base_name)+"-out.tif"
        tifffile.imwrite(out_name, final_image)



if __name__ == "__main__":


    dirname = os.path.dirname(os.path.abspath(__file__))

    os.chdir(dirname)

    sample_path = 'SampleData\\Input\\*.tif'
    predict(sample_path)