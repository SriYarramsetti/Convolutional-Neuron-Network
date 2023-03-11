import streamlit as st
import tensorflow as tf
import pandas as pd
import os
import cv2
import io
from PIL import Image, ImageOps
import numpy as np
import keras
from keras.models import Sequential, model_from_json


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
st.title(":red[Traffic] :blue[Signs] Prediction")


image_file = st.file_uploader(label="upload images",type=["png","jpg","jpeg"])

path=os.getcwd()
file_name=str(image_file.name)
dir_path = str(os.path.join(path, file_name))
img=cv2.imread(dir_path)
st.image(cv2.resize(img,(200,200)))
img = cv2.resize(img,(32,32))
img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
img=img/255
from keras.utils import np_utils
y_pred=loaded_model.predict(img)
y_pred_classes=[np.argmax(i) for i in y_pred] 
classes=[]
label_data=pd.read_csv(r"C:\Users\harsh\Downloads\traffic_signs\Indian-Traffic Sign-Dataset\traffic_sign.csv")
for i in label_data.iloc[:,1]:
    classes.append(i)
btn_click = st.button("Click here to detect the traffic sign")
if btn_click == True:
    st.write(classes[y_pred_classes[0]])