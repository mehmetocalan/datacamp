import numpy as np
import pandas as pd 
import streamlit as st
import os
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



st.title("Pneumonia ")
st.sidebar.header("PNEUMONIA DETECTION")


st.write('Pneumonia is an acute respiratory infection of the lung tissue, caused by bacteria. It usually affects one of the two lungs it can be caused by several types of bacteria.')
st.write('Most often, a bacterium called Streptococcus pneumoniae or pneumococcus is the cause.')
st.write('The bacterium streptococcus pneumoniae or pneumococcus is not transmitted from one person to another and therefore there is no epidemics.')
st.write('On the other hand, the bacterium Mycoplasma pneumoniae has human-to-human transmission by inhalation of respiratory particles and can therefore be responsible for small epidemics, especially in communities (family, class, office)')

image = Image.open('1.jpg')

st.image(image, caption='Normal')

st.sidebar.header("AUTHOR and CREATOR")

st.sidebar.write('Rayan ZEMOUR') 
st.sidebar.write('Mehmet OCALAN')
st.sidebar.write('Thayaan JEYARAJAH')






lab = ['PNEUMONIA', 'NORMAL']
def data(data_dir):
    data = [] 
    for lab1 in lab: 
        path = os.path.join(data_dir, lab1)
        class_num = lab.index(lab1)
        for image in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (150, 150)) # Reshape
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


#main_path = "../input/chest-xray-pneumonia/chest_xray/"
main_path = r"C:\Users\ocala\Downloads\archive\chest_xray"

st.set_option('deprecation.showPyplotGlobalUse', False)
train_path = os.path.join(main_path,"train")
test_path=os.path.join(main_path,"test")

train = data(r"C:\Users\ocala\Downloads\archive\chest_xray\train")
test = data(r"C:\Users\ocala\Downloads\archive\chest_xray\test")



count_pneumo = [0,0]
for i in train:
    if(i[1] == 0):
        count_pneumo[0] = count_pneumo[0]+1
    else:
        count_pneumo[1] = count_pneumo[1]+1
#fig = sns.countplot(count_pneumo)



plt.title('Part of pneumonia in train')
st.bar_chart(count_pneumo)

   
image = Image.open('pneumonia.jpg')

st.image(image, caption='Pneumonia')


image = Image.open('pneumonia1.jpg')

st.image(image, caption='Normal')


xTrain = []
yTrain = []
xTest = []
yTest = []
for feature, lab1 in train:
    xTrain.append(feature)
    yTrain.append(lab1)
for feature, lab1 in test:
    xTest.append(feature)
    yTest.append(lab1)

xTrain = np.array(xTrain) / 255
xTest = np.array(xTest) / 255

xTrain = xTrain.reshape(-1, 150, 150, 1)
yTrain = np.array(yTrain)

xTest = xTest.reshape(-1, 150, 150, 1)
yTest = np.array(yTest)




image = Image.open('pneumonia2.jpg')

st.image(image, caption='Normal')




fitmodel = ImageDataGenerator()
fitmodel.fit(xTrain)

model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))

model.compile(optimizer = "rmsprop" ,
              loss = 'binary_crossentropy' , 
              metrics = ['accuracy'])

model.summary()

model.fit(xTrain,yTrain, batch_size = 32 ,epochs = 4)   

