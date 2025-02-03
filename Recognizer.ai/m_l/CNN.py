import os 
import numpy as num 
import pandas as panda 
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as flow 
from tensorflow.keras import models, callbacks, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import load_img, img_to_array
class Recognizer_ai:
    def __init__(self, dbpath, img_width = 128, img_height = 174):
        self.dbpath = dbpath 
        self.imgwidth = img_width 
        self.imgheight = img_height 
        self.classes = ['air_conditioner', 'car_horn', 'children_playing', 
            'dog_bark', 'drilling', 'engine_idling', 
            'gun_shot', 'jackhammer', 'siren', 'street_music']
        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(self.classes)
        
    def loadProcessSpectrogram(self, fpath):
        #process the spectrograms into usable data 
        img = load_img(fpath, target_size = (self.imgheight,self.imgwidth), color_mode = 'rgb')
        imagearray = img_to_array(img)/255.0 
        return imagearray
    
    def loadDB(self):
        #loads the spectrograms from the databse into numpy arrays, where X is the array of images, Y is the array of encoded labels
        connection = sqlite3.connect(self.dbpath)
        query = "SELECT spath, pdate FROM spectrograms"
        frame = panda.read_sql_query(query, connection)
        connection.close()
        x = []
        y = []
        error = [] 
        for _, row in frame.iterrows():
            try: 
                imagearray = self.loadProcessSpectrogram(row['spath'])
                x.append(imagearray)
                y.append(self.labelencoder.transform([row['pdate']])[0])
            except Exception as e: 
                error.append((row['spath'],str(e)))

        if error:
            print("warning, some spectrograms cannot be loaded...")
            for path in error: 
                print(f" - {path}: {error}")
        return num.array(x), num.array(y)
    
    def modelC(self):
        #creates the model 
        base = flow.keras.applications.EfficientNetB0(include_top = False, weights = 'imagenet', input_shape = (self.imgheight,self.imgwidth,3))
        base.trainable = False 
        model = models.Sequential([base, layers.GlobalAveragePooling2D(),layers.Dense(512, activation = 'relu'), layers.Dropout(0.5), layers.Dense(len(self.classes), activation = 'softmax')])
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        return model

    def train(self, test_size = 0.2, random_state = 42): 
        #trains the model 
        x,y = self.loadDB()
        if len(x) == 0: 
            raise ValueError("no spectrograms have been loaded from the database; check your database and file paths...")
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = test_size, random_state = random_state, stratify = y)
        model = self.modelC()
        callbacksList = [callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6), callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')]
        history = model.fit(xtrain, ytrain, validation_data = (xtest, ytest), epochs = 20, batch_size = 32, callbacks = callbacksList)
        return model, history 
    

        
        
