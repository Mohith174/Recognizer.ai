import os 
import json 
import numpy as num 
import librosa 
import librosa.display 
import matplotlib.pyplot as plot 
import tensorflow as flow 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
with open('config.json', 'r') as configurator: 
    config = json.load(configurator)
def audioSpec(filePath): 
    y, sr = librosa.load(filePath, sr = None)
    S = librosa.feature.melspectrogram(y=y,sr=sr, n_mels = 128)
    S_DB = librosa.power_to_db(S, ref=num.max)
    plot.figure(figsize = (3,3))
    librosa.display.specshow(S_DB, sr=sr, x_axis = 'time', y_axis = 'mel')
    plot.axis('off')
    plot.savefig('spectrogram.png', bbox_inches = 'tight', pad_inches = 0)
    plot.close()
    return 'spectrogram.png'

def predict(spectrogramPath, model): 
    images = image.load_img(spectrogramPath, target_size = (174,128))
    imageArray = image.img_to_array(images)
    imageArray = num.expand_dims(imageArray, axis = 0)
    predictions = model.predict(imageArray)
    predictedClassIndex = num.argmax(predictions, axis = 1)
    return predictedClassIndex

def main():
    model = load_model(config['model'])
    print("enter the path to your audio file: ")
    path = input()
    specPath = audioSpec(path)
    predictClassIndex = predict(specPath, model)
    print(f"the predicted class index of the sound is: {predictClassIndex}")

if __name__ == '__main__':
    main()