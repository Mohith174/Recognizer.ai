# Audio Classification tool 
This project implements a machine learning model to classify audio files into predefined categories. It utilizes a convolutional neural network (CNN) trained on the UrbanSound8K dataset to identify various types of urban sounds. 

## Features
- Converts audio files into spectrograms 
- Classifies audio into categories like air conditioners, car horns, children playing, etc.
- Command-line interface for easy interaction 

## Prerequisites
Before you run this project, make sure that you have a python version below 3.13 (tested on 3.12.4), along with a working install of pip. Also make sure that you have these packages: 
- Tensorflow 2.x
- Librosa (<3.13.x)
- Matplotlib
- Numpy 

## Installation 
First clone this repository to your local machine using git.
Next, navigate into the project directory 
Then, 
- if you have windows, run install.bat, which installs the necessary packages, and MAKE SURE YOU RUN CMD AS ADMIN 
- if you have a unix-based system, run chmod+x install.sh, then run install.sh

Download UrbanSound8K from their official website, and once you have it downloaded, extract the folder to the root directory of the project. 

## Usage 
First go into config.json in the root directory, and replace all the file paths with the paths to your native machine. 

Then run ```python script.py, which allows you to set up everything. Be patient while developing the model, it will take some time, as it is hardware dependent. 
(Approx: 20 mins for 20 epochs on a 12900HK/16GB Dell XPS)

After that, run ```python user.py, which is the main function, and it allows you to process an audio file, and it outputs what type of audio it is. 

If you want to play around, make the model more accurate, go into CNN.py, and change the epochs to 50, or 100 (if your machine can handle it), and have fun!

## License 
Distributed under the MIT License. See License for more informatuon 

## Acknowledgements 

This project would not have been possible without the following resources and support:

- **UrbanSound8K Dataset**: Special thanks to the creators of the [UrbanSound8K dataset](https://urbansounddataset.weebly.com/urbansound8k.html), which provided the audio samples necessary for training our sound classification model. This dataset is pivotal for urban sound research and has enabled us to develop a robust model.

- **TensorFlow**: We acknowledge [TensorFlow](https://www.tensorflow.org/) for providing the comprehensive and powerful library that enabled the construction and training of our convolutional neural networks. Their detailed documentation and active community support have been invaluable.

- **Librosa**: Our gratitude extends to the developers of [Librosa](https://librosa.org/doc/latest/index.html), a Python package for music and audio analysis. Librosa's functionality for audio processing and feature extraction was crucial for preparing our dataset for the machine learning model.

- **Matplotlib**: Thanks to [Matplotlib](https://matplotlib.org/), the primary plotting library used in this project, which made it possible to visualize spectrograms and other analytical data effectively.

- **Python Software Foundation**: We appreciate the [Python community](https://www.python.org/) for maintaining such a versatile programming language and environment, which forms the backbone of our project.

- **GitHub**: This project is hosted on GitHub, which not only facilitated version control but also made collaboration between team members seamless. 
- **Rutgers University**: Thank you to Dr. Guna for teaching CS210, the TAs for the class,  and Rutgers for having this class. Without either, we might've not done this project at all! 

These resources have collectively enabled this project to reach its current state, and we are immensely thankful for the support.


Developed by: 
Anirudh Deveram 
Mohith Kodavati 