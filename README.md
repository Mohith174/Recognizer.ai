# Audio Classification Tool

This project implements a machine learning model to classify audio files into predefined categories using a convolutional neural network (CNN). It is trained on the UrbanSound8K dataset to identify various types of urban sounds.

## DISCLAIMER

Due to the limitations of our hardware, the accuracy of the model isn't high enough for reliable predictions. This is a final project, but we plan to refine it over time to enhance its accuracy and applicability in various contexts. Feel free to modify the parameters, alter the model, and explore the intricacies of machine learning through this project!

## Features

- Converts audio files into spectrograms.
- Classifies sounds into categories like air conditioners, car horns, children playing, etc.
- Command-line interface for easy interaction.

## Prerequisites

Ensure you have the following before starting:
- Python version below 3.13 (tested on 3.12.4)
- Pip
- Tensorflow 2.x
- Librosa (for Python versions 3.8 < x < 3.13.x)
- Matplotlib
- Numpy

## Installation

Clone this repository to your local machine using git. Navigate to the project directory and:
- On Windows, run `install.bat` (make sure to run CMD as admin).
- On Unix-based systems, run `chmod +x install.sh` then `./install.sh`.

Download the UrbanSound8K dataset from their official website. Extract the folder to the root directory of the project.

## Usage

1. Modify `config.json` in the root directory with the paths to your local machine.
2. Run `python script.py` to set up the model. Note: Model development is hardware-dependent and may take approx. 20 mins for 20 epochs on a 12900HK/16GB Dell XPS.
3. Execute `python user.py` to process an audio file and classify its type.

To enhance model accuracy, adjust the epoch count in `CNN.py` to 50 or 100, depending on your machine's capabilities.

## License

Distributed under the MIT License. See LICENSE file for more information.

## Acknowledgements

This project would not have been possible without:

- **UrbanSound8K Dataset:** Essential for urban sound research and training our model.
- **TensorFlow:** For their comprehensive machine learning library.
- **Librosa:** For audio processing and feature extraction capabilities.
- **Matplotlib:** For enabling effective visualization of spectrograms.
- **Python Software Foundation:** For maintaining the Python programming environment.
- **GitHub:** For hosting and facilitating collaboration on this project.
- **Rutgers University:** Special thanks to Dr. Guna, the TAs of CS210, and Rutgers for their support in this endeavor.

Developed by: Mohith Kodavati and Anirudh Deveram
