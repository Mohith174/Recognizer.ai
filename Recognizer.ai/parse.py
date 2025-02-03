import librosa 
import librosa.display
import matplotlib.pyplot as plot 
import numpy as num 
import os 
import pandas as panda 
from datetime import datetime
from dbman import dbman

class Parse: 
    def __init__(self, adir, odir,dbman,csv_path):
        self.adir = adir
        self.odir = odir
        self.dbman = dbman
        self.csv = csv_path
        if not os.path.exists(odir):
            os.makedirs(odir)
    def load(self):
        self.data = panda.read_csv(self.csv)
    def atos(self, fpath,odir,classlabel):
        filename = f"{os.path.splitext(os.path.basename(fpath))[0]}_{classlabel.replace(' ', '_')}.png"
        outpath = os.path.join(odir,filename)
        #os.makedirs(os.path.dirname(outpath), exist_ok=True)
        
        if not os.path.exists(outpath):
            y,sr = librosa.load(fpath, sr=None)
            S = librosa.feature.melspectrogram(y=y,sr=sr, n_mels = 128)
            sdb = librosa.power_to_db(S,ref = num.max) 
            plot.figure(figsize=(10,4))
            librosa.display.specshow(sdb,sr=sr, x_axis='time',y_axis='mel')
            plot.colorbar(format='%+2.0f dB')
            plot.title('Mel-frequency spectrogram')
            plot.tight_layout()
            plot.savefig(outpath)
            plot.close()

        self.dbman.addSpectrogramEntry(fpath, outpath,classlabel, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        #return sdb
    
    def processDirectory(self, cdir, odir):
        #if not os.path.exists(odir):
           # os.makedirs(odir)
        for entry in os.listdir(cdir):
            fpath = os.path.join(cdir, entry)
            if os.path.isdir(fpath):
                sdiro = os.path.join(odir, entry)
                self.processDirectory(fpath, odir)
            elif entry.endswith(".wav"):
                classlabel = self.data[self.data['slice_file_name'] == entry]['class'].values[0]
                self.atos(fpath, odir, classlabel)


    def processAudio(self):
        self.load()
        self.processDirectory(self.adir, self.odir)

