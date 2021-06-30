# ML
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

# General
import json
import glob
import os
import csv
import random
from tqdm import tqdm

# Image processing
from Specto import Turtle as visTurtle
import scipy
from scipy import io
from scipy.io import wavfile
import math
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class databaseHelper:
    def __init__(self, filename):
        self.dbFile = filename

    @staticmethod
    def moveFiles(testFiles, trainFiles, TargetFolder, FolderFiles):
        for i in tqdm(testFiles, desc="Moving files to Test folder"):
            try:
                os.rename(i, i.replace(TargetFolder, list(FolderFiles.keys())[0]))
            except FileExistsError:
                pass
        for i in tqdm(trainFiles, desc="Moving files to Train folder"):
            try:
                os.rename(i, i.replace(TargetFolder, list(FolderFiles.keys())[1]))
            except FileExistsError:
                pass

    def findQuality(self, filename, quality):
        with open(self.dbFile, 'r') as f:
            table = json.loads(f.read())

        if type(quality) == list:
            qualities = []
            for i in quality:
                qualities.append(table[filename][i])
            return qualities

        # else
        return table[filename][quality]

    def makeCSV(self, FolderFiles):
        for folder in FolderFiles:
            with open(FolderFiles[folder], "w") as f:
                fieldnames = ['filenames', 'emotion']
                writr = csv.DictWriter(f, fieldnames=fieldnames)
                writr.writeheader()

                files = glob.glob(folder + "\\*.png")
                for i in tqdm(files):
                    if "test" in i.lower():
                        emotion = ""
                    else:
                        emotion = self.findQuality(filename=i[-24:].replace(".png", ".wav"),
                                                   quality="Emotion_ID")
                    writr.writerow({'filenames': i[-24:],
                                    'emotion': emotion})

    def prepareData(self, FolderFiles, TargetFolder, split=0.2):
        # Moving files
        files = glob.glob(TargetFolder + "\\*.png")
        random.shuffle(files)  # shuffling all the files so all of them can be equally trained
        print("Shuffled! PNG folder")

        testFiles = files[: int((split * len(files)))]
        trainFiles = files[-int((1 - split) * len(files)):]

        # Moving Files
        self.moveFiles(testFiles, trainFiles, TargetFolder, FolderFiles)

        # Making CSV
        self.makeCSV(FolderFiles)


class visHelper:
    def __init__(self,
                 minMag=0.004416916563059203,
                 maxMag=2026134.8514368106
                 ):
        self.turtle = visTurtle(minMagnitude=minMag,
                                maxMagnitude=maxMag,
                                minPhase=-math.pi,
                                maxPhase=math.pi)

    def makeSpec(self, file, OutFolder):
        rate, audData = scipy.io.wavfile.read(file)

        # Combining 2 channels if needed
        if len(audData.shape) != 1:
            audData = audData.sum(axis=1) / 2

        img = self.turtle.genGramForWav(audData)
        fname = OutFolder + "\\" + file[-24:]
        fname = fname.replace(".wav", ".png")
        img.save(fname, "PNG")


class aiHelper:
    @staticmethod
    def OneConvLayer(input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(9, activation='softmax'))
        return model

    @staticmethod
    def TwoConvLayer(input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(9, activation='softmax'))
        return model

    @staticmethod
    def ThreeConvLayer(input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(Dropout(0.4))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(9, activation='softmax'))
        return model

    @staticmethod
    def FourConvLayer(input_shape):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(9, activation='softmax'))
        return model

    def getModel(self, modelID, input_shape):
        if modelID == 1:
            return self.OneConvLayer(input_shape)
        elif modelID == 2:
            return self.TwoConvLayer(input_shape)
        elif modelID == 3:
            return self.ThreeConvLayer(input_shape)
        elif modelID == 4:
            return self.FourConvLayer(input_shape)
