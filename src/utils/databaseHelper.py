import json
import glob
import os
import csv
import random
import pickle
import numpy as np

from tqdm import tqdm
from tensorflow import keras
from PIL import Image


class Turtle:
    def __init__(self, filename):
        """
        :param filename: Storage filename
        """
        self.dbFile = filename

    @staticmethod
    def moveFiles(testFiles, trainFiles, TargetFolder, FolderFiles):
        """

        :param testFiles: Testing files path
        :param trainFiles: Training files path
        :param TargetFolder: Folder to move the files to
        :param FolderFiles: Dict with {Folder:csvfile}
        :return:
        """
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
        """
        To look up the json file for a quality
        :param filename: name of the file to find the quality
        :param quality: quality to find
        :return: quality
        """
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
        """
        To make and save CSV file for training
        :param FolderFiles: Dict with {Folder:csvfile}
        """
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
        """
        Driver code
        :param FolderFiles: Dict with {Folder:csvfile}
        :param TargetFolder: Storage folder
        :param split: percentage of evaluating files
        """

        # Moving files
        files = glob.glob(TargetFolder + "\\*.png")
        random.shuffle(files)
        print("Shuffled! PNG folder")

        testFiles = files[: int((split * len(files)))]
        trainFiles = files[-int((1 - split) * len(files)):]

        # Moving Files
        self.moveFiles(testFiles, trainFiles, TargetFolder, FolderFiles)

        # Making CSV
        self.makeCSV(FolderFiles)

    @staticmethod
    def getSize(imgpath):
        img = Image.open(imgpath)
        img = np.array(img)
        return img.shape

    @staticmethod
    def saveModel(model, modelpath, filetype):
        if filetype == "pickle":
            with open(modelpath, 'wb') as f:
                pickle.dump(model, f)
                f.close()
        elif filetype == "hdf5":
            model.save(modelpath)

    @staticmethod
    def loadModel(modelpath, filetype):
        if filetype == "pickle":
            with open(modelpath, 'rb') as f:
                model = pickle.load(f)
                f.close()
            return model
        elif filetype == "hdf5":
            model = keras.models.load_model(modelpath)
            return model
