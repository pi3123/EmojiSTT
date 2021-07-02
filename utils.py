# ML
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# General
import json
import glob
import os
import csv
import random
from tqdm import tqdm
import pandas as pd

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


class SpecAiHelper:
    @staticmethod
    def OneConvLayer(input_shape):
        """
        function for 1 conv layer model
        :param input_shape: shape of the image
        :return: model structure
        """
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
        """
        function for 2 conv layer model
        :param input_shape: shape of the image
        :return: model structure
        """
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
        """
        function for 3 conv layer model
        :param input_shape: shape of the image
        :return: model structure
        """
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
        """
        function for 4 conv layer model
        :param input_shape: shape of the image
        :return: model structure
        """
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
        """

        :param modelID: ID of the model to return
        :param input_shape: shape of the img
        :return: model
        """
        if modelID == 1:
            return self.OneConvLayer(input_shape)
        elif modelID == 2:
            return self.TwoConvLayer(input_shape)
        elif modelID == 3:
            return self.ThreeConvLayer(input_shape)
        elif modelID == 4:
            return self.FourConvLayer(input_shape)


class TextAiHelper:
    @staticmethod
    def clean_data(text):
        """
        :param text: dirty text
        :return: cleaned, stemmed text
        """
        stemmer = PorterStemmer()

        try:
            clean_text = text.str.lower()
            clean_text = clean_text.str.replace('\d+', '')
            clean_text = clean_text.str.strip()
            clean_text = clean_text.str.replace('[^\w\s]', ' ')
            clean_text = clean_text.str.replace('br', '')
            clean_text = clean_text.str.replace(' +', ' ')
            clean_text = clean_text.str.replace('\d+', '')

            stop = stopwords.words('english')
            stop.extend(["movie", "movies", "film", "one"])
            clean_text = clean_text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
            # Stemming
            clean_text = clean_text.apply(lambda x: " ".join(stemmer.stem(x) for x in x.split()))

            return clean_text
        except Exception as e:
            print("In Exception of clean_data: ", e)
            return None

    @staticmethod
    def tokenization(df_reviews):
        """
        :param df_reviews: reviews
        :return: tokenized text
        """
        count_vec = CountVectorizer(analyzer='word', tokenizer=lambda doc: doc, lowercase=False, max_df=0.70,
                                    min_df=100)
        print(" Tokenize the Reviews")

        df_reviews["Clean_Review"] = df_reviews["Clean_Review"].astype(str).str.strip().str.split('[\W_]+')

        words_vec = count_vec.fit(df_reviews["Clean_Review"])
        bag_of_words = words_vec.transform(df_reviews["Clean_Review"])
        tokens = count_vec.get_feature_names()
        df_words = pd.DataFrame(data=bag_of_words.toarray(), columns=tokens)
        return df_words
