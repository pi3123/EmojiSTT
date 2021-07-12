import sys
import os
sys.path.append(os.getcwd())

import random
import numpy as np
import pandas as pd
import glob

from functools import partial
from tqdm import tqdm

from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from src import config
from src.utils import specAiHelper
from src.utils import databaseHelper
from src.utils import visHelper

tqdm = partial(tqdm, position=0, leave=True)
dbTurtle = databaseHelper.Turtle(filename=config.SpecAI.dbFile)


class Mouse:
    def __init__(self, input_shape, epoch, dataSize, modelID, trainCSV, testCSV, trainFolder, testFolder):
        """
        :param input_shape: shape of image
        :param epoch: runs the model is trained for
        :param dataSize: how many imgs are used to train
        :param modelID: the model structure ID;
        :param trainCSV: CSV file path
        :param testCSV: CSV file path
        :param trainFolder: Training imgs folder
        :param testFolder: Testing imgs folder
        """
        self.input_shape = input_shape
        self.epoch = epoch
        self.dataSize = dataSize
        self.modelID = modelID

        self.trainCSV = trainCSV
        self.testCSV = testCSV
        self.trainFolder = trainFolder
        self.testFolder = testFolder
        self.aiTurtle = specAiHelper.Turtle()

    def preprocess(self):
        """
        to split and clean up the data.
        :return: tuple of data for training the model
        """
        train_image = []
        train = pd.read_csv(self.trainCSV)
        for i in tqdm(range(train.shape[0])):
            img = image.load_img(f"{self.trainFolder}\\{train['filenames'][i]}",
                                 target_size=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                 color_mode="rgb" if self.input_shape[2] == 3 else "grayscale")
            img = image.img_to_array(img)
            img = img / 255
            train_image.append(img)

        X = np.array(train_image)
        y = train['emotion'].values
        y = to_categorical(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

        return X_train, X_test, y_train, y_test

    def train(self, data):
        """
        training model
        :param data: Data from the preprocess function
        :return: trained model
        """
        model = self.aiTurtle.getModel(modelID=self.modelID, input_shape=self.input_shape)
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        #         X_train, y_train                                X_test, , y_test)
        model.fit(data[0], data[2], epochs=self.epoch, validation_data=(data[1], data[3]))
        return model

    def evaluate(self):
        """
        :return: accuracy score
        """

        # getting predictions
        test_image = []
        testArray = pd.read_csv(self.testCSV)

        for i in tqdm(range(testArray.shape[0])):
            img = image.load_img(f"{self.testFolder}\\{testArray['filenames'][i]}",
                                 target_size=self.input_shape,
                                 color_mode="rgb" if self.input_shape[2] == 3 else "grayscale")
            img = image.img_to_array(img)
            img = img / 255
            test_image.append(img)

        testArray = np.array(test_image)
        prediction = np.argmax(model.predict(testArray), axis=-1)

        # checking answers
        answers = []
        testData = pd.read_csv(self.testCSV)
        for i in testData['filenames']:
            answers.append(dbTurtle.findQuality(i.replace('.png', '.wav'), 'Emotion_ID'))
        points = 0
        for i in range(len(prediction)):
            if prediction[i] == answers[i]:
                points += 1

        return int(100 * (points / len(prediction)))


if __name__ == "__main__":

    visTurtle = visHelper.Turtle()

    # reset Log file
    print("Resting log file")
    with open(config.SpecAI.logFile, "w") as f:
        f.write("")
        f.close()

    print("Starting training process")
    for size in config.SpecAI.DATA_SIZE:
        # Generating data for training
        files = glob.glob(config.SpecAI.audioFolder + "\\*\\*\\*.wav")
        random.shuffle(files)
        if size is not None:
            if size > len(files):
                print("Size too big, loading all available files")
            else:
                files = files[-size:]
                print(f"Loading last {size} files and shuffling")
        else:
            print("Loading all files and shuffling")

        pbar = tqdm(files)
        for i in pbar:
            pbar.set_description(f"Processing {i[-24:]}")
            #                                        slicing filename
            fname = config.SpecAI.specFolder + "\\" + i[-24:][:20] + ".png"
            visTurtle.makeSpec(file=i, fname=fname)

        dbTurtle.prepareData(FolderFiles={config.SpecAI.testFolder: config.SpecAI.testCSV,
                                          config.SpecAI.trainFolder: config.SpecAI.trainCSV},
                             TargetFolder=config.SpecAI.specFolder)

        for epoch in config.SpecAI.EPOCHS:
            for modelID in config.SpecAI.MODEL_STRUCTURE_ID:
                for input_shape in config.SpecAI.INPUT_SHAPES:

                    """ Driver Code"""

                    specAI = Mouse(input_shape=input_shape,
                                   epoch=epoch,
                                   dataSize=size,
                                   modelID=modelID,

                                   trainCSV=config.SpecAI.trainCSV,
                                   testCSV=config.SpecAI.testCSV,
                                   trainFolder=config.SpecAI.trainFolder,
                                   testFolder=config.SpecAI.testFolder
                                   )

                    # preprocessing data
                    data = specAI.preprocess()

                    # training model
                    model = specAI.train(data)

                    # evaluating
                    score = 0
                    for i in range(5):
                        score += specAI.evaluate()
                    score = score / 5

                    """ Logging """
                    with open(config.SpecAI.logFile, "a") as f:
                        statement = f"####################################################### \n" \
                                    f"Model : {modelID}      Shape : {input_shape}       Epochs : {epoch}        Data Volume : {size}\n" \
                                    f"Average Accuracy : {score}%" \
                                    f"\n"
                        f.write(statement)
                        f.close()

                    """ Saving model """
                    #                [size]_    [epochs]_   [modelID]_    [input_shape].h5
                    modelName = f"{str(size)}_{str(epoch)}_{str(modelID)}_{str(input_shape).replace(' ', '')}.h5"
                    model.save(f"{config.SpecAI.SpecModelsFolder}\\{modelName}")
