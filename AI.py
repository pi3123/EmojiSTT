# General modules
import config
# import os
import glob
import numpy as np
import pandas as pd
from utils import databaseHelper
from utils import SpecAiHelper
from utils import TextAiHelper
from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)
dbTurtle = databaseHelper(filename=config.SpecAI.dbFile)

# Modules for Image rec
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Modules for text rec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


class specAI:
    def __init__(self, input_shape, epoch, dataSize, modelID, trainCSV, testCSV, trainFolder, testFolder):
        """
        :param input_shape: shape of image
        :param epoch: runs the model is trained for
        :param dataSize: how many imgs are used to train
        :param modelID: the model structure ID
        :param trainCSV: CSV file path
        :param testCSV: CSV file path
        :param trainFolder: Training imgs folder
        :param testFolder: Testing imgs folder
        """
        self.input_shape = input_shape
        self.epoch = epoch
        self.dataSize = dataSize,
        self.modelID = modelID,

        self.trainCSV = trainCSV,
        self.testCSV = testCSV,
        self.trainFolder = trainFolder,
        self.testFolder = testFolder
        self.aiTurtle = SpecAiHelper()

    # Preprocess
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

    # Model Structure
    def train(self, data):
        """

        :param data: Data from the preprocess function
        :return: trained model
        """
        model = self.aiTurtle.getModel(modelID=self.modelID, input_shape=self.input_shape)
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        #         X_train, y_train                                X_test, , y_test)
        model.fit(data[0], data[2], epochs=self.epoch, validation_data=(data[1], data[3]))
        return model

    # Prediction
    def predict(self, model):
        """

        :param model: Trained model from train() command
        :return: the predicted answers
        """
        test_image = []
        testArray = pd.read_csv(self.testCSV)
        for i in tqdm(range(testArray.shape[0])):
            img = image.load_img(f"{self.testFolder}\\{testArray['filenames'][i]}",
                                 target_size=(self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                 color_mode="rgb" if self.input_shape[2] == 3 else "grayscale")
            img = image.img_to_array(img)
            img = img / 255
            test_image.append(img)
        testArray = np.array(test_image)
        prediction = np.argmax(model.predict(testArray), axis=-1)

        return prediction

    # Testing
    def evaluate(self, prediction):
        """

        :param prediction: answers from predict() function
        :return: accuracy score
        """
        answers = []
        testData = pd.read_csv(self.testCSV)
        for i in testData['filenames']:
            answers.append(dbTurtle.findQuality(i.replace('.png', '.wav'), 'Emotion_ID'))
        points = 0
        for i in range(len(prediction)):
            if prediction[i] == answers[i]:
                points += 1

        return int(100 * (points / len(prediction)))

    def run(self):
        """
        driver code
        :return: accuracy score and model for storing
        """
        # preprocessing data
        data = self.preprocess()

        # training model
        model = self.train(data)

        # testing
        prediction = self.predict(model)
        score = 0
        for i in range(5):
            score += self.evaluate(prediction)
        score = score / 5

        return score, model


class textAI:
    def __init__(self, train_path, test_path, shuffle=True):
        """

        :param train_path: training text path
        :param test_path: testing text path
        :param shuffle: Boolean, shuffling the data
        """
        self.train_path = train_path
        self.test_path = test_path
        self.shuffle = shuffle
        self.aiHelper = TextAiHelper()
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='liblinear', multi_class='auto')),
            ]
        )

    def get_data(self, path) -> pd.DataFrame:
        """

        :param path: folder path to make dataframe from
        :return: pandas dataframe with requested data
        """
        text = []
        rating = []
        try:
            for filename in glob.glob(f"{path}\\pos\\*.txt"):
                pos_data_train = open(filename, 'r', encoding="ISO-8859-1").read()
                text.append(pos_data_train)
                rating.append("1")

            for filename in glob.glob(f"{path}\\neg\\*.txt"):
                neg_data_train = open(filename, 'r', encoding="ISO-8859-1").read()
                text.append(neg_data_train)
                rating.append("0")

            train_dataset = list(zip(text, rating))

            if self.shuffle:
                np.random.shuffle(train_dataset)

            df_train = pd.DataFrame(data=train_dataset, columns=['Review', 'Rating'])
            return df_train

        except Exception as e:
            print("There is an error in get_train_data: ", e)
            pass

    @staticmethod
    def accuracy_score(Rating, Pred_Rating):
        """

        :param Rating: Actual answers
        :param Pred_Rating: predicted answers
        :return: accuracy score
        """
        score = accuracy_score(Rating, Pred_Rating)
        return score

    def predict(self, trainPath):
        """

        :param trainPath: training text path
        :return: trained model
        """
        train_df = self.get_data(trainPath)
        train_df["Clean_Review"] = self.aiHelper.clean_data(train_df["Review"])
        learner = self.pipeline.fit(train_df["Clean_Review"], train_df["Rating"])

        # Predict class labels using the learner and output DataFrame
        test_df = self.get_data(self.test_path)

        test_df['Pred_Rating'] = learner.predict(test_df['Review'])
        score = self.accuracy_score(test_df['Rating'], test_df['Pred_Rating'])
        print("Accuracy of the Logistic Regression Model is:  ", score)
        return learner


if __name__ == "__main__":
    print("...Start Building Logistic Regression Model...")
    mod_lr = textAI(train_path=config.TextEmoAI.trainPath,
                    test_path=config.TextEmoAI.testPath)
    mod = mod_lr.predict(config.TextEmoAI.trainPath)
    v = 0
    while True:

        input_sen = str(input("Enter a Text: "))
        input_sen = [input_sen]

        val = mod.predict(input_sen)

        if val == '1':
            print("The sentiment of the above statement is: Positive")
        else:
            print("The sentiment of the above statement is: Negative")

        key_in = input("Do You want to try another sentence? (yes to try again or no to stop): ").lower()

        if key_in == "no":
            break
