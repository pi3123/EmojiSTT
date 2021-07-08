import sys
import os
sys.path.append(os.getcwd())

import glob
import pandas as pd
import numpy as np
import src.config as config

from src.utils import textAiHelper
from src.utils import databaseHelper

dbTurtle = databaseHelper.Turtle(filename=config.SpecAI.dbFile)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


class Mouse:
    def __init__(self, train_path, test_path, shuffle=True):
        """

        :param train_path: training text path
        :param test_path: testing text path
        :param shuffle: Boolean, shuffling the data
        """
        self.train_path = train_path
        self.test_path = test_path
        self.shuffle = shuffle
        self.aiHelper = textAiHelper.Turtle()
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

    def train(self, modelName):
        mod = mod_lr.predict(self.train_path)
        dbTurtle.saveModel(mod, modelName, filetype='pickle')
        return mod


if __name__ == "__main__":
    print("...Start Building Logistic Regression Model...")
    mod_lr = Mouse(train_path=config.TextEmoAI.trainPath,
                   test_path=config.TextEmoAI.testPath)
    mod = mod_lr.train(config.TextEmoAI.modelPath)
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
