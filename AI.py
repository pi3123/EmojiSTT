from keras.preprocessing import image
from utils import databaseHelper
from utils import aiHelper
import numpy as np
import pandas as pd
import config
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)

dbTurtle = databaseHelper(filename=config.dbFile)


# Preprocess
def preprocess(input_shape):
    train_image = []
    train = pd.read_csv(config.trainCSV)
    for i in tqdm(range(train.shape[0])):
        img = image.load_img(f"{config.trainFolder}\\{train['filenames'][i]}",
                             target_size=(input_shape[0], input_shape[1], input_shape[2]),
                             color_mode="rgb" if input_shape[2] == 3 else "grayscale")
        img = image.img_to_array(img)
        img = img / 255
        train_image.append(img)

    X = np.array(train_image)
    y = train['emotion'].values
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    return X_train, X_test, y_train, y_test


# Model Structure
def fitModel(model_structure, input_shape, epochs, data):
    aiTurtle = aiHelper()
    model = aiTurtle.getModel(modelID=model_structure, input_shape=input_shape)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    #         X_train, y_train                                X_test, , y_test)
    model.fit(data[0], data[2], epochs=epochs, validation_data=(data[1], data[3]))
    return model


# Prediction
def predict(model, input_shape):
    test_image = []
    test = pd.read_csv(config.testCSV)
    for i in tqdm(range(test.shape[0])):
        img = image.load_img(f"{config.testFolder}\\{test['filenames'][i]}",
                             target_size=(input_shape[0], input_shape[1], input_shape[2]),
                             color_mode="rgb" if input_shape[2] == 3 else "grayscale")
        img = image.img_to_array(img)
        img = img / 255
        test_image.append(img)
    test = np.array(test_image)
    prediction = np.argmax(model.predict(test), axis=-1)

    return prediction


# Testing
def test(prediction):
    answers = []
    test = pd.read_csv(config.testCSV)
    for i in test['filenames']:
        answers.append(dbTurtle.findQuality(i.replace('.png', '.wav'), 'Emotion_ID'))

    points = 0
    for i in range(len(prediction)):
        if prediction[i] == answers[i]:
            points += 1

    return int(100 * (points / len(prediction)))


def train(inputShapes, Epochs, IDs, DataSize):
    pbar = tqdm(inputShapes)
    for EPOCH in Epochs:
        for iD in IDs:
            for input_shape in pbar:
                pbar.set_description(
                    desc=f"Model : {iD}\nShape : {input_shape}\nEpochs : {EPOCH}\nData Volume : {DataSize}\n")
                data = preprocess(input_shape)
                model = fitModel(iD, input_shape, EPOCH, data)
                prediction = predict(model, input_shape)
                score = 0
                for i in range(5):
                    score += test(prediction)
                score = score / 5

                with open(config.logFile, "a") as f:
                    statement = f"####################################################### \n" \
                                f"Model : {iD}      Shape : {input_shape}       Epochs : {EPOCH}        Data Volume : {DataSize}\n" \
                                f"Average Accuracy : {score}%" \
                                f"\n"
                    f.write(statement)
                    f.close()
