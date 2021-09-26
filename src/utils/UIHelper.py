import random
import time
import numpy as np
from keras.preprocessing import image
from textblob import TextBlob

from src.AI import specAI
from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)


def countdownTimer(secs):
    """
    Countdown timer with tqdm
    :param secs: number of seconds for total recording
    :return: None
    """
    pbar = tqdm(range(secs))
    for i in pbar:
        pbar.set_description(f"Time left {secs - i} seconds")
        time.sleep(1)


def textPredict(text):
    """
    TextBlob polarity prediction driver
    :param text: str, text to passed to TextBlob
    :return: float, polarity (-1 tp 1)
    """
    return round(TextBlob(text).sentiment.polarity, 1)


def specPredict(model, imgPath, size):
    """
    specModel prediction driver
    :param size: dimensions of the img
    :param imgPath: path of the image
    :param model: Trained model from train() command
    :return: the predicted answers
    """
    test_image = []
    img = image.load_img(imgPath,
                         target_size=size,
                         color_mode="rgb" if size[2] == 3 else "grayscale")
    img = image.img_to_array(img)
    img = img / 255
    test_image.append(img)
    testArray = np.array(test_image)
    prediction = np.argmax(model.predict(testArray), axis=-1)

    return prediction


def getEmoji(emotionID):
    """
    Returns emoji
    :param emotionID:
    :return: emoji string
    """
    emotion_map = {
        0: [":pouting_face:"],  # angry 5
        1: [":nauseated_face:"],  # disgust 7
        2: [":fearful_face:"],  # fearful 6
        3: [":pensive_face:"],  # sad 4
        4: [":slightly_smiling_face:"],  # neutral 1
        5: [":smiling_face_with_halo:"],  # calm 2
        6: [":beaming_face_with_smiling_eyes:"],  # happy 3
        7: [":astonished_face:"]  # surprised 8
    }
    return random.choice(emotion_map[emotionID])


def getID(textValue, specValue):
    # Normalizing data
    """
    TEXT VALUE
    texValue is the polarity from TextBlob module which ranges from -1 to 1
    expanded to fot the mood scale (0 to 7)
    x = 3.5(x+1)

    SPEC VALUE
    SpecAI model returns numbers which range from 1 to 8 based on training dataset
    there is no correlation between the digits, hard coded dict is used
    """
    specIdMap = {
        5: 0,
        7: 1,
        6: 2,
        4: 3,
        1: 4,
        2: 5,
        3: 6,
        8: 7
    }
    textValue = 3.5 * (textValue + 1)
    specValue = specIdMap[specValue]

    # calculating ID
    return round((textValue + specValue) / 2)


# TkInter functions
def micButton_pressed():
    # TODO: Starts listening and the whole shebang
    print("Mic Button Clicked")


def settingsButton_pressed():
    # TODO: Opens a new window to edit the settings from config file.
    print("settings Button Clicked")


def trainButton_pressed():
    # TODO: Trains the spec model
    print("train Button Clicked")
    # specAI.train()


if __name__ == '__main__':
    # Testing
    print("testing countdown for 15 secs")
    countdownTimer(15)
