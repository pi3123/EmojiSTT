import sys
import os

sys.path.append(os.getcwd())

from src import config
from src.utils import databaseHelper
from src.utils import visHelper
from src.utils import audioHelper
from src.utils import specAiHelper
from src.utils import UIHelper

import emoji
import keyboard
from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)

# Turtles
dbTurtle = databaseHelper.Turtle(filename=config.SpecAI.dbFile)
visTurtle = visHelper.Turtle()
audioTurtle = audioHelper.Turtle(rate=config.audio.sampleRate)
specTurtle = specAiHelper.Turtle()

if __name__ == "__main__":
    print(f"Press Enter to start recording for {config.audio.duration} seconds \nESC to exit")
    while True:
        if keyboard.is_pressed('enter'):
            # Record audio
            print("Recording started")
            audioTurtle.record(fname=config.audio.recordingPath,
                               duration=config.audio.duration)

            print("Recording saved")

            # convert WAV to spec
            print("Converting recording to spectogram")
            visTurtle.makeSpec(file=config.audio.recordingPath,
                               fname=config.audio.specPath)
            visTurtle.resizeImg(fname=config.audio.specPath,
                                newSize=(config.UI.imgSize[0], config.UI.imgSize[1]))

            print("Spec made!")

            # convert WAV to text (google TTS)
            print("Converting recording to text ")
            try:
                text = audioTurtle.wavToText(config.audio.recordingPath)
                print(f"you said : {text}")

            except Exception:
                print("try again")
                pass

            # load SpecModel
            specAiModel = dbTurtle.loadModel(config.SpecAI.modelPath, filetype="hdf5")

            # load TextModel
            # TODO: start using Textblob and combine the Text Emotion prediction and voice
            textAiModel = dbTurtle.loadModel(config.TextEmoAI.modelPath, filetype="pickle")

            # prediction
            textVal = textAiModel.predict([text])

            specVal = specTurtle.predict(model=specAiModel,
                                         imgPath=config.audio.specPath,
                                         size=config.UI.imgSize)

            # output
            print(emoji.emojize(f"{text} {UIHelper.getEmoji(specVal[0])}"))

        elif keyboard.is_pressed('esc'):
            print("Exiting")
            break
