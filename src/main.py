import sys
import os

from src import config
from src.utils import databaseHelper
from src.utils import visHelper
from src.utils import audioHelper
from src.utils import specAiHelper
from src.utils import UIHelper

import emoji
import keyboard
import speech_recognition as sr

# Setup
sys.path.append(os.getcwd())
dbTurtle = databaseHelper.Turtle(filename=config.SpecAI.dbFile)
visTurtle = visHelper.Turtle()
audioTurtle = audioHelper.Turtle(rate=config.audio.sampleRate)
specTurtle = specAiHelper.Turtle()

if __name__ == "__main__":

    print(f"Press Enter to start recordi ng for {config.audio.duration} seconds \n"
          f"ESC to exit")
    while True:
        if keyboard.is_pressed('enter'):
            # Record audio
            print("Recording started")
            audioTurtle.record(fname=config.audio.recordingPath,
                               duration=config.audio.duration)
            print("Recording saved!")

            # convert WAV to text (google TTS)
            try:
                print("Converting recording to text ")
                text = audioTurtle.wavToText(config.audio.recordingPath)

                # convert WAV to spec
                print("Converting recording to spectogram")
                visTurtle.makeSpec(file=config.audio.recordingPath,
                                   fname=config.audio.specPath)
                visTurtle.resizeImg(fname=config.audio.specPath,
                                    newSize=(config.UI.imgSize[0], config.UI.imgSize[1]))
                print("Spectogram made!")

                # loading
                specAiModel = dbTurtle.loadModel(config.SpecAI.modelPath, filetype="hdf5")

                # prediction
                textVal = UIHelper.textPredict(text)
                specVal = UIHelper.specPredict(model=specAiModel,
                                               imgPath=config.audio.specPath,
                                               size=config.UI.imgSize)
                emojiID = UIHelper.getID(textValue=textVal,
                                         specValue=specVal[0])
                face = UIHelper.getEmoji(emojiID)

                # output
                print(emoji.emojize(f"{text} {face}"))

                print(f"Press Enter to start recording again for {config.audio.duration} seconds \n"
                      f"ESC to exit")

            except sr.UnknownValueError:
                print("Something went wrong")
                print(f"Press Enter to start recording to try again for {config.audio.duration} seconds \n"
                      f"ESC to exit")

        elif keyboard.is_pressed('esc'):
            print("Exiting")
            break
