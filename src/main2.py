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

# JSON Format
json = {
    'audioLink': 'ID or whatever',
    'settings': {
        'EmoRecEngine': True,
        'NLPEngine': True,
        'GrammarEngine': True
    }
}

# Setup
sys.path.append(os.getcwd())
dbTurtle = databaseHelper.Turtle(filename=config.SpecAI.dbFile)
visTurtle = visHelper.Turtle()
audioTurtle = audioHelper.Turtle(rate=config.audio.sampleRate)
specTurtle = specAiHelper.Turtle()

if __name__ == "__main__":
    # Get audio

    # Normalize Audio
    print("Normalizing Volume")
    audioTurtle.normalize_volume(config.audio.recordingPath)

    print("Normalizing noise")
    audioTurtle.normalize_noise(config.audio.recordingPath)

    if (json['settings']['NLPEngine'] is True) and (json['settings']['GrammarEngine'] is True):

        """ If both NLP Engine and Grammar Engine are activated, use STT engine and the rest """

        # convert WAV to text (google TTS)
        print("Converting recording to text ")
        text = audioTurtle.wavToText(config.audio.recordingPath)
        print(f"Text from STT : {text}")

        if json['settings']['GrammarEngine'] is True:
            # grammar correct the text
            print("correcting the grammar")
            text = audioTurtle.grammarCheck(text)
            print(f"grammar corrected text : {text}")

        if json['settings']['NLPEngine'] is True:
            # NLP prediction
            textVal = UIHelper.textPredict(text)
            print(f"Text Sentiment : {textVal}")

    if json['settings']['EmoRecEngine'] is True:
        # convert WAV to spec
        print("Converting recording to spectogram")
        visTurtle.makeSpec(file=config.audio.recordingPath,
                           fname=config.audio.specPath)
        visTurtle.resizeImg(fname=config.audio.specPath,
                            newSize=(config.UI.imgSize[0], config.UI.imgSize[1]))
        print("Spectogram made!")
        # loading
        specAiModel = dbTurtle.loadModel(config.SpecAI.modelPath, filetype="hdf5")

        # EmoRec engine
        specVal = UIHelper.specPredict(model=specAiModel,
                                       imgPath=config.audio.specPath,
                                       size=config.UI.imgSize)
        print(f"Spec Sentiment : {specVal}")

    if (json['settings']['NLPEngine'] is True) and (json['settings']['EmoRecEngine'] is True):
        emojiID = UIHelper.getID(textValue=textVal,
                                 specValue=specVal[0])
        face = UIHelper.getEmoji(emojiID)
        # break

    if json['settings']['NLPEngine'] is True:
        face = UIHelper.getEmoji(textVal)
        # break

    if json['settings']['EmoRecEngine'] is True:
        face = UIHelper.getEmoji(specVal[0])
        # break

    # output
    print(emoji.emojize(f"{text} {face}"))

    print(f"Press Enter to start recording again for {config.audio.duration} seconds \n"
          f"ESC to exit")
