from flask import Flask
from flask_restful import Api, Resource

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

app = Flask(__name__)
api = Api(app)
sys.path.append(os.getcwd())
dbTurtle = databaseHelper.Turtle(filename=config.SpecAI.dbFile)
visTurtle = visHelper.Turtle()
audioTurtle = audioHelper.Turtle(rate=config.audio.sampleRate)
specTurtle = specAiHelper.Turtle()


class steps:
    @staticmethod
    def download(link):
        filePath = ""
        return filePath

    @staticmethod
    def normalizer(filePath):
        # noise reduction
        # normalize volume
        pass

    @staticmethod
    def SpectoEngine(filePath):
        specPath = ""
        visTurtle.makeSpec(file=filePath,
                           fname=specPath)
        visTurtle.resizeImg(fname=specPath,
                            newSize=(config.UI.imgSize[0], config.UI.imgSize[1]))

    @staticmethod
    def STTEngine(filePath):
        transcript = audioTurtle.wavToText(filePath)
        return transcript

    @staticmethod
    def NLPEngine(transcript):
        sentiment = UIHelper.textPredict(transcript)
        return sentiment

    @staticmethod
    def GrammarEngine(transcript):
        CorrectedTranscript = ""
        return CorrectedTranscript

    @staticmethod
    def SpectoPredict(specAiModel, specPath):
        specVal = UIHelper.specPredict(model=specAiModel,
                                       imgPath=specPath,
                                       size=config.UI.imgSize)


class Audio(Resource):
    def get(self, link):
        """link : link to the audio from the user"""

        return {'link': link}


api.add_resource(Audio, '/<string:link>')
if __name__ == '__main__':
    app.run(debug=True)
