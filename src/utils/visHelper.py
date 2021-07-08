from Specto import Turtle as visTurtle
import scipy
from scipy import io
from scipy.io import wavfile
import math

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Turtle:
    def __init__(self,
                 minMag=0.004416916563059203,
                 maxMag=2026134.8514368106
                 ):
        self.turtle = visTurtle(minMagnitude=minMag,
                                maxMagnitude=maxMag,
                                minPhase=-math.pi,
                                maxPhase=math.pi)

    def makeSpec(self, file, fname):
        rate, audData = scipy.io.wavfile.read(file)

        # Combining 2 channels if needed
        if len(audData.shape) != 1:
            audData = audData.sum(axis=1) / 2

        img = self.turtle.genGramForWav(audData)
        img.save(fname, "PNG")

    @staticmethod
    def resizeImg(fname, newSize):
        img = ImageFile.Image.open(fname)
        img = img.resize(newSize)
        img.save(fname, "PNG")
