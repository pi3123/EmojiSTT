import wave
import pyaudio
import speech_recognition as sr

from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)


class Turtle:
    def __init__(self,
                 channels=1,
                 chunkSize=1024,
                 form=pyaudio.paInt16,
                 rate=48000):
        """
        :param channels: 1 for mono, 2 for stereo
        :param chunkSize: size of each chunk in WAV
        :param rate: Bitrate of audio
        """
        self.CHANNELS = channels
        self.CHUNK_SIZE = chunkSize
        self.FORMAT = form
        self.RATE = rate
        self.engine = sr.Recognizer()

    def record(self, fname, duration):
        """
        :param fname: output filename
        :param duration: duration of recording
        :return:
        """
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK_SIZE)

        frames = []
        pbar = tqdm(range(0, int(self.RATE / self.CHUNK_SIZE * duration)))
        for i in pbar:
            data = stream.read(self.CHUNK_SIZE)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # writing to file
        wavFile = wave.open(fname, 'wb')
        wavFile.setnchannels(self.CHANNELS)
        wavFile.setsampwidth(audio.get_sample_size(self.FORMAT))
        wavFile.setframerate(self.RATE)
        wavFile.writeframes(b''.join(frames))  # append frames recorded to file
        wavFile.close()

    def wavToText(self, file):
        """
        Google STT driver code
        :param file:
        :return:
        """
        with sr.AudioFile(file) as source:
            audioData = self.engine.record(source)
            text = self.engine.recognize_google(audioData)

        return text
