import wave
from array import array
import pyaudio
import speech_recognition as sr


class Turtle:
    def __init__(self,
                 channels=1,
                 chunkSize=1024,
                 form=pyaudio.paInt16,
                 rate=48000):
        self.CHANNELS = channels
        self.CHUNK_SIZE = chunkSize
        self.FORMAT = form
        self.RATE = rate
        self.engine = sr.Recognizer()

    @staticmethod
    def normalize(data):
        """ Average the volume out """
        MAXIMUM = 16384
        times = float(MAXIMUM) / max(abs(i) for i in data)

        r = array('h')
        for i in data:
            r.append(int(i * times))
        return r

    def record(self, fname, duration):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK_SIZE)

        frames = []
        for i in range(0, int(self.RATE / self.CHUNK_SIZE * duration)):
            data = stream.read(self.CHUNK_SIZE)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # writing to file
        wavfile = wave.open(fname, 'wb')
        wavfile.setnchannels(self.CHANNELS)
        wavfile.setsampwidth(audio.get_sample_size(self.FORMAT))
        wavfile.setframerate(self.RATE)
        wavfile.writeframes(b''.join(frames))  # append frames recorded to file
        wavfile.close()

    def wavToText(self, file):
        with sr.AudioFile(file) as source:
            audio_data = self.engine.record(source)
            text = self.engine.recognize_google(audio_data)

        return text
