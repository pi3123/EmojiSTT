from os.path import abspath


class UI:
    imgSize = (64, 64, 3)


class audio:
    sampleRate = 44100
    duration = 5
    recordingPath = "X:\Projects\EmojiSTT\\tmp\\recording.wav"
    specPath = "X:\Projects\EmojiSTT\\tmp\\spec.png"


class SpecAI:
    # Paths for training SpecAI
    audioFolder = abspath("Data\\Audio")
    dbFile = abspath("Data\\Audio\\table.json")
    logFile = abspath("tmp\\output.txt")

    specFolder = abspath("Data\\Spec")

    trainFolder = abspath("Data\\Spec\\Train")
    trainCSV = abspath("Data\\Spec\\train.csv")

    testFolder = abspath("Data\\Spec\\Test")
    testCSV = abspath("Data\\Spec\\test.csv")

    SpecModelsFolder = abspath("Models\\Spec")
    modelPath = "X:\Projects\EmojiSTT\Models\Spec\\700_25_2_(64, 64, 3).h5"

    # Training Config
    INPUT_SHAPES = [(64, 64, 3), (128, 128, 3), (256, 256, 3),
                    (64, 64, 1), (128, 128, 1), (256, 256, 1)]
    EPOCHS = [10, 25, 50, 100]
    MODEL_STRUCTURE_ID = [1, 2, 3]
    DATA_SIZE = [50, 500, 1000, 2000, None]


class TextEmoAI:
    trainPath = abspath("Data\\Text\\train")
    testPath = abspath("Data\\Text\\test")

    TextModelsFolder = abspath("Models\\Text")
    modelPath = "X:\Projects\EmojiSTT\Models\Text\\textAi.sav"
