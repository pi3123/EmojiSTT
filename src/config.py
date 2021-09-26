class UI:
    imgSize = (64, 64, 3)


class audio:
    sampleRate = 44100
    duration = 10
    recordingPath = "X:\Projects\EmojiSTT\\tmp\\recording.wav"
    specPath = "X:\Projects\EmojiSTT\\tmp\\spec.png"


class SpecAI:
    # Paths for training SpecAI
    audioFolder = "X:\Projects\EmojiSTT\Data\Audio"
    dbFile = "X:\Projects\EmojiSTT\Data\Audio\\table.json"
    logFile = "X:\Projects\EmojiSTT\\tmp\output.txt"

    specFolder = "X:\Projects\EmojiSTT\Data\Spec"

    trainFolder = "X:\Projects\EmojiSTT\Data\Spec\Train"
    trainCSV = "X:\Projects\EmojiSTT\Data\Spec\\train.csv"

    testFolder = "X:\Projects\EmojiSTT\Data\Spec\Test"
    testCSV = "X:\Projects\EmojiSTT\Data\Spec\\test.csv"

    SpecModelsFolder = "X:\Projects\EmojiSTT\Models\Spec"
    modelPath = "X:\Projects\EmojiSTT\Models\Spec\\700_25_2_(64,64,3).h5"

    # Training Config
    INPUT_SHAPES = [(64, 64, 3), (128, 128, 3), (256, 256, 3),
                    (64, 64, 1), (128, 128, 1), (256, 256, 1)]
    EPOCHS = [10, 25, 50, 100]
    MODEL_STRUCTURE_ID = [1, 2, 3]
    DATA_SIZE = [50, 500, 1000, 2000, None]
