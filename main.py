#!/usr/bin/env python3

import glob
import config
import random
import sys

from utils import databaseHelper, visHelper
import AI
from functools import partial
from tqdm import tqdm

tqdm = partial(tqdm, position=0, leave=True)

# Turtles
dbTurtle = databaseHelper(filename=config.SpecAI.dbFile)
visTurtle = visHelper()


def trainModels(dataSize, epochs, ModelIDs, shapes):
    """
    trains the models for all the provided settings and saves them
    :param dataSize: How much data is used for training the model
    :param epochs: How many times the model is trained for
    :param ModelIDs: Which models are used
    :param shapes: the shape of the image (dimensions and colors)
    :return: None, it saves all the models
    """

    # reset Log file
    with open(config.SpecAI.logFile, "w") as f:
        f.write("")
        f.close()

    for size in dataSize:
        print("Making data")

        files = glob.glob(config.SpecAI.audioFolder + "\\*\\*\\*.wav")
        random.shuffle(files)
        if size is not None:
            if size > len(files):
                print("Size too big, loading all available files")
            else:
                files = files[-size:]
                print(f"Loading {size} files and shuffling")
        else:
            print("Loading all files and shuffling")

        pbar = tqdm(files)
        for i in pbar:
            pbar.set_description(f"Processing {i[-24:]}")
            visTurtle.makeSpec(OutFolder=config.SpecAI.specFolder, file=i)

        dbTurtle.prepareData(FolderFiles={config.SpecAI.testFolder: config.SpecAI.testCSV,
                                          config.SpecAI.trainFolder: config.SpecAI.trainCSV},
                             TargetFolder=config.SpecAI.specFolder)

        for epoch in epochs:
            for modelID in ModelIDs:
                for input_shape in shapes:
                    specAI = AI.specAI(input_shape=input_shape,
                                       epoch=epoch,
                                       dataSize=size,
                                       modelID=modelID,

                                       trainCSV=config.SpecAI.trainCSV,
                                       testCSV=config.SpecAI.testCSV,
                                       trainFolder=config.SpecAI.trainFolder,
                                       testFolder=config.SpecAI.testFolder
                                       )

                    score, model = specAI.run()
                    with open(config.SpecAI.logFile, "a") as f:
                        statement = f"####################################################### \n" \
                                    f"Model : {modelID}      Shape : {input_shape}       Epochs : {epoch}        Data Volume : {size}\n" \
                                    f"Average Accuracy : {score}%" \
                                    f"\n"
                        f.write(statement)
                        f.close()

                    # Saving the model
                    #                [size]_    [epochs]_   [modelID]_    [input_shape].h5
                    modelName = f"{str(size)}_{str(epoch)}_{str(modelID)}_{str(input_shape)}.h5"
                    model.save(f"{config.SpecAI.SpecModelsFolder}\\{modelName}")


if __name__ == "__main__":
    trainSpecAICommands = [sys.argv[1] == "--trainSPEC",
                           sys.argv[1] == "-tSPEC"]
    runCommands = [sys.argv[1] == "--run",
                   sys.argv[1] == "-r"]

    if any(trainSpecAICommands):
        INPUT_SHAPES = [(64, 64, 3), (128, 128, 3), (256, 256, 3),
                        (64, 64, 1), (128, 128, 1), (256, 256, 1)]
        EPOCHS = [10, 25, 50, 100]
        MODEL_STRUCTURE_ID = [1, 2, 3]
        DATA_SIZE = [50, 500, 1000, 2000, None]

        trainModels(DATA_SIZE, EPOCHS, MODEL_STRUCTURE_ID, INPUT_SHAPES)
    elif any(runCommands):
        pass
