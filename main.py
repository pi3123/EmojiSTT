#!/usr/bin/env python3

import glob
import config
import random

from utils import databaseHelper, visHelper
import AI
from tqdm import tqdm

dbTurtle = databaseHelper(filename=config.dbFile)
visTurtle = visHelper()


def makeDB(lim=None):
    files = glob.glob(config.audioFolder + "\\*\\*\\*.wav")
    random.shuffle(files)
    if lim is not None:
        if lim > len(files):
            print("Size too big, loading all available files")
        else:
            files = files[-lim:]
            print(f"Loading {lim} files and shuffling")
    else:
        print("Loading all files and shuffling")

    pbar = tqdm(files)
    for i in pbar:
        pbar.set_description(f"Processing {i[-24:]}")
        visTurtle.makeSpec(OutFolder=config.specFolder, file=i)

    dbTurtle.prepareData(FolderFiles={config.testFolder: config.testCSV,
                                      config.trainFolder: config.trainCSV},
                         TargetFolder=config.specFolder)


if __name__ == "__main__":
    INPUT_SHAPES = [(32, 32, 1), (64, 64, 1), (128, 128, 1), (256, 256, 1),
                    (32, 32, 3), (64, 64, 3), (128, 128, 3), (256, 256, 3)]
    EPOCHS = [10, 25, 50]
    MODEL_STRUCTURE_ID = [1, 2, 3]
    DATA_SIZE = [500, 1000, 2000, 3000]
    for size in DATA_SIZE:
        print("Making data")
        makeDB(lim=size)

        print("Training...")
        AI.train(inputShapes=INPUT_SHAPES,
                 Epochs=EPOCHS,
                 IDs=MODEL_STRUCTURE_ID,
                 DataSize=size)
