# EmojiSTT
Speech to Text that detects emotions from voice and text to make the texting experience better by adding emojis and embellishments (Caps, exclamation points etc.)

## Installation
```pip install -r requirements.txt```

## Using pre-trained model
1. Run the ```main.py in``` [EmojiSTT/src](https://github.com/pi3123/EmojiSTT/tree/main/src)

## Training models
1. Edit ```config.py``` in [EmojiSTT/src](https://github.com/pi3123/EmojiSTT/tree/main/src) if you want save the model somewhere else other than the default path.

2.  Edit the training configuration in ```config.py```.
```Note: There can be more than one value in the configs, the training code will iterate through all of them and save a model with a filename based on the configuration in the following format:```
```		[size]_[epochs]_[modelID]_[input_shape].h5```
3. Download training data from [RAVDESS Emotional speech audio | Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) and unzip it to EmojiSTT/Data/Audio

4. Run ```SpecAI.py```

## Roadmap
1. Better Training system
2. Better Model (512x512 RGB)
3. UI for ```main.py```

## Citation
Training data from : 
[RAVDESS Emotional speech audio | Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)