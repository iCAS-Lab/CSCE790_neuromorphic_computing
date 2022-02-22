# ANN Model Generation
The code in this directory, specifically `./src`, has several specific examples of different convolutional neural networks. The files have the same basic boilerplate to load the dataset, save the dataset to npz files and generate/run the networks. 

Feel free to use this code to help you learn tensorflow if you do not know it already and to help in creating ANNs that can be converted to SNNs.

I specifically created the code to make it easy to take the ANNs and dataset npz files and use them with the SNNToolbox. 

Please feel free to reach out to me if you have any questions.

You need just run `python3 <network>` in the `./src` directory and it will create all the folders you need to store your models and data. 

Please take note of the global variables at the top and change them to your needs.

## Output Data
The data is formated and saved to a directory called `data` in the form of `*.npz` files. You will need to copy this data for later use in the SNNTB conversion.

## Output ANN
The output trained ANN will be stored in the `models` directory.

## For example:

The variables below are meant to toggle some settings like removing the model before running, removing the data,  or to train a new model, etc. You do not have to change these by default but you can if you wish not to retrain the network every time.
```
REMOVE_MODEL = True
REMOVE_DATA = True
TRAIN_NEW_MODEL = True
SAVE_DATA_NPZ = True
USE_GPUS = False
N_GPUS = 4
```

The next group of variables are things that you can tweak about the network's dataset like the number of output classes, the batch size, train epochs, and the distribution between the training datset, validation, and test sets.
```
DS_NAME = 'mnist'
OUTPUT_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 30
TRAIN_PERC = 60 # Train percent
VAL_PERC = 10   # Validation percent
TEST_PERC = 30  # Not Needed
```