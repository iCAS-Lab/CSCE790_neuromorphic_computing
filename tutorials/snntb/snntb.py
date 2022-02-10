import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import configparser
import tensorflow as tf
################################################################################
# SNNToolbox Imports
from snntoolbox.bin.run import main
################################################################################
# Configure parameters for config
NUM_STEPS_PER_SAMPLE = 18  # Number of timesteps to run each sample (DURATION)
BATCH_SIZE = 32             # Affects memory usage. 32 -> 10 GB
NUM_TEST_SAMPLES = 100      # Number of samples to evaluate or use for inference
CONVERT_MODEL = True      
################################################################################
# Paths
MODEL_FILENAME = 'lenet.h5'
MODEL_NAME = MODEL_FILENAME.strip('.h5')
CURR_DIR = os.path.abspath('.')
ANN_MODEL_PATH = os.path.join(CURR_DIR, 'models', MODEL_FILENAME)
WORKING_DIR = os.path.join(CURR_DIR, 'snntb_runs')
DATASET_DIR = os.path.join(CURR_DIR, 'data')
################################################################################
# Check paths exist and TODO: cleanup last run
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
assert os.path.exists(ANN_MODEL_PATH) ,'ERROR --> model path not found.'
assert os.path.exists(DATASET_DIR), 'ERROR --> data dir not found.'
assert len(os.listdir(DATASET_DIR)) > 1, 'ERROR --> data files in data dir not valid.'
################################################################################
# Copy model to working directory
os.system('cp {} {}'.format(ANN_MODEL_PATH, WORKING_DIR))  
################################################################################
# Generate Config file
config = configparser.ConfigParser()
config['paths'] = {
    'path_wd': WORKING_DIR,
    'dataset_path': DATASET_DIR,
    'filename_ann': MODEL_NAME,
    'runlabel': MODEL_NAME+'_'+str(NUM_STEPS_PER_SAMPLE)
}
config['tools'] = {
    'evaluate_ann': True,
    'parse': True,
    'normalize': True,
    'simulate': True
}
config['simulation'] = {
    'simulator': 'INI',
    'duration': NUM_STEPS_PER_SAMPLE,
    'num_to_test': NUM_TEST_SAMPLES,
    'batch_size': BATCH_SIZE,
    'keras_backend': 'tensorflow'
}
config['output'] = {
    'verbose': 0,
    'plot_vars': {
        'input_image',
        'spiketrains',
        'spikerates',
        'spikecounts',
        'operations',
        'normalization_activations',
        'activations',
        'correlation',
        'v_mem',
        'error_t'
    },
    'overwrite': True
}
# Write the configuration file
config_filepath = os.path.join(WORKING_DIR, 'config')
with open(config_filepath, 'w') as configfile:
    config.write(configfile)
################################################################################
# Convert the model using SNNToolbox
if CONVERT_MODEL:
    main(config_filepath)
################################################################################
# Copy the resulting file to a snn_out dir.
print('Working Directory: '+WORKING_DIR)
print('Dataset Directory: '+DATASET_DIR)
print('SNN is located at: '+WORKING_DIR)