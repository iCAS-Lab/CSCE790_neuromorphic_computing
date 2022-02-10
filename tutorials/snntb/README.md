# SNNToolbox Conversion and Simulation Tutorial

Before running the `snntb.py` python script you will need to have ran the code for the respective model in `../modelgen` to produce the dataset files (`*.npz`) and the model file (`*.h5`).

The `snntb.py` file contains the code to convert an ANN to an SNN using the SNNToolbox. The main part you will need to modify is the configurable parameters at the top of the file as seen below.

Configurable parameters:

```
NUM_STEPS_PER_SAMPLE = 18  # Number of timesteps to run each sample
BATCH_SIZE = 32             # Affects memory usage. 32 -> 10 GB
NUM_TEST_SAMPLES = 100      # Number of samples to evaluate
CONVERT_MODEL = True
```

You should note that changing the `BATCH_SIZE` to higher values increases the memory usage significantly. This is because more images are being simulated at once.

The code generates a configuration file that is used by the SNNToolbox. This configuration, excluding the parameters changed by the configurable parameters above, does not necessarily need to be changed unless you want to experiment with different configurations.

See the [SNNToolbox](https://snntoolbox.readthedocs.io/en/latest/guide/configuration.html) documentation for more information on the configuration files.