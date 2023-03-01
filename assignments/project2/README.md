# Project 2: SNN Conversion Toolbox

[CSCE 790: Neuromorphic Computing](https://www.icaslab.com/teaching/csce790nc)  
Instructor: [Dr. Ramtin Zand](https://icaslab.com)  
Term: Spring 2022  
Assignment Author: [Peyton Chandarana](https://peytonsc.com)

---

## Assignment Summary

The first assignment's purpose is to get familiar with creating ANN models using a specific dataset and ANN architecture along with using the SNNToolbox to convert the trained model to a Spiking Neural Network (SNN).

There will be two primary steps in this assignment:

1. Create and train the ANN.
2. Convert the ANN to an SNN using the SNNTB built-in simulator, INIsim.

More specificially, you will implement an ANN, VGG-9, which can classify the CIFAR-10 image dataset, train the model, and then convert the ANN model to an SNN.

---

## 0. Preliminaries

You will need to familiarize yourself with the basics of Python3 and Tensorflow prior to starting this assignment. Additionally, if you have not yet watched the SNN Conversion Toolbox lecture (Module 5), please watch it on the MS Teams class meeting chat.

Also take a look at what the CIFAR-10 image dataset looks like so you can more easily visualize what it is you are trying to accomplish.

Take a look at the Python3 scripts in `tutorials/modelgen/src` and familiarize yourself with the code. You will also want to look at the SNN Conversion Toolbox example tutorial that I have provided in `tutorials/snntb`.

To start the project you can just copy over the `modelgen` and the `snntb` directories to the `assignment/hw2` directory.

Please feel free to reach out if you have any questions about what certain chunks do.

---

## 1. Creating the ANN

First, you will need to create a model. You will most likely want to take the python script from the `tutorials/modelgen/src` directory and make a copy to a new model file.

This way you can just worry about changing the dataset and the model architecture.

Once you copy the MNIST example of `lenet.py` from `tutorials/modelgen/src` you will need to first change the dataset to the CIFAR-10 dataset. The simplest way to do this is by importing the dataset from the Tensorflow Datasets package which I have already used in the tutorial files.

The tutorial should handle the image size differences and the fact that the dataset has 3 channels for RGB.

Now you need just implement the model for VGG-9. To learn more about VGG style networks just search for the network architecture online. Note that VGG-11 is more common and you may not find a VGG-9 example. To create VGG-9 you simply need to reduce VGG-11 by two layers.

You may want to ensure that everything is working by first running LeNet-5 on CIFAR-10. The accuracy of CIFAR-10 on LeNet-5 is expected to be poor due to CIFAR-10 containing much more complex images compared to MNIST.

---

## 2. Converting the ANN to an SNN

The second part of this assignment is to convert the trained ANN model for VGG-9 that you created to an SNN using the SNN Conversion Toolbox.

Please take a look at the `snntb.py` file in `tutorials/snntb` prior to starting this part of the assignment.

The SNN Conversion Toolbox requires 2 things:

1. The trained ANN model.
2. The data files from generating the model (i.e. `*.npz` files).

By default the code I have created in the tutorials looks for the models in `tutorials/snntb/models` and the `*.npz` data files in `tutorials/snntb/data`.

Once you have the conversion running and successfully performing inference (i.e. test images are tested and you are given an accuracy).

From here you will need to adjust the duration (NUM_STEPS_PER_SAMPLE) to analyze the accuracy vs. latency tradeoff.

---

## 3. Caveats

Be careful how you adjust the `batch_size` or `BATCH_SIZE` parameter in the `snntb.py` script. This drammatically increases the amount of ram needed to run the simulations. See below:

Batch size approximate memory usage:

- 32 images uses about 10 GB
- 64 images uses about 20 GB
- 128 images uses about 36 GB

---

## 4. Submission

Please provide a write-up of how you solved this assignment. Your write-up should include the following:

1. How you implemented your VGG-9 network? What is the architecture? Include the number of filters and filter sizes.
2. Did you run into any issues while running your experiements or designing your model?
3. Summarize your experimental processes.
4. An analysis of the accuracy vs. latency tradeoff. You should include graphs/figures/tables (at least one).
5. What did you learn?

Your write-up should be submitted as a **_PDF_** and should be written in the style of a professional paper. In other words, your submission should include an abstract, introduction, methodology, results, and conclusion section.

**_Please submit your assignment write-up AND code to Blackboard._**
