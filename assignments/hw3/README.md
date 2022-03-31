# Assignment 3: Encoding/Decoding

[CSCE 790: Neuromorphic Computing](https://www.icaslab.com/teaching/csce790nc)  
Instructor: [Dr. Ramtin Zand](https://icaslab.com)  
Term: Spring 2022  
Assignment Author: [Peyton Chandarana](https://peytonsc.com)

---

## Assignment Summary

In this assignment we will implement encoding and decoding algorithms for encoding temporal signal to spike trains and decoding spike trains to signals. We will also implement several functions for measuring the performance of the implementations.

Three algorithms will be required for full credit with an optional extra credit custom encoding/decoding approach.

Two sets of input will be provided:

- Randomly generated signals
- Real-world signals

Your task is to take these two sets of temporal signals, encode them into spike trains and then decode the spike trains to reconstruct the signals. From there, you will then analyze the orignal signals, the spike trains, and the reconstructed signals to draw some conclusions about the different methods.

## 0. Preliminaries

- To familiarize yourself with encoding and decoding methods you should first read this paper:

> [B. Petro, N. Kasabov and R. M. Kiss, "Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks," in IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 2, pp. 358-370, Feb. 2020, doi: 10.1109/TNNLS.2019.2906158.](https://ieeexplore.ieee.org/document/8689349)

- You can see an example of the work we are expecting in:

> [P. Chandarana, J. Ou and R. Zand, "An Adaptive Sampling and Edge Detection Approach for Encoding Static Images for Spiking Neural Networks," 2021 12th International Green and Sustainable Computing Conference (IGSC), 2021, pp. 1-8, doi: 10.1109/IGSC54211.2021.9651610.](https://arxiv.org/pdf/2110.10217.pdf)

- You should take a look and run the tutorials for the random signal/spike generator and the image to signal conversion tutorials.

---

## 1. Metrics Implementation

You will need to implement 4 functions in this section. The 4 functions are defined in _[B. Petro, et. al.](https://ieeexplore.ieee.org/document/8689349)_ with their mathematical formulas/equations.

- Root Mean Square Error (RMSE)

  Takes in two signals and computes the error between the two signals i.e. how different the signals are.

- Average Firing Rate (AFR)

  Takes in a spike train and computes the average of how often the spike train has an action potential/spike.

- Signal to Noise Ration (SNR)

  Computes the difference between the original signal and the new signal. This difference is considered noise.

- Fitness Function

  This metric combines multiple metrics and allows you to tune which of the metrics matter to you more. It takes in several metrics along with their weights and outputs a value which represents how well the signal was encoded/decoded. You can see an example of a fitness function in _[P. Chandarana, et. al.](https://arxiv.org/pdf/2110.10217.pdf)_

---

## 2. Encoder/Decoder Implementations

In this section you will implement 3 differening encoding algorithms and their respective decoding algorithm(s).

You will need to implement the following 3 encoding algorithms and the decoding algorithm to get full credit step-forward (SF), moving-window (MW), and threshold-based representation (TBR). So in total you should have at least 4 functions.

You can find the algorithms in _[B. Petro, et. al.](https://ieeexplore.ieee.org/document/8689349)_

---

## 3. Extra Credit

For extra credit you can choose to implement your own version(s) of encoding/decoding signals or other data into spike trains. Note, however, you must be able to explain your method and analyze its performance compared to the rest of the algorithms you implement in step 2.

The neuromorphic computing community is always looking for new encoding/decoding algorithms to efficiently represent data as spike trains.

---

## 4. Analysis

This is the most important part of the assignment. After you implement your metric functions and the encoding/decoding algorithms, you should perform an analysis on the performance of the encoding/decoding algorithms.

If you chose to modify the algorithms in any way you should explain how you modified them and why you changed what you did.

---

## 5. Submission

For your assignment submission, like [Assignment 2](../hw2/README.md), you should submit both your code and a report about your experiments to BlackBoard.

The report should be well structured with abstract/introduction, methodology, results, and conclusion sections. You should include enough detail so that we can reproduce your work and get reproduceable results. Your report should include figures like graphs, tables, etc. to help the reader understand what you did.

---
