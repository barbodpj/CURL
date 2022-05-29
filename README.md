# [Batch Size > 1 version of CURL: Neural Curve Layers for Global Image Enhancement (ICPR 2020)](https://github.com/sjmoran/CURL)
This code supports Batch Size > 1 and in addition some parts of it have been optimized. For instance, for computing the loss function, the loops have been replaced by matrix operations so now they will be calculated in parallel.
In the training phase every few steps (determined by "valid_every" argument) evaluation is done on the validation and the results get written in the tensorboard. In addition to the scalar values I've provided output images for different steps in the tensorboard. 

You can choose between training and testing in the arguments.
Other Arguments are similar to the official code.



