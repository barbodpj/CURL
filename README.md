# [Batch Size > 1 version of CURL: Neural Curve Layers for Global Image Enhancement (ICPR 2020)](https://github.com/sjmoran/CURL)
This code supports Batch Size > 1 and in addition some parts of it have been optimized. For instance, in computing the loss function, the loops have been replaced by matrix operations so now they will be calculated in parallel.
In the training phase every few steps (determined by "valid_every" argument) evaluation on the validation is done and the results get written in the tensorboard. In addition to the scalar values I've provided output images for different steps in the tensorboard. 

You can choose between training and testing in the arguments
Other Arguments are similar to the base code.


<div class="row">
  <div class="column">
    <img src="samples/in1.jpg" alt="Snow" style="width:80%">
      <figcaption><center>Input</center></figcaption>
  </div>
  <div class="column">
    <img src="samples/res1.jpg" alt="Forest" style="width:80%">
      <figcaption><center>Output</center></figcaption>
  </div>
</div>

