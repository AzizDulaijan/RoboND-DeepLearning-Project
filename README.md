[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png
[image_1]: images/0_run1_cam1_00038.jpeg
[image_2]: images/1_run2_cam1_00004.jpeg
[image_3]: images/Diagram.png


## Collecting Training Data ##
Although I did collect training data from the simulator, I didn't use them to train the network. here are some of the processed training data I collected:

![alt text][image_1] 
![alt text][image_2] 

On one run I did with the patrol points, hero path, and spawn points. The other I did the following manually.  

## Segmentation Network

### Encoders block:
the Encoder block contain one or multiple conventional layers, that is used to extract and identify characteristics from images. each layer captures features then feeds it in to the next layer that will find more convoluted features. 
the encoders used are separable conventional layers that reduces the number of parameters in the network, which makes it more efficient. 

```python
def encoder_block(input_layer, filters, strides):
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
  
    return output_layer
```

### Conv 1x1:
in order to keep spactial information and have a fully connected layer the 1x1 convelation layer is impelemnted. the layer lives between the Encoder layer and the decoder. 

### Decoders block:
Decoders are used to do upsampling and recover information that was lost in the Encoder layers. the block has one or more layers that enable precise localization of features.  

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsample = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    output = layers.concatenate([upsample, large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(output,filters)
    
    return output_layer
```

### skip connections:
to better retain information the "skip connections" technic is used. using features from diffrent resolutions helps combining characteristics information with spatial information.



![alt text][image_3] 


## Training, Predicting and Scoring ##
I did the network training in the netbook Lab, and with the given dataset. 


With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

### Training your Model ###
**Prerequisites**
To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected. 

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

## Scoring ##

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

