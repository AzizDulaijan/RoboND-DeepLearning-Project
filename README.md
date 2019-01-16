[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Deep Learning Project 

In this project, I will train a deep neural network to identify and track a target in a simulation. the trained network will be used to run an applications that called “follow me”.   

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
the encoders used are separable conventional layers that reduces the number of parameters in a conventional layer, which makes it more efficient. 

```python
def encoder_block(input_layer, filters, strides):
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
  
    return output_layer
```

### 1x1 convolutions:
in order to keep spatial information and have a fully connected layer the 1x1 convolution layer is implemented. the layer lives between the Encoder layer and the decoder. 

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
To better retain information that was lost from encoders the "skip connections" technique is used .Using features from different resolutions helps combining characteristics information with spatial information.


![alt text][image_3] 


```python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_layer = encoder_block(inputs, 16,2)
    encoder_layer_2 = encoder_block(encoder_layer, 64,2)
    encoder_layer_3 = encoder_block(encoder_layer_2, 128,2)
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv1x1_layer =  conv2d_batchnorm(encoder_layer_3, 128, kernel_size=1, strides=1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_layer  = decoder_block(conv1x1_layer, encoder_layer_2, 64)
    decoder_layer_1  = decoder_block(decoder_layer, encoder_layer, 32)
    decoder_layer_2  = decoder_block(decoder_layer_1, inputs, num_classes)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(decoder_layer_2)
```

## Training, Predicting and Scoring ##
I did the network training in the netbook Lab, and with the given dataset. 


with the fully convoluted network built, the network can now be trained. the training preformace can deffer based on the values of some prameters.

### learning rate:
As I tested 

### batch size 
I started training with 128 bachs in bigning, but I soon relize it the training time was taking 14-15 for each epoch. so I started lowring the size and the result was not very diffrent, so I ended with 16. 
### num_epochs
### steps_per_epoch

### Training your Model ###
**Prerequisites**



To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected. 

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

## Scoring ##

>> show results


## Limitations:

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

