[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Deep Learning Project 

In this project, I will train a deep neural network to identify and track a target in a simulation. the trained network will be used to run an applications that called “follow me”.   

[image_0]: ./docs/misc/sim_screenshot.png
[image_1]: images/0_run1_cam1_00038.jpeg
[image_2]: images/1_run2_cam1_00004.jpeg
[image_3]: images/Diagram_2.png
[image_4]: images/test_5.PNG
[image_5]: images/test_2_2ff.PNG
[image_6]: images/test_(1).PNG
[image_7]: images/test_(2).PNG
[image_8]: images/test_nn.PNG
[image_9]: images/test_nn_1.PNG
[image_10]: images/test_nn_2.PNG



## Collecting Training Data ##
Although I did collect training data from the simulator, I didn't use them to train the network. here are some of the processed training data I collected:

![alt text][image_1] 
![alt text][image_2] 

On one run I did with the patrol points, hero path, and spawn points. The other I did the following manually.  

## Segmentation Network

### Encoders block:
The Encoder block contain one or multiple conventional layers, that is used to extract and identify characteristics from images. each layer captures features then feeds it in to the next layer that will find more convoluted features. 
The encoders used are separable conventional layers that reduces the number of parameters in a conventional layer, which makes it more efficient. 

```python
def encoder_block(input_layer, filters, strides):
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
  
    return output_layer
```

### 1x1 convolutions:
In order to keep spatial information and have a fully connected layer the 1x1 convolution layer is implemented. The layer lives between the Encoder block and the decoder block. 

### Decoders block:
Decoders are used to do upsampling and recover information that was lost in the Encoder layers. The block has one or more layers that enable precise localization of features.  

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

Here is a diagram that demonstrate the architecture of the network:

![alt text][image_3]

Network model code:

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

## Training and Scoring ##
I did the network training in the netbook Lab, and with the given dataset. 

With the fully convoluted network built, the network can now be trained. the training performance can differ based on the values of some Hyperparameters that are listed below:


### Batch size:
I started training with 128 baches in beginning, but I soon realized 128 was too large and the training time was taking 14-15 for each epoch. so I started lowering the size until I ended with 16. Low value batch size allows me to train for more data. 

### learning rate:
Lowering the learning rate should increase the accuracy, but 0.001 seems too low. the below graphs shows that the network stopped improving even after running it for 100 e_poches. but after increasing the learning rate to 0.01 I got lower than 0.2 loss, and the final results got better. 


![alt text][image_4] 

Final score: 0.329997240629

loss: 0.0233 - val_loss: 0.0398

![alt text][image_5] 


final score: 0.399550650298

loss: 0.0141 - val_loss: 0.0348

### Num_epochs
At the start I was using high batch size, so it was unrealistic to run more than 5 epochs. but as I lowered the batch size I started increasing the number of epochs. I start with a high number at the beginning between 20-50 for two to three times. Then I run between 5-10 epochs to see if network stabilized or not. I do that until I see the final results starts decreasing. 

## Scoring ##
In my the run that I got the best score (0.41) I did trained the network twice, here are the two runs and its results:

First training run:

learning_rate = 0.01,
batch_size = 16,
num_epochs = 50,
steps_per_epoch = 200,
validation_steps = 50,
workers = 2

![alt text][image_6] 

loss: 0.0197 - val_loss: 0.0257

Final score: 0.378018058452

Secound training run:

learning_rate = 0.01,
batch_size = 16,
num_epochs = 20,
steps_per_epoch = 500,
validation_steps = 50,
workers = 2


![alt text][image_7] 

 loss: 0.0148 - val_loss: 0.0223
 
 

Closed range: number true positives: 539, number false positives: 0, number false negatives: 0

The average IoU for the hero is 0.8917756054110004. 

![alt text][image_9] 

patrolling with no target: number true positives: 0, number false positives: 74, number false negatives: 0

As you can tell the average IoU for the hero is 0.0.

![alt text][image_8] 

Far range target: number true positives: 134, number false positives: 3, number false negatives: 167

The average IoU for the hero is 0.22613506447592274, which probably the number that holding the performance down.

![alt text][image_10] 


Sum all the true positives: 0.7339149400218102

IoU for the dataset that never includes the hero: 0.558955334943

#### Final score: 0.41022567112


## Limitations:
Because the hero was wearing red clothes she can be easily distinguished from the background and she was the only one wearing red, this made training the network comparing with the real world easy. And sense other objects such as car, cats, dogs are not easily distinguished from the background it would require a more complicated network to achieve it. 

## Future Enhancements
As I stated before when the target was far from the drone is where the network is having the greatest difficulty. If I have the time I would collect more data that has this kind of images, then use it in the training. I would also try to test the hyperparameters more, and try to understand what is the limit of the network. 


I would also like to run the training on different hardware , and cloud platforms to see how training time would vary

### Note:
The "model_training.ipynb" and its  html version can be found in the code folder. and the weights can be found under the data folder. 

