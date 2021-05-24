# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: /writeup_images/loss_visualization.png "Loss Visualization"
[image2]: /writeup_images/model_summary.png "Model Summary"
[image3]: /writeup_images/model_visualization.png "Model Visualization"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network simialar to the model architecture published by the Autonomous Vehicle Team at NVIDIA. The model includes 5 convolutional layers followed by 4 fully-connected layers. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. A cropping layer is added to remove pixels that do not contain relevant features. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The training loss and validation loss were both low and close to each other after 4-5 epochs - indicating no signs of overfitting.

![Visualizing Loss][image1]

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. In addition, the model was tested on an altogether new track (not part of training and validation sets) and performed fairly well - thus, the model shows great generalisation as well.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Hyper-parameters such as dropout, correction, batch_size were tuned in order to enable the car to stay on the road throughout track 1.

#### 4. Appropriate training data

I utilized the sample training data provided.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build it step-by-step, gradually increasing the complexity and layers in the model and the size of the training data set.

A pre-design model consisting of a single dense layer was used to check if image and steering data was being loaded into the model appropriately.

My first step was to use a convolution neural network model similar to the NVIDIA model architecture as I thought this model might be appropriate because it had already been implemented on a real self-driving car as opposed to other architectures such as LeNet. Initially, only central camera images were used as part of the data set. The data was normalised using a Lambda layer and irrelevant pixels cropped out using a Cropping layer.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

This first model did surprisingly well on the track- manouvering all the left turns but went off-track on the only right turn of the track. I attributed this to the left-turn bias of the track and used image augmentation to add flipped versions of all frames in the data set- thereby doubling training data and improving generalisation.

The next iteration succeeded to stay on track for both left and right turns but frequenty went off-centre and was unable to recover. Left and right camera images (along with their flipped versions) were added to the data set along with corresponding corrected steering angles to aid recoveries. The correction factor was tuned from 0.2-0.4, with the model performing better at a correction value of 0.4 (bolder responses at sharp turns). 

However, this version resulted in touching the roof of memory (6 times increase in data from first version) and also resulted in erratic movement on the bridge of the track. A generator was added to deal with memory issues, reading in a batch_size of data at a time for training on the fly. Dropout layers and activation functions ('relu') were added to each of the fully-connected layers to prevent overfitting and introduce non-linearity respectively.

These final changes resulted in the car being able to traverse the whole track and staying in the centre of the road at most times.

#### 2. Final Model Architecture

The final model architecture is as described in the model summary below:

![Model Summary][image2]

Here is a visualization of the architecture (adopted from the NVIDIA paper). A few layers: Pre-cropped input, Normalisation and one Dense layer are missing in this representation.

![Model Visualization][image3]

#### 3. Creation of the Training Set & Training Process

I did not collect training data as I did not like the way the car was driving using arrow-key inputs. I understood that it would result in a case of garbage-in, garbage-out. I tried to connect a joystick to improve steering and throttle sensitivity, but the simulator did not support it.
