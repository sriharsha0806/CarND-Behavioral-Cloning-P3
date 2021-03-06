#**Behavioral Cloning** 
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



[//]: # (Image References)

[image1]: ./nvidia_architecture.PNG "Model Visualization"
[image2]: ./dataset_steering.png "steering  Dataset "
[image3]: ./center_2016_12_01_13_30_48_287.jpg "Center Image"
[image4]: ./left_2016_12_01_13_30_48_287.jpg "Left Image"
[image5]: ./right_2016_12_01_13_30_48_287.jpg "Right Image"
[image6]: ./2017_04_03_16_19_07_524.jpg "preprocessed Image"
[video1]: ./run1.mp4
[video2]: ./run1_02.mp4

# Approach for Designing Model Architecture
* Initally i tried with Nvidia neural network without any preprocessing and but with Data Augmentation. The Model is not performing well. So i tried some preprocessing steps of my own and the model is based on S color of HSV image It is getting confused because of shadow of tree before bridge. 
* I used Udacity Datatset so that i can work and compare my code with other members of my project. 
* Udacity training set consists of 8036 samples. Each sample has frontal, left, right camera, throttle, steering, throttle, brake, speed.
* The dataset has been cropped for saving computational power. The images after cropping can be seen in the images below.
* Based on vivek's blog, I have implemented mirroring the images and changing the brighness of images randomly.

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* config.py: project configuration and hyperparameters
* load_data.py: definition of data generator + handling data augmentation
* Data Analysis.ipynb: Data Analysis on the datase

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run1
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy
I have implemented NVIDIA and commaai models. NVIDIA is comparatively performing during initial stages so i stick with Nvidia for working on Behavioral Cloning. 

####1. An appropriate model architecture has been employed
# NVIDIA 
```sh

Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 66, 220, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 108, 24)   1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 31, 108, 24)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 52, 36)    21636       elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 14, 52, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 24, 48)     43248       elu_2[0][0]                      
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 5, 24, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 22, 64)     27712       elu_3[0][0]                      
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 22, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 20, 64)     36928       elu_4[0][0]                      
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 1, 20, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1280)          0           elu_5[0][0]                      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1280)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           128100      dropout_1[0][0]                  
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 100)           0           elu_6[0][0]                      
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_2[0][0]                  
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 50)            0           elu_7[0][0]                      
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_3[0][0]                  
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 10)            0           elu_8[0][0]                      
____________________________________________________________________________________________________
elu_9 (ELU)                      (None, 10)            0           dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          elu_9[0][0]                      
====================================================================================================
Total params: 265,019
Trainable params: 265,019
```

# Comma ai
```sh
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_2 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 40, 80, 16)    3088        lambda_2[0][0]                   
____________________________________________________________________________________________________
elu_10 (ELU)                     (None, 40, 80, 16)    0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 20, 40, 32)    12832       elu_10[0][0]                     
____________________________________________________________________________________________________
elu_11 (ELU)                     (None, 20, 40, 32)    0           convolution2d_7[0][0]            
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 10, 20, 64)    51264       elu_11[0][0]                     
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 12800)         0           convolution2d_8[0][0]            
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 12800)         0           flatten_2[0][0]                  
____________________________________________________________________________________________________
elu_12 (ELU)                     (None, 12800)         0           dropout_5[0][0]                  
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 512)           6554112     elu_12[0][0]                     
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 512)           0           dense_5[0][0]                    
____________________________________________________________________________________________________
elu_13 (ELU)                     (None, 512)           0           dropout_6[0][0]                  
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 1)             513         elu_13[0][0]                     
====================================================================================================
Total params: 6,621,809
Trainable params: 6,621,809
```

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 28, 31, 34, 47). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 70-71). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Several other techniques were used to prevent overfitting like regularizers for Convolution layers.

####3. Model parameter tuning
The model was trained using an Adam optimizer with a learning rate of 0.00001. The training was done for 5 epochs(model.py line 69).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road based on probability of 0.5. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a NVIDIA model and commaai model

My first step was to implement the network as mentioned in the NVIDIA and comma ai paper model and added dropout layers based on the performance of car on track. I thought this model might be appropriate because It is not wobbling at all at throttle of 0.25. The comma ai model is wobbling a lot during the drive so i dropped comma ai model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a low mean squared error on the validation set. This implied that the model was not overfitting. 

To combat the overfitting, I modified the model by adding dropouts and regularizers so that car will stay on track. 


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track like after the bridge, near lake and i tried to mprove the driving behavior in these cases by adding more data to the driving_log,

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and two laps on tracks using center lane driving using reverse direction and two laps of track 2. Here is an example image of center lane driving and steering angle. The visualization shows the most of times the car is on the straight road:

![alt text][image2]
![alt text][image3]
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help in generalizing for neural network

After the collection process, I had X number of data points. I then preprocessed this data.
I finally randomly shuffled the data set and put 2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer.

The final results can be found in the links given below
[with a throttle 0.25](./run1.mp4)
[with a throttle 0.2](./run.mp4)

# Discussion
 I have checked car driving with various throttle values. The car can succesfully complete the track  with a maximum  throttle of 0.27 but with lot of wobbling. So i wanted to address the relationship between throttle and wobbling of the car using reinforcement learning in the future. I tried to complete track 2 but for ascending the car i need at least throttle of 0.3 so i had manually move the car at some places to complete the track. 
