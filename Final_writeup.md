# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Test_Data/training_histogram.png "Visualization"
[image4]: ./Test_Data/11_(rightofway).jpg "Traffic Sign 1"
[image5]: ./Test_Data/12_(priority).jpg "Traffic Sign 2"
[image6]: ./Test_Data/13_(yield).jpg "Traffic Sign 3"
[image7]: ./Test_Data/17_(no_entry).jpg "Traffic Sign 4"
[image8]: ./Test_Data/34_(turn_left).jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
The pickled data is a dictionary with 4 key/value pairs:

'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data and the amount of pictures that belong to certain classes

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale and I shuffled the data as a part of preprocessing just to make sure that the order in which the data comes does not matters to CNN

As a last step, I normalized the image data because it flattens the image to work with LeNET. it changes the array from a 32 by 32 by 3 to a 32 by 32 by 1



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I studied LeNet Architecture and I started implementing LeNet Architecture but soon I realized that there is a need of additional Convolutional Layer and Fully Connected Layer. So below I am describing my modified LeNet Architecture.

Layer	Description
Input	32x32x1 gray scale image
Convolution 5x5	1x1 stride, valid padding, outputs 28x28x6
RELU
Max pooling	2x2 stride, outputs 14x14x6
Convolution 5x5	1x1 stride, valid padding, outputs 10x10x16
RELU
Max pooling	2x2 stride, outputs 5x5x16
Flatten	outputs 400
Dropout          	                                   
Fully connected	outputs 120
RELU
Dropout          	                                   
Fully connected	outputs 84
RELU
Dropout          	                                   
Fully connected	outputs 43
Softmax


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained my model using the following hyperparameters-

Epochs - 150

Batch Size - 128

Learning Rate - 0.0009

Optimizer- AdamOptimizer

mu - 0

sigma - 0.1

dropout keep probability - 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.973
* test set accuracy of 0.957


As an initial model architecture the original LeNet model from the course was chosen. The inputs and the outputs had to be adjusted to work with our data. The training accuracy was 81.3%. (hyper parameters: EPOCHS=8, BATCH_SIZE=128, learning_rate = 0,001, mu = 0, sigma = 0.1)

After adding the grayscaling preprocessing the validation accuracy increased to 91% (hyperparameter unmodified)

The additional normalization of the training and validation data resulted in a small improvement (hyperparameter unmodified)


increased number of epochs and reduce learning rate, added dropout layer after relu of final fully connected layer: validation accuracy = 94,7% (EPOCHS = 30, BATCH_SIZE = 128, rate = 0,0009, mu = 0, sigma = 0.1)

added dropout after relu of first fully connected layer and before validation

increase of epochs. validation accuracy = 97,5% (EPOCHS = 150, BATCH_SIZE = 128, rate = 0,0009, mu = 0, sigma = 0.1)

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

some of these images might be hard to classify due to the background in them

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right of way    		| Right of way 									|
| priority     			| priority 										|
| Yield					| Yield											|
| no entry	      		| no entry  					 				|
| turn left 			| turn left         							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a right of way (probability of 0.99), and the image does contain a a rigth of way. The top five soft max probabilities were: (The probability was so high that the other values we close to 0 but eventually all add up to 1)

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| right of way 									|
| 0.0     				| keep right 									|
| 0.0					| Yield											|
| 0.0	      			| ahead only  					 				|
| 0.0				    | priority road        							|

second image
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| priority road 								|
| 0.0     				| Roundabout mandatory 							|
| 0.0					| keep right									|
| 0.0	      			| Speed limit (20km/h)  		 				|
| 0.0				    | no passing        							|

third image:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| yield      									|
| 0.0     				| ahead only 									|
| 0.0					| road work										|
| 0.0	      			| Turn left ahead  								|
| 0.0				    | Go straight or right 							|

fourth image:
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99         			| Right-of-way at the next intersection 		|
| 0.0     				| Beware of ice/snow 							|
| 0.0					| Pedestrians									|
| 0.0	      			| Double curve          		 				|
| 0.0				    | Children crossing    							|

fifth image:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.99         			| No entry      								|
| 0.0     				| stop      									|
| 0.0					| Speed limit (20km/h)							|
| 0.0	      			| Keep right    								|
| 0.0				    | Speed limit (120km/h) 						|
