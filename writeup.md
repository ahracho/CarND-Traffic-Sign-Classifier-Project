<script type="text/javascript"  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>  

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

[Histogram1]: ./dataset/histogram_trainset.png "Histogram of Train Dataset"
[Histogram2]: ./dataset/histogram_testset.png "Histogram of Test Dataset"
[Normalization]: ./dataset/normalization.jpg "Normalization"
[GradientDescent]: ./dataset/gradientdescent.png "GD"
[Adam]: ./dataset/Adam.png "Adam"
[RMSProp]: ./dataset/RMSprop.png "RMSProp"
[Test Image 1]: ./test-images/test_image_1.jpg "Image 1"
[Test Image 2]: ./test-images/test_image_2.jpg "Image 2"
[Test Image 3]: ./test-images/test_image_3.jpg "Image 3"
[Test Image 4]: ./test-images/test_image_4.jpg "Image 4"
[Test Image 5]: ./test-images/test_image_5.jpg "Image 5"
[Top K]: ./dataset/top_k.png "Top K"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### 1. Submission Files

Assignments are uploaded on [github repository](https://github.com/ahracho/CarND-Traffic-Sign-Classifier-Project).

- [Ipython Notebook](https://github.com/ahracho/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
- [HTML File](https://github.com/ahracho/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)
- [Write-up](https://github.com/ahracho/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md)


### 2. Data Set Summary & Exploration

#### (1) Basic summary of the data set.
I used numpy and pandas built-in methods to get basic statistics of given datasets.

~~~python
n_train = X_train.shape[0]                  # Number of training examples = 34799
n_validation = X_valid.shape[0]             # Number of validation examples = 4410
n_test = X_test.shape[0]                    # Number of testing examples = 12630

image_shape = (X_train.shape[1:])           # Image data shape = (32, 32, 3)

n_classes = len(pd.unique(y_train))         # Number of classes = 43
~~~


#### (2) Exploratory visualization of the dataset
Train / Validation / Test sets should have similar distribution so as to get meaningful results from test dataset. I draw histograms of labels from train and test dataset to check if they share similar distribution. Results are as below. They seem to have same distribution. 

**Train Dataset**  
![Train Dataset][Histogram1]  

**Test Dataset**  
![Test Dataset][Histogram2]  


### Design and Test a Model Architecture

#### (1) Data Preprocessing

I tried two Preprocessing techniques; One is grayscaling, the other is normalization. 

~~~python 
import numpy as np
import cv2

def gray_scale(dataset):
    gray_image = np.empty((0, 32, 32))
    for image in dataset:
        gray_image = np.append(gray_image, [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)], axis=0)
        
    return gray_image

X_train = gray_scale(X_train)
~~~

For grayscaling, I used opencv API as I tried in previous finding line lanes project. I expected using grayscaling would reduce the operations and shorten the training time. But using cv2.cvtColor() API for each image takes too much time while it has little effects on increasing prediction accuracy. Between accuracy and training time trade-off, I chose training time, and decided not to use grayscale. I am not sure if there is the way to grayscale bunch of images in a very short time, but at this point, I couldn't find one.  


~~~python 
def pixel_normalization(dataset):
    return (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
~~~

I used \\(\frac { X\quad -\quad \mu  }{ \sigma }\\) to normalize pixel data. Images get different as below after normalization.  

![Normalization][Normalization]  


#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| 5x5x16 to 1x400 				|
| Fully connected		| 400 to 120, ReLu        								|
| Fully connected		| 120 to 43        									|

I found out for the convolution layers, using tanh leads to better results than relu. And removed the last fully connected layer, because I thought it drives overfitting on training data. After some epochs training accuracy is close to 100% but validation accuracy was not increased with 3 fully connected layer. So I removed the last one so as to simplify the network and solve overfitting problem. Other than that, I used the same architecture as LeNet Lab. I tried smaller convolution kernel size 3x3 but there was no big improvement.
 

#### (3) Hyperparameters

- Optimizer  
I tried several optimizers: Gradient Descent, Adam Optimizer, and RMSProp. As appeared in graphs below, during 50 epochs gradient descent gets better but it was too slow. Adam and RMSprop get similar result but normally RMSprop leads better result, so I chose RMSprop Optimizer.  

**Gradient Descent**  
![Gradient Descent][GradientDescent]    

**Adam Optimizer**  
![Adam Optimizer][Adam]    

**RMSprop**  
![RMSprop][RMSprop]  


- Epoch  
I tried 10 ~100 epochs. 10 epochs was not enough, and after 50 epochs validation accuracy was not dramatically increased. 

- Batch size  
I chose 128 batch size for this project.


#### 4. the approach taken for finding a solution

I started from LeNet example used in previous exercies. It is quite a good architecture but doesn't reach to the accuracy better than around 90%, so I need to find factors that can improve the validation/test accuracy. I used iterative process which I tuned some hyper-parameters and see how it worked. 

During iterative approach, I just observed train and validation accuracy so that I would not use test dataset like another training set. I tried different hyper-parameters and observed the results.

I changed optimizer to RMSProp; as mentioned earlier, I tried Gradient Descent and Adam Optimizer, and they were either too slow to train or not good enough to get 93% accuracy.

And I changed the network architecture a little bit. I started from LeNet, but in this traffic sign classifier, the output layer should be 43 rather than 10. So I lessen the last fully connected layer (I thought it seemed better and it also can reduce overfitting).

For the convolution layer, I tried several kernel size:smaller and bigger than 5x5, and found out 5x5 normally gave me better result.

In many cases, trade-off between accuracy and time happened to me. I put accuracy first unless the operation takes too much time while improvement is not that big.

 
My final model results were:
* training set accuracy of 100%
* validation set accuracy of 93.3%
* test set accuracy of 92.4%


### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![Test Image 1] ![Test Image 2] ![Test Image 3] 
![Test Image 4] ![Test Image 5]

I tried to find the images from the pictures taken by human rather than graphic images since, traffic signs from real world contains noises (different background, tilted, etc.). 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road      		| Bumpy Road   									| 
| Children Crossing    			| Slippery road					|
| Stop					| Speed limit (70km/h)								|
| 70 km/h	      		| Speed limit (50km/h)				|
| Wild Animal			| Dangerous curve to the left				|


My final model gives 20% accuracy on images found on web. I tried several times but it didn't get better while with given test data, the accuracy was nearly 90%. I'm not sure if this model was overfit to the given data set, or my new images are not good enough for testing.

The only one example I correctly predicted was "Bumpy Road", and as it seems in graph below, it shows biggest difference between the 1st and 2nd choice. I expected even though predictions were not correct, the actual labels were in top 5, but most of them were not :(. I am not sure how to fix this.

![Top K]
