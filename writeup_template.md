

#Traffic Sign Recognition

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:

    Load the data set (see below for links to the project data set)
    Explore, summarize and visualize the data set
    Design, train and test a model architecture
    Use the model to make predictions on new images
    Analyze the softmax probabilities of the new images
    Summarize the results with a written report

Rubric Points

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

    You're reading it! and have a look at my html file for full code.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle to load the data set and then used sklearn split to split training data set into training and validation data set (20%).

    The size of training set is = 27839
    The size of the validation set is = 6960
    The size of test set is =12630
    The shape of a traffic sign image is 32,32,3
    The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

    The vidualisation of datasets could be seen on html file.

    It hightlights number of data for each label and it could be seen some labels will not get enough training. 

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

    As a first step, I decided to convert the images to grayscale because it will reduce the data size and colour is not a deciding factor for classifing traffic signs.

    I used openCV function to process the whole training dataset into a new array and image could be find on html file.

    As a last step, I normalized the image data using a built in function in opencv because it will improve learning with zero mean and variance. However, I;m not sure why the output is not what i expected.

    I decided to generate additional data because it will help me understand if the model is able to classify new images. I downloaded images from Google.

    To add more data to the the data set, I cropped the images and the passed it through same processing as above using openCV commands. View images using html file. 



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

    My final model based on LeNet lab consisted of the following layers:
Layer 	2 convolutional layers with 2 pooling layers in between and then connected to 2 fully connected layers and 1 output layer.
Input 	32x32x1 grayscale
Convolution 3x3 	1x1 stride, valid padding, outputs 32x32x64
RELU 	
Max pooling 	2x2 stride, outputs 16x16x64
Convolution 3x3 	etc.
Fully connected 	etc.
Softmax 	etc.
	
	

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

    To train the model, I used adam optimizer with learning rate of 0.001 and batch size of 128 to train the model. I only ran 10 epochs on my laptop as that was sufficient to obtain high accuracy. 
    Also, I created a prediction function to obtain predictions for new images using argmax function. 
    To view my model in Tensorboard - I assigned each parameter a name to make it easier to visualise the state of network later on the code.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

    training set accuracy of =96.2%
    validation set accuracy of 96%
    test set accuracy of 86%


If a well known architecture was chosen:

    What architecture was chosen?
    I chose LeNet architecture because it is a proven model and was sufficient to obtain required accuracy
    Why did you believe it would be relevant to the traffic sign application?
    Since the colour was not a factor and traffic signs are consistent then this model was sufficient. If it was faces of people then maybe a more complex architecture would have been required.
    How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    The model achieved its desired target and it could be further improved in future by increasing epochs and adding more layers.
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

    My images can be found on my html file. 
    I am suprised the model was not able to predict my first image because the training data set had over 1000 images of 100km sign. Maybe my image processing requires tweaking in future. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

    Here are the results of the prediction:
    
    100 km/h 	No passing for vehicles over 3.5 metric tons (With 99% accuracy)
    Yield 	Yield (with 100% accuracy)
    Stop Sign 	Stop sign (with 100% accuracy)
    Ahead only	 Ahead only (with 100% accuracy)
    Keep right 	Keep right(with 100% accuracy)

    The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 86%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

    The code for making predictions on my final model is located in the 16 -26th cell of the Ipython notebook.

    For the first image, the model is relatively sure that this is a no passing sign (probability of 0.99), but the image does not contain that sign. This makes me wonder if my processing of images was not sufficient for this model. The top five soft max probabilities were
    
    Probability 	Prediction
    1.0 	Yield
    1.0 	Stop Sign
    1.0 	Ahead only
    1.0 	Keep right
    0.99 	No passing for vehicles


(Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

        This was really interesting to discover feature maps for convolutional layer 1 because it can be seen for my code that model was able to detect arrow as a feature but when it came to 100km it only detected the circle (roundness) as a feature which explains why it did not make an accurate prediction. 
        I was able to define this code by only looking at the architecture to visualise the network because if i did not assign names for the parameters then it would have been impossible to view their states.

