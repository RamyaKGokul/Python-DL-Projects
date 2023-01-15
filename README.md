# Handwritten Digit Classifier with MNIST dataset

An implementation of convolution neural network using keras and tensorflow on the MNIST data to classify handwritten digits and achieved an accuracy of over 99%.

## MNIST dataset:

The MNIST database (Modified National Institute of Standards and Technology database) of handwritten digits consists of a training set of 60,000 examples and a test set of 10,000 examples. 
It is a subset of a larger set available from NIST. Additionally, the black and white images from NIST were size-normalized and centered to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.

## CNN:

The convolution neural network is a feed forward neural network that is generally used for Image recognition and object classification. 
It is made up of neurons stacked in layers which are defined by weights and biases. The model is built using different layers and activation functions and compiled with 'adam' optimizer, loss function 'categorical crossentropy' and with 'accuracy' evaluation metrics.

This model contains Conv2d, Maxpooling2d, Flatten and Dropout layers with 'Relu' activation function. The kernel size and pool size are set to (3,3).

Conv2D - This layer creates a convolution kernel that moves over the layer's input to produce a tensor of outputs
Maxpooling2D - This layer does a pooling operation that selects the maximum element from the region of the feature map covered by the filter
Flatten - This layer is used to convert all the resultant 2-Dimensional arrays from pooled feature maps into a single long continuous linear vector
Dropout - This layer randomly removes unwanted neurons during training to prevent overfitting of the model

<img width="365" alt="Screen Shot 2023-01-15 at 1 08 19 PM" src="https://user-images.githubusercontent.com/122699563/212558944-50fe13db-bb99-4fae-903b-5f4beaaf7050.png">

### Sample output:

Sample Predictions with test data

Predicted Labels
 [7 2 1 0 4 1 4 9 5 9 0 6]
 
Actual Labels
 [7 2 1 0 4 1 4 9 5 9 0 6]
 
Overall Accuracy:	99.03%

<img width="525" alt="Screen Shot 2023-01-15 at 1 09 05 PM" src="https://user-images.githubusercontent.com/122699563/212559019-6eb8bde4-932f-4c21-a08d-09929a17391e.png">

