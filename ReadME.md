# Image Classifier with CIFAR-10 dataset
An implementation of RNN(Long Short term Memory Neural Network)in a transfer learning approach using ResNet50 architecture on the CIFAR-10 data to classify images and achieved an accuracy of over 97%.

## CIFAR-10 dataset:
The CIFAR-10 dataset(Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms.It is one of the most widely used datasets for machine learning research.The CIFAR-10 dataset contains 60,000 32x32 color images that represents 10 different classes such as airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.
Computer algorithms for recognizing objects in photos often learn by example. CIFAR-10 is a set of images that can be used to teach a computer how to recognize objects. Since the images in CIFAR-10 are low-resolution (32x32), this dataset can allow researchers to quickly try different algorithms to see what works.It is a labeled subset of the 80 million tiny images dataset.

## RNN - Transfer Learning:
RNN generally suffers from the vanishing gradient problem where it does not have the ability to propagate useful gradient information from the output end of the model back to the layers near the input end of the model. Hence, to overcome this issue, transfer learning approach with ResNet50 architecture is implemented. The model is built with Imagenet weights(pretrained and helps the model converge in less epochs) on different dense layers with 'relu' activation function and compiled with 'adam' optimizer, loss function 'categorical crossentropy' on 'accuracy' evaluation metrics.

## Output:
The model is built, trained and tested on the CIFAR-10 dataset with ResNet50 Architecture with an input shape (32,32,3), batch size of 128 samples, learning rate of the model 0.001 and 40 epochs. The errors are easily inspected by plotting a confusion matrix. The output images are displayed with both predicted and true labels for an easy understanding of the model's performance.

### Input Images

<img width="342" alt="Sample Input" src="https://user-images.githubusercontent.com/122699563/213935829-83201980-921f-4256-930a-bc40b8a96955.png">

### Confusion Matrix

<img width="852" alt="Image_Classifier_confmatrix" src="https://user-images.githubusercontent.com/122699563/213935734-733c6cfc-61d3-486a-bef3-ef0166b3c876.png">

### Correct Predictions

<img width="409" alt="Correct1" src="https://user-images.githubusercontent.com/122699563/213936318-c1b9013e-753d-4f32-b6e3-8b119750a5ae.png">
<img width="226" alt="Correct2" src="https://user-images.githubusercontent.com/122699563/213936333-826e0486-8f17-412e-8186-fb3eb3e4fd83.png">

### Incorrect Predictions

<img width="720" alt="Sample Predictions" src="https://user-images.githubusercontent.com/122699563/213935841-6a11128c-630e-413e-89c4-c1dedec63abf.png">

### Model Accuracy and Loss

<img width="924" alt="Model Accuracy and Loss" src="https://user-images.githubusercontent.com/122699563/213936410-47ff8a6c-2366-4034-b7e9-b00c50ff1e36.png">




 
