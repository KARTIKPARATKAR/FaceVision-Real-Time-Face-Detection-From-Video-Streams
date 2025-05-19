# FaceVision-Real-Time-Face-Detection-From-Video-Streams

## [Final NoteBook With Results](https://www.kaggle.com/code/kartikparatkar/facevision-real-timefacedetectionfromvideostreams?scriptVersionId=234191458)

Aim - To built a Deep Learning model which processes video as input and draw a bounding box around face regardless of the location of face in video.

Data Collection - We require high amount of data to train the neural network deep learning model. We captured a total of 387 images from laptop camera.

Data Setting & Labelling of Images - This image dataset was uploaded to kaggle.We created "labels" directory to store the labels of each image.For labeling the images , we used "labelimg" inbuild library. Followed this [Github link](https://github.com/HumanSignal/labelImg) and [Video](https://www.youtube.com/watch?v=fjynQ9P2C08) . Basically when we run the labelling tool , it allows us to label the images with square bounding box around face which stores different information about face(object) in xml format. We converted xml formatted labells into json formatted labels becuase we will be using Albumentation for data augmentation.We splitted the dataset into 70% in training and 15% each in test and validation data. So out of 387 originally captured images , 268 will go for training , 57 in test and 9 in validation.

Applying Image Augmentation on Images and Labels using Albumenation - Image Augmentation is the process of creating new training images by applying random transformations to existing images. Albumenation is a python ibrary for fast and flexible augmentation , this is especially useful in computer vision projects.Albumentations is a fast, flexible image augmentation library for deep learning, offering rich transformations (flip, rotate, noise, etc.) that enhance model generalization, especially in computer vision tasks like detection and segmentation.

Building Deep Neural Network Model Using Keras Functional API's - Keras functional API's is a way to build more flexible and complex neural network architecture compared to the simpler sequential API . Here API means Application Programming INterface provided by Keras. We will use pretrained model of VGG16 which is pretrained on large dataset called "Imagenet".We will freeze the classification layer of VGG16 and use only convolutional layer of VGG16 and we will add our custome classification and regression model to the output of VGG16 convolutional layer.We have 2 problems to solve here , one is regression -> Identifying the coordinates of the bounding box and another one is classification -> Detecting whether face is available in the video or not. 


## VGG16 Model Architecture

![VGG16 Model Architecture](https://github.com/KARTIKPARATKAR/FaceVision-Real-Time-Face-Detection-From-Video-Streams/blob/main/VGG16_Model.jpg)

Building Custome Convolutional Neural Network Model - 
   - Input layer with shape (120,120,3) as our input image size is 120*120
   - Passed input layer through VGG16 imagenet pre-trained model with freezed classification layer.
   - Defining 2 different models , one for classification(F1) and one for regression(F2).
   - VGG16 output is passed through both F1 and F2.
   - F1 output is passed through the class1 dense layer with 2048 nodes and relu activation function.
   - F2 output is passed through the regress1 dense layer with 2048 nodes and relu activation function.
   - class1 output is passed through class2 layer which is also the output output for classification model.
   - regress1 output is passed through regress2 layer which is also the output for regression model.
   - At the end , we will combine the output of both classification and regression models.

     ![Custome_CNN_Model_Architecture](https://github.com/KARTIKPARATKAR/FaceVision-Real-Time-Face-Detection-From-Video-Streams/blob/main/facetracker_model.png)




