# 🧠📹 FaceVision - Real-Time Face Detection From Video Streams 👁️🖥️

## 📓 [Final Notebook With Results](https://www.kaggle.com/code/kartikparatkar/facevision-real-timefacedetectionfromvideostreams?scriptVersionId=234191458)

🎯 **Aim** - To build a Deep Learning model which processes video as input and draws a bounding box around the face regardless of the location of the face in the video.

📸 **Data Collection** - We require a high amount of data to train the neural network deep learning model. We captured a total of **387 images** from the laptop camera.

🏷️ **Data Setting & Labelling of Images** - This image dataset was uploaded to **Kaggle**.  
We created a `"labels"` directory to store the labels of each image.  
For labeling the images, we used the **"labelimg"** inbuilt library.  
Followed this [Github link](https://github.com/HumanSignal/labelImg) and [Video](https://www.youtube.com/watch?v=fjynQ9P2C08).  
Basically, when we run the labeling tool, it allows us to label the images with a square bounding box around the face which stores different information about the face (object) in `.xml` format.  
We converted `.xml` formatted labels into `.json` formatted labels because we will be using **Albumentations** for data augmentation.  

🗂️ We split the dataset as follows:  
- **70%** (268 images) → Training  
- **15%** (57 images) → Test  
- **15%** (62 images) → Validation

🧪 **Applying Image Augmentation on Images and Labels using Albumentations** -  
Image Augmentation is the process of creating new training images by applying random transformations to existing images.  
**Albumentations** is a Python library for fast and flexible augmentation.  
It’s especially useful in computer vision projects.  
Albumentations is a fast, flexible image augmentation library for deep learning, offering rich transformations (flip, rotate, noise, etc.) that enhance model generalization, especially in computer vision tasks like detection and segmentation.

🧠 **Building Deep Neural Network Model Using Keras Functional API** -  
Keras Functional APIs offer a way to build more flexible and complex neural network architecture compared to the simpler Sequential API.  
Here, API means **Application Programming Interface** provided by Keras.  
We used a **pretrained VGG16 model** which is pretrained on a large dataset called **ImageNet**.  
We **froze the classification layer** of VGG16 and used only its **convolutional layer**.  
To that output, we added our **custom classification and regression model**.  

We have 2 problems to solve here:
1. 🧮 **Regression** → Identifying the coordinates of the bounding box  
2. 🧪 **Classification** → Detecting whether a face is available in the video or not  

## 🧊 VGG16 Model Architecture

![VGG16 Model Architecture](https://github.com/KARTIKPARATKAR/FaceVision-Real-Time-Face-Detection-From-Video-Streams/blob/main/VGG16_Model.jpg)

🛠️ **Building Custom Convolutional Neural Network Model** -  
- Input layer with shape (120,120,3) as our input image size is 120x120  
- Passed input layer through VGG16 ImageNet pre-trained model with frozen classification layer  
- Defining 2 different models, one for classification (**F1**) and one for regression (**F2**)  
- VGG16 output is passed through both F1 and F2  
- F1 output is passed through `class1` dense layer with 2048 nodes and relu activation  
- F2 output is passed through `regress1` dense layer with 2048 nodes and relu activation  
- `class1` output is passed through `class2` layer → output for classification model  
- `regress1` output is passed through `regress2` layer → output for regression model  
- At the end, we combine the output of both classification and regression models

## 🧠 Custome_CNN_Model_Architecture

![Custome_CNN_Model_Architecture](https://github.com/KARTIKPARATKAR/FaceVision-Real-Time-Face-Detection-From-Video-Streams/blob/main/facetracker_model.png)

⚙️ **Loss Function and Training of the Model** -  
For the classification problem, we used the loss function **binary crossentropy**.  
We defined **localization loss** explicitly for predicting where an object is located within an image.  
Predicting bounding boxes is not a classification task, so it requires a different loss function than cross-entropy.  
Localization loss is a custom loss function that penalizes the model based on how far off its predicted bounding box is from the ground truth.  

We defined a custom keras model class **`FaceTracker`**, designed for a multi-task learning problem of face detection where we perform both classification and regression.

### Key Components:
- `FaceTracker` class → Combines classification and regression in a custom model  
- `train_step()` → Custom training logic: forward pass, compute losses, backpropagation  
- `test_step()` → Custom testing logic: forward pass and loss calculation  
- `compile()` → Adds support for custom losses (classification + localization)  
- `call()` → Defines model behavior when it is called  

We created an instance of the custom FaceTracker class by passing the FaceTracker model to it.  
We compiled the model with:
- **Adam optimizer** with learning rate and decay  
- **Binary crossentropy** for classification  
- **Localization loss** for regression  

We trained the custom FaceTracker model using `.fit()` method for **10 epochs** with `"val"` as validation loss and **callback**.

📊 **Output Plots & Testing Model on Test Data** -  
We drew 3 plots after training the deep learning model:
- 📉 Total loss vs epochs  
- 🧠 Classification loss vs epochs  
- 📏 Regression loss vs epochs  

🎥 Then we visualized the bounding box on test data.  
We captured a **video of nearly 1 min** and saved it on the local machine.  
Uploaded this video to the **Kaggle dataset directory**.  
We passed this video frame through the model and detected faces every **0.3 seconds**.  
Bounding boxes were drawn around detected faces.  
🖼️ Displayed sampled frames with annotations (i.e., the predicted bounding boxes and labels drawn on the video frames).

Checkout handwritten detailed notes of this project [Here]()
