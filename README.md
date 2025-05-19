# 🎯 FaceVision - Real-Time Face Detection From Video Streams

## [Final Notebook with Results](https://www.kaggle.com/code/kartikparatkar/facevision-real-timefacedetectionfromvideostreams?scriptVersionId=234191458)

**Aim** – To build a deep learning model that processes video as input and draws bounding boxes around faces, regardless of their position in the video.

**Data Collection** – We require a significant amount of data to train the deep learning model. A total of **387 images** were captured using a laptop webcam.

**Data Setting & Labeling of Images** – The image dataset was uploaded to Kaggle. A `labels` directory was created to store the label files for each image. For labeling, we used the `labelImg` tool. Refer to this [GitHub link](https://github.com/HumanSignal/labelImg) and [YouTube video](https://www.youtube.com/watch?v=fjynQ9P2C08). This tool allows us to draw square bounding boxes around faces, saving information in XML format. These XML labels were converted to JSON format to be compatible with the **Albumentations** library for data augmentation.

We split the dataset as follows:
- 70% Training – 268 images  
- 15% Testing – 57 images  
- 15% Validation – 62 images

**Applying Image Augmentation using Albumentations** – Image augmentation is the process of creating new training samples by applying random transformations to existing images. **Albumentations** is a fast, flexible Python library especially suited for computer vision tasks. It includes rich transformations like flip, rotate, blur, noise, and more, which help improve model generalization.

**Building the Deep Neural Network Model Using Keras Functional API** – The Keras Functional API allows us to build more complex and flexible architectures compared to the Sequential API. We use a **pre-trained VGG16 model** (trained on ImageNet), freeze its classification layers, and add custom classification and regression heads on top of its convolutional base.

This model tackles two tasks:
- **Regression** – Predicting the coordinates of the bounding box
- **Classification** – Detecting whether a face is present in the frame

## VGG16 Model Architecture

![VGG16 Model Architecture](https://github.com/KARTIKPARATKAR/FaceVision-Real-Time-Face-Detection-From-Video-Streams/blob/main/VGG16_Model.jpg)

**Building the Custom Convolutional Neural Network Model** –
- Input layer with shape (120, 120, 3), matching our input image size
- Passed through the VGG16 pre-trained model with frozen classification layers
- Two separate heads:
  - One for classification (F1)
  - One for regression (F2)
- VGG16 output is passed to both F1 and F2
- F1 is connected to a `class1` dense layer (2048 nodes, ReLU activation)
- F2 is connected to a `regress1` dense layer (2048 nodes, ReLU activation)
- `class1` connects to `class2` which is the output of the classification head
- `regress1` connects to `regress2` which is the output of the regression head
- Outputs of both heads are combined

## Custome_CNN_Model_Architecture

![Custome_CNN_Model_Architecture](https://github.com/KARTIKPARATKAR/FaceVision-Real-Time-Face-Detection-From-Video-Streams/blob/main/facetracker_model.png)

**Loss Function and Training the Model** –  
- For the classification task, we use **Binary Crossentropy**
- For bounding box prediction (regression), we define a **custom localization loss** that penalizes the model based on the distance between predicted and actual bounding boxes

We create a custom Keras model class called `FaceTracker`, designed for this multi-task learning setup:

- `FaceTracker` – Combines classification and regression into a single model
- `train_step()` – Custom training logic (forward pass, compute loss, backpropagation)
- `test_step()` – Custom test logic (forward pass, loss computation)
- `compile()` – Adds support for both loss functions
- `call()` – Defines model behavior on execution

We then create an instance of the `FaceTracker` class and compile it using the **Adam optimizer** with learning rate decay, binary crossentropy for classification, and localization loss for bounding box regression.

The model is trained using `.fit()` for **10 epochs**, using validation loss and callbacks.

**Output Plots & Testing on Test Data** –
After training, we plotted:
- Total Loss vs Epochs
- Classification Loss vs Epochs
- Regression Loss vs Epochs

We then visualized model predictions by drawing bounding boxes on test data images.

We also captured a **1-minute video** and saved it locally. This video was uploaded to the Kaggle dataset directory. Frames were extracted every **0.3 seconds**, passed through the model, and bounding boxes were drawn around detected faces. We displayed the sampled frames with annotations (i.e., visualized bounding boxes and labels predicted by the model).
