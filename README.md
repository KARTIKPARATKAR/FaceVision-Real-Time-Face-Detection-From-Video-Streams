# 🤖 FaceVision – Real-Time Face Detection from Video Streams 🎥

## 📌 [Final Notebook with Results on Kaggle](https://www.kaggle.com/code/kartikparatkar/facevision-real-timefacedetectionfromvideostreams?scriptVersionId=234191458)

---

## 🎯 Aim  
To build a Deep Learning model that processes **video as input** and draws a **bounding box** 🟥 around the face(s), regardless of their position in the video.

---

## 📸 Data Collection  
We collected **387 images** using a laptop webcam 💻 to train the deep learning model effectively.

---

## 🗂️ Data Setup & Annotation  
- Uploaded the images to **Kaggle** 📁  
- Created a `labels/` directory for storing label files  
- Used [`labelImg`](https://github.com/HumanSignal/labelImg) 🏷️ for annotating images (bounding boxes around faces)  
- Output format was **XML**, later converted to **JSON** for compatibility with Albumentations  
- Dataset Split:
  - 🔹 70% → Training (268 images)
  - 🔹 15% → Testing (57 images)
  - 🔹 15% → Validation (62 images)

📽️ [Labeling Tutorial](https://www.youtube.com/watch?v=fjynQ9P2C08)

---

## 🔄 Data Augmentation using Albumentations  
**Albumentations** is a powerful and fast image augmentation library used for:
- Rotation 🔁
- Flipping ↔️
- Blurring 🌫️
- Adding noise 🎚️  
These augmentations improve the model’s ability to generalize in real-world scenarios.

---

## 🧠 Model Building with Keras Functional API  
Using **Keras Functional API**, we:
- Load **VGG16** pre-trained on **ImageNet**
- Freeze the classification head ❄️
- Add two custom heads:
  - 🟢 Classification – detects presence of a face
  - 🔴 Regression – predicts bounding box coordinates

---

## 🧱 VGG16 Model Architecture

![VGG16 Model Architecture](https://github.com/KARTIKPARATKAR/FaceVision-Real-Time-Face-Detection-From-Video-Streams/blob/main/VGG16_Model.jpg)

---

## 🏗️ Custom CNN Architecture Overview  
Here's how the custom CNN model is built:
- 📥 Input: `(120, 120, 3)`
- 🔄 Pass through VGG16 base
- 🔁 Two branches:
  - `F1` ➡️ Classification Head
    - Dense Layer → `2048` nodes + ReLU
    - Output: `class2`
  - `F2` ➡️ Regression Head
    - Dense Layer → `2048` nodes + ReLU
    - Output: `regress2`
- 🔗 Merge both outputs into a unified model

## 🧬 Custom CNN Model Diagram

![Custom CNN Model](https://github.com/KARTIKPARATKAR/FaceVision-Real-Time-Face-Detection-From-Video-Streams/blob/main/facetracker_model.png)

---

## ⚙️ Training Strategy & Loss Functions  
- **Classification Loss**: `Binary Crossentropy`
- **Regression Loss**: Custom **Localization Loss** 🧮 – penalizes based on deviation from ground-truth bounding boxes

Implemented a custom `FaceTracker` class in Keras:
- `train_step()` – Forward pass, compute losses, backpropagation 🔁
- `test_step()` – Forward pass, compute test loss 🧪
- `compile()` – Combines classification and localization loss 🧩
- `call()` – Defines model behavior on execution 📞

Used **Adam optimizer** with learning rate decay and trained the model using `.fit()` for **10 epochs** 🏋️‍♂️

---

## 📊 Training Metrics & Evaluation  
Generated the following plots:
- 📉 Total Loss vs Epochs
- 📉 Classification Loss vs Epochs
- 📉 Regression Loss vs Epochs

---

## 🧪 Real-Time Testing with Video 🎥  
- Captured a **1-minute video** using a webcam 🎦  
- Uploaded the video to Kaggle directory  
- Extracted frames every **0.3 seconds** ⏱️  
- Passed each frame through the model  
- Drew bounding boxes 🟥 around detected faces  
- Displayed sampled frames with **annotations** (bounding boxes + class labels)

> ✅ *Annotations* here refer to **visual cues** like bounding boxes drawn over images, indicating model predictions.

---

Feel free to ⭐ this repo if you found it useful!
