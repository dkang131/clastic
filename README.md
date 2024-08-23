# Clastic: Classify Your Plastic

![CLASTIC](https://github.com/dkang131/clastic/assets/93761533/f62451e1-7ef2-4aac-92b2-9bb4f91e0f80)

## About Clastic
Clastic is a sustainable living mobile application that is used to classify plastic based on its types and can be exchanged with e-money.

This project is one of the Bangkit Academy 2023 Batch 1 Product Capstone Project

The author, [Darren Kang Wan Chee](https://www.linkedin.com/in/darren-kang-wan-chee/), with his partner [Irene Patricia Wibowo](https://www.linkedin.com/in/irene-patricia-w/) is the one who responsible for building the machine learning classifier of Clastic. 

More details can be seen below
## What you need
1. Code Platform: Google Collaboration and VSCode
2. Programming Language: Python
3. Library: TensorFlow, NumPy, Pillow, Matplotlib, OS, zipfile, CV2, h5py

## Dataset
The dataset can be accessed via [dataset](https://github.com/dkang131/clastic/tree/main/DATASET)

## Steps for Training
1. Import data into the Code Platform
2. Resize the image to the desired size, we resize it to 224x224
3. Define the pre-trained model as the base model, we use a pre-trained model of MobileNetV2
```python
#Load the pre-trained MobileNet model without the top (classification) layers:
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Freeze the base model's layers to prevent them from being trained:
base_model.trainable = False
```
4. Add more dense layers, add dropout layers, add L2 regularizers to the dense layers
5. Define loss function and optimizer.
```python
base_learning_rate = 0.0001
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])
```
6. Train the model
7. After we obtain the desired accuracy, we can convert the model into an H5 file
```python
!pip install h5py
model.save('best_model.h5')
```

For the complete code please check our [notebook](https://github.com/dkang131/clastic/blob/main/CLASTIC_CLASSIFIER_MACHINE_LEARNING_MODEL.ipynb)
