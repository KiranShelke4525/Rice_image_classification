# Rice Grain Image Classification using CNN & Transfer Learning

1) # Install all required libraries:
# pip install tensorflow numpy matplotlib scikit-learn seaborn gradio

A custom CNN model, and Transfer Learning with MobileNetV2.

# select tool  : Built and tested entirely in Jupyter Notebook. 
# Dataset was downloaded locally — please update your own dataset path.



##### Flow Chart #####

+-------------------+
|  Load Dataset     |
+--------+----------+
         |
         v
+-------------------+
| Preprocess Images |
+--------+----------+
         |
         v
+--------+---------+        +---------------------------+
| Custom CNN Model |        | MobileNetV2 Transfer Learn|
+--------+---------+        +---------------------------+
         |                             |
         +-------------+---------------+
                       v
             +------------------+
             | Evaluate Results |
             +------------------+
                       |
                       v
             +----------------------+
             | Gradio Web Interface |
             +----------------------+

### Psudocode ####

# 1. Import all necessary libraries
import tensorflow as tf, numpy as np, gradio as gr

# 2. Load and preprocess dataset using ImageDataGenerator
- Define dataset_path (local path)
- Use flow_from_directory with target size (224, 224)

# 3. Build CNN Model
- Add Conv2D, MaxPooling2D, Flatten, Dense layers
- Compile using Adam and categorical_crossentropy
- Train and validate model

# 4. Evaluate CNN
- Use classification_report and confusion_matrix

# 5. Build MobileNetV2 Transfer Learning Model
- Load base_model from keras.applications
- Freeze base layers
- Add custom top classifier
- Compile and train

# 6. Evaluate Transfer Learning Model
- Generate predictions and report metrics

# 7. Create Gradio interface
- Define function to preprocess input image and predict
- Launch app using interface.launch(share=True)


             



