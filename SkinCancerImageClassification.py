# Author: Rifat Rahman Hridhoy
# ID: 10402885

"""
Skin cancer lesion classification using the HAM10000 dataset

Dataset link:
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000

The dataset contains 7 classes of skin cancer lesions:
- Melanocytic nevi (nv)
- Melanoma (mel)
- Benign keratosis-like lesions (bkl)
- Basal cell carcinoma (bcc) 
- Actinic keratoses (akiec)
- Vascular lesions (vas)
- Dermatofibroma (df)
"""

# ----------------- Import Libraries -----------------
import matplotlib.pyplot as plt   # For plotting graphs/images
import numpy as np                # For numerical operations
import pandas as pd               # For handling CSV dataset
import os                         # For file path operations

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import keras                      # Deep learning library
import tensorflow as tf           # Backend for Keras
from glob import glob             # For file searching
import seaborn as sns             # Advanced data visualization
from PIL import Image             # For image processing

np.random.seed(42)  # Fix random seed for reproducibility

from sklearn.metrics import confusion_matrix  # For model evaluation
from tensorflow.keras.utils import to_categorical  # Converts labels to one-hot encoding
from keras.models import Sequential           # Sequential model type
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D  # Layers for CNN
from sklearn.model_selection import train_test_split  # Train-test split
from sklearn.preprocessing import LabelEncoder       # Converts text labels to numbers
from sklearn.utils import resample                   # For balancing dataset

# ----------------- Load Dataset -----------------
skin_df = pd.read_csv(r"C:\Users\Rahma\Downloads\archive\HAM10000_metadata.csv")  # Load metadata

SIZE = 32  # Resize images to 32x32

# ----------------- Encode Labels -----------------
le = LabelEncoder()                 # Initialize encoder
le.fit(skin_df['dx'])               # Fit on lesion type column
print(list(le.classes_))            # Print mapping (nv→0, mel→1, etc.)
skin_df['label'] = le.transform(skin_df["dx"])  # Create new numeric label column
print(skin_df.sample(10))           # Print 10 random rows

# ----------------- Data Visualization -----------------
fig = plt.figure(figsize=(12, 8))   # Create figure

# Lesion type distribution
ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type')

# Sex distribution
ax2 = fig.add_subplot(222)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count')
ax2.set_title('Sex')

# Localization (body part) distribution
ax3 = fig.add_subplot(223)
skin_df['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count')
ax3.set_title('Localization')

# Age distribution
ax4 = fig.add_subplot(224)
sample_age = skin_df[pd.notnull(skin_df['age'])]  # Drop rows with missing ages
sns.displot(sample_age['age'], kde=True, color='red', stat="density")
ax4.set_title('Age')

plt.tight_layout()
plt.show()

# ----------------- Balance Dataset -----------------
print(skin_df['label'].value_counts())  # Show imbalance before resampling

# Separate data by class
df_0 = skin_df[skin_df['label'] == 0]
df_1 = skin_df[skin_df['label'] == 1]
df_2 = skin_df[skin_df['label'] == 2]
df_3 = skin_df[skin_df['label'] == 3]
df_4 = skin_df[skin_df['label'] == 4]
df_5 = skin_df[skin_df['label'] == 5]
df_6 = skin_df[skin_df['label'] == 6]

# Resample each class to 500 samples
n_samples = 500
df_0_balanced = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
df_1_balanced = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
df_2_balanced = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=n_samples, random_state=42)

# Combine all balanced subsets
skin_df_balanced = pd.concat([df_0_balanced, df_1_balanced, df_2_balanced,
                              df_3_balanced, df_4_balanced, df_5_balanced, df_6_balanced])
print(skin_df_balanced['label'].value_counts())  # Verify balanced dataset

# ----------------- Load Images -----------------
# Map image_id → file path
image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join(r"C:\Users\Rahma\Downloads\HAM10000\HAM10000_images_part_*/*.jpg"))}

# Add path column
skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)

# Load and resize images
images = []
for path in skin_df_balanced['path']:
    if os.path.exists(path):
        try:
            img = Image.open(path).resize((SIZE, SIZE))  # Resize to 32x32
            images.append(np.asarray(img))               # Convert to array
        except Exception as e:
            print(f"Error loading {path}: {e}")
            images.append(None)
    else:
        print(f"File not found: {path}")
        images.append(None)

skin_df_balanced['image'] = images  # Add images column

# ----------------- Sample Visualization -----------------
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize=(4 * n_samples, 3 * 7))

# Show 5 random images from each class
for n_axs, (type_name, type_rows) in zip(m_axs,
                                         skin_df_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')

# ----------------- Data Preparation -----------------
X = np.asarray(skin_df_balanced['image'].tolist())  # Convert list to NumPy array
X = X / 255.0                                       # Normalize to [0,1]
Y = skin_df_balanced['label']                       # Labels
Y_cat = to_categorical(Y, num_classes=7)            # One-hot encoding

# Train-test split (75/25)
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=42)

# ----------------- CNN Model -----------------
model = Sequential()

# Conv layer 1
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Conv layer 2
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Conv layer 3
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Flatten + Fully connected layers
model.add(Flatten())
model.add(Dense(32))                          # Hidden dense layer
model.add(Dense(7, activation='softmax'))     # Output layer (7 classes)

model.summary()  # Print model summary

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

# ----------------- Train Model -----------------
batch_size = 16
epochs = 50

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

# Evaluate model
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])

# ----------------- Training Curves -----------------
# Loss curves
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy curves
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ----------------- Model Evaluation -----------------
# Predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities → class index
y_true = np.argmax(y_test, axis=1)          # Convert one-hot → class index

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize=(6, 6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

# Incorrect prediction fraction
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.bar(np.arange(7), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.show()
