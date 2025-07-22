# accuracy > 97%

import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Load data from Kaggle Digit Recognizer CSV
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

# Extract labels and pixel values
labels = df['label'].values
images = df.drop(columns=['label']).values

# Normalize pixel values to [0, 1]
images = images / 255.0

# Split: first 10,000 rows for testing, rest for training
x_test, y_test = images[:10000], labels[:10000]
x_train, y_train = images[10000:], labels[10000:]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate on the test set (first 10,000 rows)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy on first 10,000 rows: {test_accuracy * 100:.2f}%")
