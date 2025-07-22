# accuracy ~98.5%

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load Kaggle training data
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

# Split manually: first 10,000 for testing, rest for training
test_df = df.iloc[:10000]
train_df = df.iloc[10000:]

# Prepare training data
x_train = train_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_df["label"].values

# Prepare internal test data (used for evaluation)
x_test = test_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_test = test_df["label"].values

# Define CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Evaluate on internal test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy on first 10,000 samples: {test_accuracy * 100:.2f}%")

# ✅ Prepare Kaggle submission
# Load test.csv (this is the file we need to predict on)
submission_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
x_submission = submission_test.values.reshape(-1, 28, 28, 1) / 255.0

# Predict using the trained model
predictions = model.predict(x_submission)
predicted_labels = np.argmax(predictions, axis=1)

# Create submission file
submission_df = pd.DataFrame({
    "ImageId": np.arange(1, len(predicted_labels) + 1),
    "Label": predicted_labels
})

submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv is ready!")
