# accuracy ~98.9%

import pandas as pd
import numpy as np
import tensorflow as tf# type: ignore

# Load and split dataset
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = df.iloc[:10000]
train_df = df.iloc[10000:]

x_train = train_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_df["label"].values

x_test = test_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_test = test_df["label"].values

# ✅ Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# ✅ Enhanced Model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
    data_augmentation,
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train longer with larger batch
model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.1)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"✅ Test Accuracy on first 10,000 samples: {test_accuracy * 100:.2f}%")

# Submission
submission_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
x_submission = submission_data.values.reshape(-1, 28, 28, 1) / 255.0

predictions = model.predict(x_submission)
predicted_labels = np.argmax(predictions, axis=1)

submission_df = pd.DataFrame({
    "ImageId": np.arange(1, len(predicted_labels) + 1),
    "Label": predicted_labels
})
submission_df.to_csv("submission.csv", index=False)