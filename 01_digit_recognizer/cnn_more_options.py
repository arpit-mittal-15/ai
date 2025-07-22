# accuracy ~98.8%

import pandas as pd
import numpy as np
import tensorflow as tf# type: ignore

# Load and split
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = df.iloc[:10000]
train_df = df.iloc[10000:]

x_train = train_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_df["label"].values

x_test = test_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_test = test_df["label"].values

# ✅ Best-performing simple CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.1, shuffle=True)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")
