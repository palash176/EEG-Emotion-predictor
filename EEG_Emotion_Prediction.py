import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

import cv2
import sys

data = pd.read_csv('Before_Writing.csv')

label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}


def preprocess_inputs(df):
    df = df.copy()
    
    df['label'] = df['label'].replace(label_mapping)
    
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)

inputs = tf.keras.Input(shape=(X_train.shape[1],))

expand_dims = tf.expand_dims(inputs, axis=2)

gru = tf.keras.layers.GRU(256, return_sequences=True)(expand_dims)

flatten = tf.keras.layers.Flatten()(gru)

outputs = tf.keras.layers.Dense(3, activation='softmax')(flatten)


model = tf.keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Before Writing Test Accuracy: {:.3f}%".format(model_acc * 100))



y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(X_test))))

cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred, target_names=label_mapping.keys())

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xticks(np.arange(3) + 0.5, label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Before Writing Confusion Matrix")
# plt.show()

plt.savefig('Before Writing Confusion Matrix.png')


####################### After # Writing ##################################

aw_data = pd.read_csv('After_Writing.csv')


aw_label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

def aw_preprocess_inputs(df):
    df = df.copy()
    
    df['label'] = df['label'].replace(aw_label_mapping)
    
    aw_y = df['label'].copy()
    aw_X = df.drop('label', axis=1).copy()
    
    aw_X_train, aw_X_test, aw_y_train, aw_y_test = train_test_split(aw_X, aw_y, train_size=0.7, random_state=123)
    
    return aw_X_train, aw_X_test, aw_y_train, aw_y_test

aw_X_train, aw_X_test, aw_y_train, aw_y_test = aw_preprocess_inputs(aw_data)

aw_inputs = tf.keras.Input(shape=(aw_X_train.shape[1],))

aw_expand_dims = tf.expand_dims(aw_inputs, axis=2)

aw_gru = tf.keras.layers.GRU(256, return_sequences=True)(aw_expand_dims)

aw_flatten = tf.keras.layers.Flatten()(aw_gru)

aw_outputs = tf.keras.layers.Dense(3, activation='softmax')(aw_flatten)


aw_model = tf.keras.Model(inputs=aw_inputs, outputs=aw_outputs)


aw_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

aw_history = aw_model.fit(
    aw_X_train,
    aw_y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

aw_model_acc = aw_model.evaluate(aw_X_test, aw_y_test, verbose=0)[1]
print("After Writing Test Accuracy: {:.3f}%".format(aw_model_acc * 100))



aw_y_pred = np.array(list(map(lambda x: np.argmax(x), aw_model.predict(aw_X_test))))

aw_cm = confusion_matrix(aw_y_test, aw_y_pred)
aw_clr = classification_report(aw_y_test, aw_y_pred, target_names=aw_label_mapping.keys())

plt.figure(figsize=(8, 8))
sns.heatmap(aw_cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xticks(np.arange(3) + 0.5, aw_label_mapping.keys())
plt.yticks(np.arange(3) + 0.5, aw_label_mapping.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("After Writing Confusion Matrix")
# plt.show()

plt.savefig('After Writing Confusion Matrix.png')

img = cv2.imread('Before Writing Confusion Matrix.png', cv2.IMREAD_ANYCOLOR)

aw_img = cv2.imread('After Writing Confusion Matrix.png', cv2.IMREAD_ANYCOLOR)

while True:
    cv2.imshow("Before Writing Confusion Matrix", img)
    cv2.waitKey(0)
    cv2.imshow("After Writing Confusion Matrix", aw_img)
    cv2.waitKey(0)
    sys.exit() # to exit from all the processes
cv2.destroyAllWindows()


#print("Classification Report:\n----------------------\n", clr)