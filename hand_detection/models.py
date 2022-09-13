import pickle
import pandas as pd
import numpy as np

# 텐서플로우
import tensorflow as tf

# 사이킷런
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# csv 파일 가져오기
df = pd.read_csv('./data/chopsticks.csv')

X = df.drop('class', axis=1)
y = df['class']

###############################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=500)

# print
print('Train shape:')
print(X_train.shape)
print(X_test.shape)
print('-------------------')
print('Test Shape:')
print(y_train.shape)
print(y_test.shape)

# train shape result
'''Train shape:
(2627, 67)
(1127, 67)
-------------------
Test Shape:
(2627,)
(1127,)
'''

###############################
# Sklearn training

# 랜덤 포레스트 분류
rc = RandomForestClassifier()
skl_model = rc.fit(X_train, y_train)
y_pred = skl_model.predict(X_test)


print(accuracy_score(y_test, y_pred))

# Export sklearn model
with open('./data/chopsticks.pkl', 'wb') as f:
    pickle.dump(skl_model, f)

################################

# lr = LogisticRegression()
# skl_model = lr.fit(X_train, y_train)
# y_pred = skl_model.predict(X_test)
# print(accuracy_score(y_test, y_pred))

################################

# Transform tensorflow model
# create a TF model with the same architecture
tf_model = tf.keras.models.Sequential()
tf_model.add(tf.keras.Input(shape=(67,)))
tf_model.add(tf.keras.layers.Dense(1))

# assign the parameters from sklearn to the TF model
tf_model.layers[0].weights[0].assign(skl_model.coef_.transpose())
tf_model.layers[0].bias.assign(skl_model.intercept_)

# # Convert tf model to tflite model
tfl_converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tfl_converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = tfl_converter.convert()

# # Export tflite model
open('./data/chopsticks.tflite', 'wb').write(tflite_quantized_model)