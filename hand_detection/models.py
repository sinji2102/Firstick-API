import pickle
import pandas as pd

# 사이킷런
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# csv 파일 가져오기
df = pd.read_csv('./data/chopsticks.csv')

X = df.drop('class', axis=1)
y = df['class']

###############################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=1000)

# print
print('Train shape:')
print(X_train.shape)
print(X_test.shape)
print('-------------------')
print('Test Shape:')
print(y_train.shape)
print(y_test.shape)

# trainning result
'''Train shape:
(1501, 67)
(2253, 67)
-------------------
Test Shape:
(1501,)
(2253,)
0.9968930315135375'''

###############################


# 랜덤 포레스트 분류
rc = RandomForestClassifier()
model = rc.fit(X_train, y_train)
y_pred = model.predict(X_test)


print(accuracy_score(y_test, y_pred))

################################



# 사이킷런 모델 export -> 추후 tf model 형식으로 변환해야함
with open('./data/chopsticks.pkl', 'wb') as f:
    pickle.dump(model, f)