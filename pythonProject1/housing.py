import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #학습데이터 나누기
import matplotlib.pyplot as plt  #그래프
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 데이터 읽기
header = ['CRIM', 'ZN', 'INDU', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/4.hr.csv', delim_whitespace=True, names=header)

array = data.values
X = array[:, 0:13]  # 모든 행과 마지막 열을 제외한 열 선택
y = array[:, 13]  # 모든 행과 마지막 열 선택

# 학습데이터. 테스트 데이터
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train:shape, X_test:shape , y_train, y_test)

#학습
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
#rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(mae)

# 시각화: 모든 변수와 주택 가격(MEDV)의 관계
plt.scatter(range(len(X_test[:15]), y_test[:15], color='blue')
plt.scatter(range(len(X_test[:15]), y_pred[:15], color='red')
plt.xlabel('index')
plt.ylabel('MEDV')
plt.show()

KFold= KFold(n_splits=5)
mse= cross_val_score(model, X, Y, scoring= )