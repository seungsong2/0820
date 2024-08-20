import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 읽기  delim_whitespace= True 는 공백을 읽는 함수
header = ['CRIM', 'ZN', 'INDU', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/4.hr.csv', delim_whitespace= True, names=header)

# 데이터 요약
print(data.describe())
print(data.info())

# 시각화: 모든 변수와 주택 가격(MEDV)의 관계
plt.figure(figsize=(16, 12))
for i, col in enumerate(data.columns[:-1]):
    plt.subplot(4, 4, i + 1)
    plt.scatter(data[col], data['MEDV'], alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    plt.title(f'{col} vs MEDV')

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 특성과 타겟 변수 설정
X = data.drop(columns=['MEDV'])
y = data['MEDV']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# K-fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=0)  # K=5로 설정

mse_scores = []

# K-fold 교차 검증 실행
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

# 평균 MSE 출력
average_mse = np.mean(mse_scores)
print(f'Average MSE from K-fold Cross-Validation: {average_mse:.2f}')

# 예측과 실제 값 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted MEDV')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()
