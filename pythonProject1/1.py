import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 데이터 읽기
data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)
print(data.columns)
# 데이터 전처리
data['Height_cm'] = data['Height(Inches)'] * 2.54
data['Weight_kg'] = data['Weight(Pounds)'] * 0.453592

# 데이터에서 숫자만 가져오도록
array=data.values

# 특성과 타겟 변수 설정
X = array[:,1]
y = array[:,1]
X = X.reshape(-1,1)
# 독립변수가 엑셀 들어가는 2차원이상의 변수는 reshape 안필요

# 데이터 요약
#print(data.describe())
#print(data.info())

# 시각화: Height_cm과 Weight_kg의 산점도
#plt.figure(figsize=(10, 6))
plt.scatter(data['Height_cm'], data['Weight_kg'], alpha=0.5)
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight')
plt.show()

# 특성과 타겟 변수 설정
X = data[['Height_cm']]
y = data['Weight_kg']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 스케일링
#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
#rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(mae)

#print(f'MSE: {mse:.2f}')
#print(f'RMSE: {rmse:.2f}')
#print(f'MAE: {mae:.2f}')

# 시각화: 실제 값과 예측 값 비교
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:100], y_test[:100], color='blue', label='Actual Values')
plt.scatter(X_test[:100], y_pred[:100], color='red', label='Predicted Values', alpha=0.5)
plt.plot(X_test[:100], y_pred[:100], color='red', linestyle='--')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Actual vs Predicted Weight')
plt.legend()
plt.show()
