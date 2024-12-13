import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Шаг 1: Загрузка данных
file_path = r"C:\Users\ZenBook\Downloads\ДЗ_2_Прогнозирование_оценки_за_экз_по_ТОЭ.csv"
df = pd.read_csv(file_path)

# Шаг 2: Предобработка данных
df.drop(columns=["С какого раза вы сдали экзамен по ТОЭ (если улучшали оценку, так же указать с какого раза получили желаемую оценку)"], inplace=True)

# Замена 'да'/'нет' на 1/0
df.replace({'да': 1, 'нет': 0}, inplace=True)

# Преобразование категориальных данных
df = pd.get_dummies(df, columns=['Где проживаете?', 'Гражданство'], drop_first=True)

# Заполнение пропусков для числовых столбцов медианой
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Удаляем все нечисловые столбцы для корреляционного анализа
df_numeric = df.select_dtypes(include=[np.number])

# Шаг 3: Корреляционный анализ
corr_matrix = df_numeric.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица')
plt.show()

# Отбор корреляций с целевым признаком
target_corr = corr_matrix['Итоговая оценка за экзамен по ТОЭ [2-5]'].sort_values(ascending=False)
print(target_corr)

# Шаг 4: Построение модели
X = df.drop(columns=['Итоговая оценка за экзамен по ТОЭ [2-5]'])
y = df['Итоговая оценка за экзамен по ТОЭ [2-5]']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R²: {r2}')

# Шаг 5: Сохранение модели
model_filename = 'exam_score_prediction_model.joblib'
joblib.dump(model, model_filename)
print(f'Модель сохранена в {model_filename}')
