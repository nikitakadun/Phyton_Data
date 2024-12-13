import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/ZenBook/Downloads/bank+marketing/bank/bank.csv', sep=';')

# Оставляем только нужные столбцы
columns = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
df = df[columns]

# Проверка на пропущенные значения
missing_values = df.isnull().sum()

missing_values

# Находим моду для столбца education
mode_education = df['education'].mode()[0]

mode_education

# Оставляем только числовые признаки
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
corr_matrix = df[numerical_columns].corr()

# Смотрим корреляцию между признаками
corr_matrix

# Поиск двух признаков с наибольшей корреляцией
corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)

# Исключаем корреляцию признаков с самими собой
corr_pairs = corr_pairs[corr_pairs < 1]

# Вывод самой высокой корреляции
highest_corr = corr_pairs.head(1)

highest_corr



# Кодируем целевую переменную y
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Отделяем категориальные признаки
categorical_columns = ['job', 'marital', 'education', 'housing', 'contact', 'month', 'poutcome']

# Разделяем данные на тренировочный, валидационный и тестовый наборы
X_train, X_temp, y_train, y_temp = train_test_split(df.drop(columns=['y']), df['y'], test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Применяем one-hot кодирование для категориальных признаков
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)

# Рассчитываем взаимную информацию для категориальных признаков
mi_scores = mutual_info_classif(X_train_encoded, y_train, discrete_features='auto')
mi_scores_series = pd.Series(mi_scores, index=X_train_encoded.columns).sort_values(ascending=False)

# Округляем до двух знаков после запятой
mi_scores_rounded = mi_scores_series.round(2)

mi_scores_rounded

# Применяем one-hot кодирование ко всему набору данных
X_val_encoded = pd.get_dummies(X_val, columns=categorical_columns, drop_first=True)

# Инициализируем модель
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)

# Обучаем модель на тренировочных данных
model.fit(X_train_encoded, y_train)

# Предсказания на валидационных данных
y_val_pred = model.predict(X_val_encoded)

# Рассчитываем точность
val_accuracy = accuracy_score(y_val, y_val_pred)

# Округляем точность до двух знаков
val_accuracy_rounded = round(val_accuracy, 2)

val_accuracy_rounded

# Список всех признаков, включая закодированные категориальные признаки
all_features = X_train_encoded.columns

# Обучаем модель на всех признаках для получения исходной точности
base_model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
base_model.fit(X_train_encoded, y_train)

# Рассчитываем точность исходной модели
base_accuracy = accuracy_score(y_val, base_model.predict(X_val_encoded))

# Инициализируем словарь для хранения разницы в точности
accuracy_diff = {}

# Исключаем каждый признак по одному и оцениваем изменение точности
for feature in all_features:
    # Исключаем текущий признак
    X_train_excluded = X_train_encoded.drop(columns=[feature])
    X_val_excluded = X_val_encoded.drop(columns=[feature])
    
    # Обучаем модель на данных без текущего признака
    model_excluded = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model_excluded.fit(X_train_excluded, y_train)
    
    # Рассчитываем точность модели
    accuracy_excluded = accuracy_score(y_val, model_excluded.predict(X_val_excluded))
    
    # Разница в точности
    accuracy_diff[feature] = base_accuracy - accuracy_excluded

# Сортируем разницу в точности по возрастанию
sorted_accuracy_diff = sorted(accuracy_diff.items(), key=lambda x: x[1])

# Выводим признак с наименьшей разницей
sorted_accuracy_diff[:4]

# Значения параметра C, которые мы будем проверять
C_values = [0.01, 0.1, 1, 10, 100]  
accuracy_results = {}

for C in C_values:
    model_reg = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model_reg.fit(X_train_encoded, y_train)
    y_val_pred_reg = model_reg.predict(X_val_encoded)
    val_accuracy_reg = accuracy_score(y_val, y_val_pred_reg)
    accuracy_results[C] = round(val_accuracy_reg, 3)

best_C = max(accuracy_results, key=accuracy_results.get)
print("Точность для разных C:", accuracy_results)
print("Лучшее C:", best_C)

accuracy_results, best_C

mode_education = df['education'].mode()[0]
print("Мода для education:", mode_education)

highest_corr = corr_pairs.head(1)
print("Наибольшая корреляция:", highest_corr)

print("Взаимная информация:", mi_scores_rounded)

print("Точность логистической регрессии:", val_accuracy_rounded)

sorted_accuracy_diff = sorted(accuracy_diff.items(), key=lambda x: x[1])
least_impact_feature = sorted_accuracy_diff[0]  # Наименьшая разница
print("Наименьшая разница в точности:", least_impact_feature[0], "с разницей", least_impact_feature[1])

