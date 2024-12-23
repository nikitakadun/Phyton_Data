import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# Загрузка данных
# Сначала нужно скачать и распаковать данные из ссылки
# Здесь предполагается, что файл bank.csv уже распакован.

data = pd.read_csv("C:/Users/ZenBook/Downloads/bank.csv", sep=",")

# Оставляем только нужные столбцы
columns_to_use = [
    'age', 'job', 'marital', 'education', 'balance', 'housing', 'contact',
    'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'
]
data = data[columns_to_use]

# Преобразование целевой переменной в числовую
data['y'] = (data['y'] == 'yes').astype(int)

# Разделение данных на train, val, test
df_train, df_temp = train_test_split(data, test_size=0.4, random_state=1)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=1)

# Сброс индексов
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

numerical_features = ['balance', 'day', 'duration', 'previous']
auc_scores = {}

for feature in numerical_features:
    auc = roc_auc_score(df_train['y'], df_train[feature])
    # Инвертируем переменные с AUC < 0.5
    if auc < 0.5:
        auc = roc_auc_score(df_train['y'], -df_train[feature])
    auc_scores[feature] = auc

max_auc_feature = max(auc_scores, key=auc_scores.get)
auc_scores, max_auc_feature

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# One-hot encoding
train_dicts = df_train.drop(columns=['y']).to_dict(orient='records')
val_dicts = df_val.drop(columns=['y']).to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

# Целевая переменная
y_train = df_train['y'].values
y_val = df_val['y'].values

# Обучение модели
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# AUC на валидационной выборке
y_val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
val_auc

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Предсказания вероятностей
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred)

# График Precision-Recall
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.legend()
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.show()

# Найдем пересечение
for i in range(len(thresholds)):
    if abs(precisions[i] - recalls[i]) < 0.01:
        best_threshold = thresholds[i]
        break
best_threshold

# Вычисление F1 для каждого порога
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1])
best_f1_threshold = thresholds[np.argmax(f1_scores)]
best_f1_threshold

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=1)
auc_scores = []

for train_idx, val_idx in kf.split(df_train):
    df_fold_train = df_train.iloc[train_idx]
    df_fold_val = df_train.iloc[val_idx]
    
    X_fold_train = dv.fit_transform(df_fold_train.drop(columns=['y']).to_dict(orient='records'))
    X_fold_val = dv.transform(df_fold_val.drop(columns=['y']).to_dict(orient='records'))
    
    y_fold_train = df_fold_train['y'].values
    y_fold_val = df_fold_val['y'].values
    
    model.fit(X_fold_train, y_fold_train)
    y_fold_pred = model.predict_proba(X_fold_val)[:, 1]
    auc = roc_auc_score(y_fold_val, y_fold_pred)
    auc_scores.append(auc)

# Стандартная ошибка
std_error = np.std(auc_scores)
std_error

C_values = [0.000001, 0.001, 1]
results = []

for C in C_values:
    auc_scores = []
    for train_idx, val_idx in kf.split(df_train):
        df_fold_train = df_train.iloc[train_idx]
        df_fold_val = df_train.iloc[val_idx]
        
        X_fold_train = dv.fit_transform(df_fold_train.drop(columns=['y']).to_dict(orient='records'))
        X_fold_val = dv.transform(df_fold_val.drop(columns=['y']).to_dict(orient='records'))
        
        y_fold_train = df_fold_train['y'].values
        y_fold_val = df_fold_val['y'].values
        
        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_fold_train, y_fold_train)
        y_fold_pred = model.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, y_fold_pred)
        auc_scores.append(auc)
    
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    results.append((C, mean_auc, std_auc))

results
