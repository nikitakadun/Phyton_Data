import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

# Загрузка данных
df = pd.read_csv("C:/Users/ZenBook/Downloads/jamb_exam_results.csv")

# Преобразование названий колонок
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Удаление столбца student_id
df = df.drop(columns=['student_id'])

# Заполнение пропущенных значений нулями
df = df.fillna(0)

# Разделение данных на train/validation/test
df_train, df_temp = train_test_split(df, test_size=0.4, random_state=1)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=1)

# Преобразование данных в матрицы
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.drop(columns=['jamb_score']).to_dict(orient='records'))
y_train = df_train['jamb_score']

X_val = dv.transform(df_val.drop(columns=['jamb_score']).to_dict(orient='records'))
y_val = df_val['jamb_score']

X_test = dv.transform(df_test.drop(columns=['jamb_score']).to_dict(orient='records'))
y_test = df_test['jamb_score']

from sklearn.tree import DecisionTreeRegressor

# Обучение модели
dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train, y_train)

# Получаем информацию о разбиении
feature = dv.feature_names_[dt.tree_.feature[0]]
print("Feature used for splitting:", feature)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Обучение модели
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

# Предсказание и расчет RMSE
y_pred = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("RMSE:", round(rmse, 2))

rmse_values = []
for n in range(10, 210, 10):
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_values.append((n, round(rmse, 3)))

# Печатаем RMSE для каждого n
for n, rmse in rmse_values:
    print(f"n_estimators={n}, RMSE={rmse}")

# Найти первое значение n, после которого улучшений нет
best_n = min(rmse_values, key=lambda x: x[1])
print("Best n_estimators:", best_n[0])

depth_values = [10, 15, 20, 25]
n_estimators_values = range(10, 210, 10)

results = {}
for depth in depth_values:
    rmse_depth = []
    for n in n_estimators_values:
        rf = RandomForestRegressor(max_depth=depth, n_estimators=n, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_depth.append(rmse)
    results[depth] = np.mean(rmse_depth)

# Определяем лучшую глубину
best_depth = min(results, key=results.get)
print("Best max_depth:", best_depth)

rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

# Получение важности признаков
importances = rf.feature_importances_
features = dv.feature_names_

# Печатаем важные признаки
important_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
print("Most important feature:", important_features[0][0])

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Watchlist
watchlist = [(dtrain, 'train'), (dval, 'eval')]

# Обучение с eta=0.3
xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'seed': 1
}
model_1 = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False)
y_pred_1 = model_1.predict(dval)
rmse_1 = np.sqrt(mean_squared_error(y_val, y_pred_1))

# Обучение с eta=0.1
xgb_params['eta'] = 0.1
model_2 = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False)
y_pred_2 = model_2.predict(dval)
rmse_2 = np.sqrt(mean_squared_error(y_val, y_pred_2))

print("RMSE for eta=0.3:", round(rmse_1, 3))
print("RMSE for eta=0.1:", round(rmse_2, 3))

