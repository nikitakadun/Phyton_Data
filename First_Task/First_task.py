import pandas as pd
import numpy as np

# Загружаем данные
DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
DF = pd.read_csv(DATA_URL + "adult.data.csv", header=None)

# Присваиваем имена столбцам для удобства работы
DF.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
              'marital-status', 'occupation', 'relationship', 'race', 'sex', 
              'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# 1. Число столбцов в наборе данных
num_columns = DF.shape[1]
print(f"Число столбцов: {num_columns}")

# 2. Проверка пропусков (проверим наличие пропусков, закодированных как '?')
missing_symbol = (DF == '?').sum()
missing_cols = missing_symbol[missing_symbol > 0]
if not missing_cols.empty:
    print(f"Пропуски, закодированные как '?':\n{missing_cols}")
else:
    print("Пропусков в данных нет.")

# Заменяем все '?' на NaN
DF.replace('?', np.nan, inplace=True)

# 3. Количество уникальных значений в столбце 'race'
unique_race_values = DF['race'].nunique()
print(f"Количество уникальных значений в столбце 'race': {unique_race_values}")

# 4. Преобразуем столбец 'hours-per-week' в числовой формат
DF['hours-per-week'] = pd.to_numeric(DF['hours-per-week'], errors='coerce')

# Теперь можем найти медиану в столбце 'hours-per-week'
median_hours = DF['hours-per-week'].median()
print(f"Медиана 'hours-per-week': {median_hours}")

# 5. Кто преобладает - мужчины или женщины с ЗП >50K?
high_income_df = DF[DF['income'] == '>50K']
gender_count = high_income_df['sex'].value_counts()
print(f"Кого больше среди людей с ЗП >50K:\n{gender_count}")
if gender_count['Male'] > gender_count['Female']:
    print("Мужчин больше с ЗП >50K")
else:
    print("Женщин больше с ЗП >50K")

# 6. Заполнение пропущенных данных наиболее частыми значениями
for column in DF.columns:
    most_common_value = DF[column].mode()[0]
    DF[column] = DF[column].fillna(most_common_value)

# Проверка, что пропуски были заполнены
missing_data_after = DF.isnull().sum()
if missing_data_after.sum() == 0:
    print("Все пропущенные данные успешно заполнены.")
else:
    print(f"Остались пропуски:\n{missing_data_after[missing_data_after > 0]}")

# Альтернативные способы заполнения пропущенных данных:
# 1. Среднее значение для числовых столбцов
# 2. Медианное значение для числовых данных
# 3. Специальные методы (например, с использованием машинного обучения)

# Число столбцов: 15
# Пропуски, закодированные как '?':
# workclass         1836
# occupation        1843
# native-country     583
# dtype: int64
# Количество уникальных значений в столбце 'race': 6
# Медиана 'hours-per-week': 40.0
# Кого больше среди людей с ЗП >50K:
# sex
# Male      6662
# Female    1179
# Name: count, dtype: int64
# Мужчин больше с ЗП >50K