import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
from scipy.spatial.distance import cdist

def import_dataset_from_file(path_to_file: str) -> pd.DataFrame:
    """
    Функция импортирования исходных данных.
    :param path_to_file: путь к загружаемому файлу;
    :return: структура данных.
    """
    dataset = pd.read_table(path_to_file, delim_whitespace=True, names=['x', 'y', 'z'])

    return dataset


def export_dataset_to_file(dataset: pd.DataFrame):
    """
    Функция экспортирования результата в файл result.txt.
    :param dataset: входная структура данных.
    """
    n, c = dataset.shape

    assert c == 3, 'Количество столбцов должно быть 3'
    assert n == 1196590, 'Количество строк должно быть 1196590'

    with open('..\Data\\Result.txt', 'w') as f:
        for i in range(n):
            f.write('%.2f %.2f %.5f\n' % (dataset.x[i], dataset.y[i], dataset.z[i]))


    # Вспомогательные данные, по которым производится моделирование
map_1_dataset = import_dataset_from_file("..\Data\\Map_1.txt")
map_2_dataset = import_dataset_from_file("..\Data\\Map_2.txt")
map_3_dataset = import_dataset_from_file("..\Data\\Map_3.txt")
map_4_dataset = import_dataset_from_file("..\Data\\Map_4.txt")
map_5_dataset = import_dataset_from_file("..\Data\\Map_5.txt")

# Данные, по которым необходимо смоделировать
point_dataset = import_dataset_from_file("..\Data\\Point_dataset.txt")
# Точки данных, в которые необходимо провести моделирование (сетка данных)
point_grid = import_dataset_from_file("..\Data\\Result_schedule.txt")

# Блок вычислений
full_df = pd.merge(map_1_dataset, map_2_dataset, how="left", on=["x", "y"])
full_df = full_df.rename(columns={'z_x': 'z1', 'z_y': 'z2'})
full_df = pd.merge(full_df, map_3_dataset, how="left", on=["x", "y"])
full_df = full_df.rename(columns={'z': 'z3'})
full_df = pd.merge(full_df, map_4_dataset, how="left", on=["x", "y"])
full_df = full_df.rename(columns={'z': 'z4'})
full_df = pd.merge(full_df, map_5_dataset, how="left", on=["x", "y"])
full_df = full_df.rename(columns={'z': 'z5'})

filtered_point_dataset = point_dataset[point_dataset['x'] != 68473.37]
filtered_point_dataset = filtered_point_dataset[point_dataset['x'] != 63779.36]
filtered_point_dataset = filtered_point_dataset[point_dataset['x'] != 44232.60]
filtered_point_dataset = filtered_point_dataset[point_dataset['x'] != 43745.18]
filtered_point_dataset = filtered_point_dataset[point_dataset['x'] != 43773.71]
filtered_point_dataset = filtered_point_dataset[point_dataset['x'] != 43778.17]
filtered_point_dataset = filtered_point_dataset[point_dataset['x'] != 53717.64]
filtered_point_dataset = filtered_point_dataset[point_dataset['x'] != 57760.00]
filtered_point_dataset = filtered_point_dataset[point_dataset['x'] != 71483.43]
filtered_point_dataset = filtered_point_dataset[point_dataset['x'] != 71584.98]

point_dataset = filtered_point_dataset

distances = cdist(point_dataset[['x', 'y']], map_1_dataset[['x', 'y']], metric='euclidean')

closest_indices = distances.argmin(axis=1)

closest_coordinates = map_1_dataset.iloc[closest_indices]

closest_coordinates = pd.DataFrame({
    'x': map_1_dataset.loc[closest_indices, 'x'].values,
    'y': map_1_dataset.loc[closest_indices, 'y'].values,
    'z': point_dataset['z']
    })

test = pd.merge(full_df, closest_coordinates, how="left", on=["x", "y"])
test = test.dropna()





df = test

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = df[['x', 'y', 'z1', 'z2', 'z3', 'z4', 'z5']]
y_train = df['z']


model = xgb.XGBRegressor(n_estimators=3845, learning_rate=0.13063986534170344, max_depth= 94, min_child_weight= 10, gamma= 0.5170048777749798, subsample= 0.8386179928436251, colsample_bytree= 0.8518023755243972, scale_pos_weight=2)
model.fit(X_train, y_train)

y_pred = model.predict(full_df)

fin_df = full_df
fin_df['pred'] = y_pred
fin_point_df = point_grid
fin_point_df = pd.merge(fin_point_df, fin_df, how="left", on=["x", "y"])
fin_point_df = fin_point_df.drop(['z', 'z1', 'z2', 'z3', 'z4', 'z5'], axis = 1)
fin_point_df.rename(columns={'pred': 'z'}, inplace=True)
fin_point_df['z'] = fin_point_df['z'].fillna(fin_point_df['z'].median())



                 
        # Экспорт данных в файл (смотри Readme.txt)
export_dataset_to_file(fin_point_df)
print('done')
