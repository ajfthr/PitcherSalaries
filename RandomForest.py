import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

pitching_df = pd.read_csv('data/Pitching.csv')
salary_df = pd.read_csv('data/Salaries.csv')

joint_df = pd.merge(pitching_df, salary_df, on=['teamID', 'playerID', 'yearID', 'lgID'])

joint_df = joint_df.sort_values(by=['playerID', 'yearID', 'stint'])

joint_df['salary_lag'] = joint_df.groupby('playerID')['salary'].shift(1)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

cols_to_use = ['yearID', 'W', 'L', 'GS', 'SV', 'BK', 'R', 'H', 'ERA', 'SO', 'salary', 'salary_lag']
joint_df['salary_lag'] = joint_df.groupby('playerID')['salary'].shift(1)

joint_df = joint_df.dropna()

df_train, df_test = split_train_test(joint_df, .2)
df_train = df_train.filter(cols_to_use)
df_test = df_test.filter(cols_to_use)

train_x = df_train.drop('salary', 1)
train_y = np.log(df_train['salary'])
test_x = df_train.drop('salary', 1)
test_y = np.log(df_train['salary'])

search = False
if(search):

    param_grid = [
        {'n_estimators':[3, 10, 30, 50, 100], 'max_features': [2,4,6,8,10,11], 'max_depth':[5,10,20,40,50,100]}
    ]

    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(train_x,train_y)


    print("Best gridsearch params")
    print(grid_search.best_params_)


    y_pred = grid_search.predict(test_x)

    print("MSE:", mean_squared_error(test_y, y_pred))


estimator = RandomForestRegressor(n_estimators=100,max_features=6, max_depth=10)
estimator.fit(train_x,train_y)


y_pred = estimator.predict(test_x)
print("MSE:", mean_squared_error(test_y, y_pred))

full_sample = joint_df.filter(['yearID', 'W', 'L', 'GS', 'SV', 'BK', 'R', 'H', 'ERA', 'SO', 'salary_lag'])
true_y = np.log(joint_df['salary'])
predictions = estimator.predict(full_sample)

top_five = { 0: {}}
for player, year, prediction, label in zip(joint_df['playerID'], joint_df['yearID'], predictions, true_y):

    error = abs(prediction - label)
    if((len(top_five.keys()) < 5) or (min(top_five.keys()) < error)):
        if(len(top_five.keys()) >= 5):
            del top_five[min(top_five.keys())]

        top_five[error] = { 'player': player, 'year': year, 'salary': label, 'prediction': prediction}

for key in top_five.keys():
    print(key, "  ", top_five[key], "\n\n")

# diffs = np.absolute(true_y - estimated_salaries)
#
# top_errors = diffs.argsort()[-20:][::-1]
#
# for x in top_errors.index.values.tolist():
#     print(top_errors.at[x])
#     print("PlayerID:" + str(joint_df.at[x, 'playerID']) + " Salary: " + str(joint_df.at[x, 'salary']))


# scores = cross_val_score(clf, train_x, train_y, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
#
# print("Scores: ", scores)
# print("Mean: ", scores.mean())
# print("Std Dev: ", scores.std())