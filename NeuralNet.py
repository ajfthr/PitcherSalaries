# Importing the Keras libraries and packages
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

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


num_buckets = 100

# cols_to_use = ['yearID','W','L','GS','SV','BK','R','H','ERA','SO', 'salary_bucket', 'salary_bucket_lag']
# joint_df['salary_bucket'] = pd.cut(joint_df['salary'], num_buckets, labels=False)
# joint_df['salary_bucket_lag'] = joint_df.groupby('playerID')['salary_bucket'].shift(1)

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


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

print("before estimator")
# Initializing Neural Network
estimator = KerasRegressor(build_fn=baseline_model,epochs=100, batch_size=5, verbose=0)

print("fitting estimator")
# Fitting our model


scores = cross_val_score(estimator, train_x, train_y, scoring="neg_mean_squared_error", cv=2)
rmse_scores = np.sqrt(-scores)

print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Std Dev: ", scores.std())

estimator.fit(train_x, train_y, batch_size=10, nb_epoch=100)

y_pred = estimator.pred(test_x)
print("Accuracy Score:", accuracy_score(test_y, y_pred) )
