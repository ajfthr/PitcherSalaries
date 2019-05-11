# Importing the Keras libraries and packages
import keras
import pandas as pd
import numpy as np
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import normalize
from sklearn.model_selection import cross_val_score, GridSearchCV
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


#num_buckets = 100
#
# cols_to_use = ['yearID','W','L','GS','SV','BK','R','H','ERA','SO', 'salary_bucket', 'salary_bucket_lag']
# joint_df['salary_bucket'] = pd.cut(joint_df['salary'], num_buckets, labels=False)
# joint_df['salary_bucket_lag'] = joint_df.groupby('playerID')['salary_bucket'].shift(1)

cols_to_use = ['yearID', 'W', 'L', 'GS', 'SV', 'BK', 'R', 'H', 'ERA', 'SO', 'salary', 'salary_lag']
joint_df['salary_lag'] = joint_df.groupby('playerID')['salary'].shift(1)

#for x in ['W', 'L', 'GS', 'SV', 'BK', 'R', 'H', 'ERA', 'SO', 'salary', 'salary_lag']:
#    joint_df[x] = (joint_df[x] - joint_df[x].mean()) / (joint_df[x].max() - joint_df[x].min())

joint_df = joint_df.dropna()

df_train, df_test = split_train_test(joint_df, .2)
df_train = df_train.filter(cols_to_use)
df_test = df_test.filter(cols_to_use)

train_x = df_train.drop('salary', 1)
train_y = np.log(df_train['salary'])
test_x = df_train.drop('salary', 1)
test_y = np.log(df_train['salary'])


estimators = []

predictions = None

dropout_size = [0,.1,.2,.25,.4,.5,.7]
layer_size = [4,5,6,7,8,9,10]

for drop_one in dropout_size:
    for drop_two in dropout_size:
        for lsize in layer_size:

            print("\n\nDrop_one: ",drop_one, " drop_two: ", drop_two, " hidden layer size: ",lsize, "\n")

            def baseline_model():
                # create model
                model = Sequential()
                model.add(Dropout(drop_one, input_shape=(11,)))
                model.add(Dense(11, input_dim=11, kernel_initializer='normal', activation='relu'))
                model.add(Dropout(drop_two))
                model.add(Dense(lsize, kernel_initializer='normal', activation='relu'))
                model.add(Dense(1, kernel_initializer='normal', activation='relu'))
                # Compile model
                model.compile(loss='mean_squared_error', optimizer='adam')
                return model


            #Initializing Neural Network
            param_grid = dict(epochs = [200])
            # param_grid = dict(epochs = [400,500], batch_size=[10])
            estimator = KerasRegressor(build_fn=baseline_model, verbose=0)

            estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=5)

            #estimator = KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=10, verbose=0)

            train_x = normalize(x=train_x)

            estimator.fit(train_x, train_y)
            print(estimator.best_params_)
            print(estimator.cv_results_)

            #scores = cross_val_score(estimator, train_x, train_y, scoring="neg_mean_squared_error", cv=5)

            # print("Scores: ", scores)
            # print("Mean: ", scores.mean())
            # print("Std Dev: ", scores.std())

            #Normalize
            test_x = normalize(x=test_x)
            y_pred = estimator.predict(test_x)

            top_five = { 0: {}}

            total_error = 0
            for player, year, prediction, label in zip(joint_df['playerID'], joint_df['yearID'], y_pred, test_y):

                error = abs(prediction - label)

                total_error += error ** 2
                if((len(top_five.keys()) < 5) or (min(top_five.keys()) < error)):
                    if(len(top_five.keys()) >= 5):
                        del top_five[min(top_five.keys())]

                    top_five[error] = { 'player': player, 'year': year, 'salary': label, 'prediction': prediction}

            print("MSE: ", total_error/len(test_y), "\n")

            for key in top_five.keys():
                print(key, "  ", top_five[key], "\n\n")

            # full_sample = joint_df.filter(['yearID', 'W', 'L', 'GS', 'SV', 'BK', 'R', 'H', 'ERA', 'SO', 'salary_lag'])
            # full_sample = normalize(x=full_sample)
            # true_y = np.log(joint_df['salary'])
            # y_pred = estimator.predict(full_sample)
            #
            # top_five = { 0: {}}
            # for player, year, prediction, label in zip(joint_df['playerID'], joint_df['yearID'], y_pred, true_y):
            #
            #     error = abs(prediction - label)
            #     if((len(top_five.keys()) < 5) or (min(top_five.keys()) < error)):
            #         if(len(top_five.keys()) >= 5):
            #             del top_five[min(top_five.keys())]
            #
            #         top_five[error] = { 'player': player, 'year': year, 'salary': label, 'prediction': prediction}
            #
            # for key in top_five.keys():
            #     print(key, "  ", top_five[key], "\n\n")

            #
            # estimator.fit(train_x, train_y, batch_size=10, nb_epoch=100)
            #
            # y_pred = estimator.predict(test_x)
            #
            #
            # print("Accuracy Score:", mean_squared_error(test_y, y_pred) )
