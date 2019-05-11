
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score

pitching_df = pd.read_csv('data/Pitching.csv')
salary_df = pd.read_csv('data/Salaries.csv')

joint_df = pd.merge(pitching_df,salary_df, on=['teamID','playerID','yearID','lgID'])

joint_df = joint_df.sort_values(by=['playerID','yearID','stint'])

joint_df['salary_lag'] = joint_df.groupby('playerID')['salary'].shift(1)


def split_train_test( data, test_ratio):
    shuffled_indices = np.random.permutation( len( data))
    test_set_size = int( len( data) * test_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[ test_set_size:]
    return data.iloc[ train_indices], data.iloc[ test_indices]

joint_df = joint_df.dropna()

#joint_df = joint_df[joint_df['salary'] < 5000000]

features = ['playerID','yearID','teamID','lgID','W','L','GS','SV','BK','R','H','ERA','SO']

CONTINUOUS_COLUMNS = ['yearID','W','L','GS','SV','BK','R','H','ERA','SO', 'salary_lag']
CATEGORICAL_COLUMNS = ['playerID','teamID','lgID']
LABEL_COLUMN = 'salary'

#joint_df = joint_df.filter(items=CONTINUOUS_COLUMNS.append("salary"))

df_train, df_test = split_train_test(joint_df, .2)

k,n = df_train.shape
n_inputs = k*n
#x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
#y = tf.placeholder(tf.int32, shape=(None), name="y")

x = df_train.filter(items=CONTINUOUS_COLUMNS)
y = df_train.pop('salary')

clf = RandomForestRegressor(n_estimators=120, max_depth=100, random_state=0)
clf.fit(x,y)

#print(clf.feature_importances_)
x = df_test.filter(items=CONTINUOUS_COLUMNS)
y = df_test.pop('salary')
predictions = clf.predict(x)

# scores = cross_val_score(clf, x, y)
# print(scores)

print(mean_squared_error(y.values, predictions.astype(int)))

exit(1)

#Bucketized
##The below bucketizing doesn't work.
# Try this: (KBinsDiscretizer) https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_strategies.html

joint_df['salary_bucket'] = pd.cut(joint_df['salary'], 20)


joint_df['salary__bucket_lag'] = joint_df.groupby('playerID')['salary_bucket'].shift(1)
joint_df = joint_df.dropna()
df_train, df_test = split_train_test(joint_df, .2)

CONTINUOUS_COLUMNS = ['yearID','W','L','GS','SV','BK','R','H','ERA','SO', 'salary_bucket_lag']
x = df_train.filter(items=CONTINUOUS_COLUMNS)
y = df_train.pop('salary_bucket')
print(y.tail())
clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
clf.fit(x,y)

#print(clf.feature_importances_)
x = df_test.filter(items=CONTINUOUS_COLUMNS)
y = df_test.pop('salary_bucket')
predictions = clf.predict(x)

# scores = cross_val_score(clf, x, y)
# print(scores)

print(mean_squared_error(y, predictions)/len(y))

#
# def neuron_layer(X, n_neurons, name, activation=None):
#     with tf.name_scope(name):
#         n_inputs = int(X.get_shape()[1])
#         stddev = 2/np.sqrt(n_inputs + n_neurons)
#         init = tf.truncated_normal((n_inputs,n_neurons), stddev=stddev)
#         W = tf.Variable(init, name="kernel")
#         b = tf.Variable(tf.zeros([n_neurons]), name="bias")
#         Z = tf.matmul(X,W) + b
#         if(activation is not None):
#             return activation(Z)
#         else:
#             return Z
#
# with tf.name_scope("dnn"):
#     hidden_1 = neuron_layer(x, n_hidden1, name="hidden1", activation=tf.nn.relu)
#     hidden_2 = neuron_layer(hidden_1, n_hidden2, name="hidden2", activation=tf.nn.relu)
#     logits = neuron_layer(hidden_2, n_outputs, name="outputs")
#
# with tf.name_scope("loss"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#     loss = tf.reduce_mean(xentropy, name="loss")
#
# learning_rate = 0.01
#
# with tf.name_scope("train"):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     training_op = optimizer.minimize(loss)
#
# with tf.name_scope("eval"):
#     correct = tf.nn.in_top_k(logits,y,1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# n_epochs = 40
# batch_size = 50
#
# with tf.Session () as sess :
#     init.run()
#     for epoch in range ( n_epochs ):
#         for iteration in range ( mnist.train.num_examples // batch_size ):
#             X_batch , y_batch = mnist.train.next_batch ( batch_size )
#             sess.run ( training_op , feed_dict = { x : X_batch , y : y_batch })
#         acc_train = accuracy.eval ( feed_dict = { x : X_batch , y : y_batch })
#         acc_val = accuracy.eval ( feed_dict = { x : mnist.validation.images , y : mnist.validation.labels })
#         print ( epoch , "Train accuracy:" , acc_train , "Val accuracy:" , acc_val )
#     save_path = saver.save ( sess , "./my_model_final.ckpt" )
