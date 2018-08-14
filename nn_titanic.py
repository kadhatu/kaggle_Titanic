#!/usr/bin/python

import pandas as pd 
import numpy as np 
from pandas import Series, DataFrame
import tensorflow as tf

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

# use random forest to fill some of the missing data
from sklearn.ensemble import RandomForestRegressor 
def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    
    y = known_age[:, 0]
    X = known_age[:, 1:]
 
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges  
    return df, rfr

# deal with features with half of missing value
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


# handling the training data
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
#data_test = set_Cabin_type(data_test)
#data_test = set_missing_ages(data_test)

# set one-hot encode for no-nonnumeric features
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

#datasets re-construction to the copies of df
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# features scaler
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

#reshape features before put into scaler
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)

train_df = df.filter(regex="Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*")
train_np = train_df.values


# handling the test data
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

age_scale_param_test = scaler.fit(df_test['Age'].values.reshape(-1,1))
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param_test)
fare_scale_param_test = scaler.fit(df_test['Fare'].values.reshape(-1,1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param_test)
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test.values
x_test = test_np[:, 0:]



# tensorflow model
y_train = train_np[:, 0, np.newaxis]
#y_train = train_np[:, 0]
x_train = train_np[:, 1:]

def add_layer(inputs,input_size,output_size,activation_function=None):
    with tf.variable_scope("Weights"):
        Weights = tf.Variable(tf.random_normal(shape=[input_size,output_size]),name="weights")
    with tf.variable_scope("biases"):
        biases = tf.Variable(tf.zeros(shape=[1,output_size]) + 0.1,name="biases")
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
    with tf.name_scope("dropout"):
        Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob=keep_prob_s)
    if activation_function is None:
        return Wx_plus_b
    else:
        with tf.name_scope("activation_function"):
            return activation_function(Wx_plus_b)


xs = tf.placeholder(shape=[None,x_train.shape[1]],dtype=tf.float32,name="inputs")
ys = tf.placeholder(shape=[None,1],dtype=tf.float32,name="y_true")
keep_prob_s = tf.placeholder(dtype=tf.float32, name='keep_prob')

with tf.name_scope("layer_1"):
    l1 = add_layer(xs,14,15,activation_function=tf.nn.sigmoid)
with tf.name_scope("layer_2"):
	l2 = add_layer(l1,15,15,activation_function=tf.nn.sigmoid)
with tf.name_scope("y_pred"):
    pred = add_layer(l1,15,1,activation_function=tf.nn.sigmoid)

# save pred
pred = tf.add(pred,0,name='pred')

with tf.name_scope("loss"):
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred),reduction_indices=[1]))  # mse
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=pred))
    tf.summary.scalar("loss",tensor=loss)
with tf.name_scope("train"):
    # train_op =tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


# parameters
keep_prob=1  # prevent from overfitting
ITER =5000  # training iter

# define the training function
def fit(X, y, n, keep_prob):
    init = tf.global_variables_initializer()
    feed_dict_train = {ys: y, xs: X, keep_prob_s: keep_prob}
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="nn_titanic_log", graph=sess.graph)  #写tensorbord
        sess.run(init)
        for i in range(n):
            _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict_train)

            if i % 100 == 0:
                print("epoch:%d\tloss:%.5f" % (i, _loss))
                y_pred = sess.run(pred, feed_dict=feed_dict_train)
                rs = sess.run(merged, feed_dict=feed_dict_train)
                writer.add_summary(summary=rs, global_step=i)  #写tensorbord
                saver.save(sess=sess, save_path="nn_titanic_model/nn_titanic.model", global_step=i) # 保存模型

        saver.save(sess=sess, save_path="nn_titanic_model/nn_titanic.model", global_step=n)  # 保存模型

fit(X=x_train,y=y_train,n=ITER,keep_prob=keep_prob)
print('Train Finish.')


# define the predict function
def predict(X,keep_prob):
    with tf.Session() as sess:
        # restore saver
        saver = tf.train.import_meta_graph(meta_graph_or_file="nn_titanic_model/nn_titanic.model-5000.meta")
        model_file = tf.train.latest_checkpoint(checkpoint_dir="nn_titanic_model")
        saver.restore(sess=sess,save_path=model_file)

        # init graph
        graph = tf.get_default_graph()

        # get placeholder from graph
        xs = graph.get_tensor_by_name("inputs:0")
        #ys = graph.get_tensor_by_name("y_true:0")
        keep_prob_s = graph.get_tensor_by_name("keep_prob:0")

        # get operation from graph
        pred = graph.get_tensor_by_name("pred:0")

        # run pred
        #feed_dict = {xs: X, ys: y, keep_prob_s: keep_prob}
        feed_dict = {xs: X, keep_prob_s: keep_prob}
        y_pred = sess.run(pred,feed_dict=feed_dict)

    return y_pred.reshape(-1,1)


#y_pred = predict(X=x_test,y=y_train,keep_prob=1)
y_pred = predict(X=x_test,keep_prob=1)
y_pred[np.where(y_pred[:,0]<0.5)] = 0
y_pred[np.where(y_pred[:,0]>=0.5)] = 1
print(y_pred)

# drop the extra columns for uploading to kaggle 
y_pred = y_pred.astype(np.int32)
data_test['Survived'] = y_pred
data_test.drop(['Pclass'], axis=1, inplace=True)
data_test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
data_test.drop(['Age', 'SibSp', 'Parch', 'Fare'], axis=1, inplace=True)

data_test.to_csv("result.csv",index=False,sep=',')  
