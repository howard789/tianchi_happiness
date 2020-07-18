import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from common_kaggle import common_util

def load_data_regression():
    X = np.array(list(range(1, 11))).flatten()
    y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]).flatten()
    data=pd.DataFrame()
    data['x']=X
    data['target']=y
    return data


# def __load_iris():
#     iris = datasets.load_iris()
#     train_x,test_x,train_y,test_y=train_test_split(iris.data,iris.target,test_size=0.25)
#
#     df_train=pd.DataFrame(train_x)
#     df_train.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#     df_train['target'] = train_y
#
#     df_test=pd.DataFrame(test_x)
#     df_test.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#     df_test['target'] = test_y
#
#     return df_train,df_test

def __get_iris_data(test_size=0.2):
    iris = datasets.load_iris()
    train_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    train_data['label'] = iris.target
    print(iris.target_names)
    test_data = pd.DataFrame()

    if (test_size > 0):
        train_data, test_data = train_test_split(train_data, test_size=test_size)
    return train_data, test_data




def get_iris_data(test_size=0.2):
    df_train, df_test=__get_iris_data(test_size=test_size)
    train_X, train_y=common_util.split_df_to_array(df_train, label_column='label')
    if (len(df_test)>0):
        test_X, test_y=common_util.split_df_to_array(df_test, label_column='label')
    return df_train,df_test,train_X, test_X, train_y, test_y












def get_data(model):
    dic = {}
    dic['regression'] = [pd.DataFrame(data=[[1, 5, 20, 1.1],
                                  [2, 7, 30, 1.3],
                                  [3, 21, 70, 1.7],
                                  [4, 30, 60, 1.8],
                                  ], columns=['id', 'age', 'weight', 'label']),
                         pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])]

    dic['binary_cf'] = [pd.DataFrame(data=[[1, 5, 20, 0],
                        [2, 7, 30, 0],
                        [3, 21, 70, 1],
                        [4, 30, 60, 1],
                        ], columns=['id', 'age', 'weight', 'label']),
                        pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])]

    dic['multi_cf'] = [pd.DataFrame(data=[[1, 5, 20, 0],
                        [2, 7, 30, 0],
                        [3, 21, 70, 1],
                        [4, 30, 60, 1],
                        [5, 30, 60, 2],
                        [6, 30, 70, 2],
                        ], columns=['id', 'age', 'weight', 'label']),
                       pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])]

    return dic[model]


if __name__ == '__main__':
    get_iris_data(test_size=0.2)