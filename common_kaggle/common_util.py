import pandas as pd
import numpy as np
import math
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
from common_kaggle import mathUtil

def round_arr(arr):
    arr2=[]
    for i in range(len(arr)):
        arr2.append(round(arr[i]))
    return np.array(arr2)



def cal_periodic_value(value,num,percent=0.05):
    totalValue=0.
    for i in range(1,num+1):
        dis=(1-percent)**i
        totalValue+=value*dis
    return totalValue





def scale_data_oneHundred(start,end,value):
    return int((value-start)/(end-start)*100)




    return minV,maxV,mean,contain_nan

def cal_age(yMd,split_sym):
    birth_day=str(yMd).split(split_sym)
    age = datetime.date.today().year - datetime.date(birth_day[0],birth_day[1],birth_day[2])
    return age


def split_train_test(train_df,test_size=0.2):
    print(train_df.shape)
    train_df, test_df=train_test_split(train_df,test_size=test_size)
    print(train_df.shape)
    print(test_df.shape)

    train_df=train_df.reset_index(drop=True)
    test_df=test_df.reset_index(drop=True)

    return train_df, test_df

def split_trainX_to_four(train_df,label_column,test_size=0.2):
    train_df, test_df = split_train_test(train_df, test_size)
    train_X, train_y=split_df_to_array(train_df, label_column)
    test_X, test_y=split_df_to_array(test_df, label_column)
    return train_X,train_y,test_X, test_y


def cal_correct_rate__df(df,label_column='label',forcast_column='forcast'):
    df['correct']=df.apply(lambda x:1 if x[label_column]==x[forcast_column] else 0, axis=1)
    return df['correct'].sum()/df.shape[0]


def cal_correct_rate_df(y_true_array,y_predict_array):
    if (len(y_true_array) != len(y_predict_array)):
        raise ValueError('number diff')
    df=pd.DataFrame()
    label_column = 'label'
    forcast_column = 'forcast'
    df[label_column]=pd.Series(y_true_array)
    df[forcast_column]=y_predict_array
    result=cal_correct_rate__df(df,label_column,forcast_column)
    print('正确率{:.2%}'.format(result))
    return result


# def cal_correct_rate_3args(y_true_array,y_predict_array,y_true_ids):
#     if (len(y_true_array) != len(y_predict_array)):
#         raise ValueError('number diff')
#     incorrect_list=[]
#     count=0
#     for i in range(len(y_true_array)):
#         if(y_true_array[i]!=y_predict_array[i]):
#             id=y_true_ids[i]
#             true=y_true_array[i]
#             predict=y_predict_array[i]
#             incorrect_list.append((id,true,predict))
#         else:
#             count+=1
#     rate=count/len(y_true_array)
#     print('正确率{:.2%}'.format(rate))
#     print('错误的id,true,predict',incorrect_list)
#     return rate,incorrect_list

def cal_mse(arr1,arr2,pre_text=None):
    mse_value=mathUtil.mse(arr1,arr2)
    if(pre_text is not None):
        print('{}:MSE {:.4f}'.format(pre_text,mse_value))
    else:
        print('MSE {:.4f}'.format(mse_value))
    return mse_value


def cal_correct_rate(y_true_array,y_predict_array,id_list=None,need_round=False,pre_text=None):
    if (len(y_true_array) != len(y_predict_array)):
        raise ValueError('number diff')
    if (id_list is not None and len(id_list) != len(y_predict_array)):
        raise ValueError('id_list number diff')
    incorrect_list=[]
    count=0
    for i in range(len(y_true_array)):
        true_v=y_true_array[i]
        if(need_round):
            pred_v=round(y_predict_array[i])
        else:
            pred_v = y_predict_array[i]

        if(true_v!=pred_v):
            count+=1
            if(id_list is not None):
                incorrect_list.append(id_list[i])
    rate=(len(y_true_array)-count)/len(y_true_array)
    if(pre_text is not None):
        print('{}:正确率{:.2%}'.format(pre_text,rate))
    else:
        print('正确率{:.2%}'.format(rate))
    # print('错误的id,true,predict',incorrect_list)
    return rate,incorrect_list



def split_df_to_array(df,label_column='label'):
    y_array=np.array((df[label_column].tolist()))
    df2=df.drop(columns=[label_column])
    x_array = np.array(df2)
    return x_array,y_array









def xgb_demo():
    train_data, testData=get_iris_data()

    dtrain = xgb.DMatrix(train_data, label=[0,1,2])
    print(dtrain.num_col())
    print(dtrain.num_row())

    # max_depth： 树的最大深度。缺省值为6，取值范围为：[1,∞]
    # eta：为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。
    # eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0
    # .3，取值范围为：[0, 1]
    # silent：取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时信息。缺省值为0
    # objective： 定义学习任务及相应的学习目标，“binary: logistic” 表示二分类的逻辑回归问题，输出为概率。

    param = {'max_depth': 2, 'eta': 1, 'silent': 0, 'obj': 'binary:logistic'}
    print(param)
    num_round = 2
    bst = xgb.train({}, dtrain)  # dtrain是训练数据集
    train_preds = bst.predict(dtrain)
    train_predictions = [round(value) for value in train_preds]
    y_train = dtrain.get_label()  # 值为输入数据的第一行
    train_accuracy = accuracy_score(y_train, train_predictions)
    print("Train Accuary: %.2f%%" % (train_accuracy * 100.0))

    # fit model no training data
    # model = XGBClassifier()
    # eval_set = [(X_test, y_test)]
    # model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
    # # make predictions for test data
    # y_pred = model.predict(X_test)
    # predictions = [round(value) for value in y_pred]
    # # evaluate predictions
    # accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))




def scale_data(train_df,max_limit,min_limit,labels=['happiness','id'],test_df=None):
    if(test_df is None):
        df=train_df
    else:
        df=pd.concat([train_df,test_df],axis=0,ignore_index=True)
        df.reset_index(drop=True,inplace=True)

    max_min_dic={}
    del_col_names=[]
    amend_col_names=[]
    for col_name in df.columns:
        if(col_name in labels):
            continue
        else:
            col_data=np.array(df[col_name])
            max_value=max(col_data)
            min_value=min(col_data)
            diff=max_value-min_value
            if(diff==0):
                del_col_names.append(col_name)
            else:
                amend_col_names.append(col_name)
                max_min_dic[col_name]=(max_value,min_value,diff)

    for col_name in amend_col_names:
        max_value=max_min_dic[col_name][0]
        min_value=max_min_dic[col_name][1]
        diff=max_min_dic[col_name][2]

        s=train_df[col_name]
        s=s.map(lambda x:(x-min_value)/diff*(max_limit-min_limit)+min_limit)
        train_df[col_name]=s

        if(test_df is not None):
            s = test_df[col_name]
            s = s.map(lambda x: (x - min_value) / (max_value - min_value) * (max_limit - min_limit) + min_limit)
            test_df[col_name] = s

    train_df.drop(columns=del_col_names,inplace=True)
    test_df.drop(columns=del_col_names,inplace=True)

    return train_df,test_df








if __name__ == '__main__':
    va=cal_periodic_value(100, 2, percent=0.05)
    print(va)