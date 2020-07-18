import xgboost as xgb
from mSklearn import m_xgboost
import pandas as pd
from common_kaggle import common_util
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
label = 'happiness'
train_df = pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/train_df_minus.csv')
test_df = pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/test_df_minus.csv')
# train_df_bk=train_df.copy()
# test_df_bk=test_df.copy()


train_df[label] = train_df[label].apply(lambda x: 0 if x == 5 else x)
test_df[label] = test_df[label].apply(lambda x: 0 if x == 5 else x)
train_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)

train_x, train_y = common_util.split_df_to_array(train_df, label)
test_x, test_y = common_util.split_df_to_array(test_df, label)


m_xgboost.try_best_param_3(train_x, train_y, test_x, test_y)