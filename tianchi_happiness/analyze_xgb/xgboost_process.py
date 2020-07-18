import xgboost as xgb
from mSklearn import m_xgboost
import pandas as pd
from common_kaggle import common_util
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np


from tianchi_happiness import deal_data
train_df,test_df,test_df_with_id,features=deal_data.getData(submit=False)


label = 'happiness'
train_x, train_y = common_util.split_df_to_array(train_df, label)
test_x, test_y = common_util.split_df_to_array(test_df, label)


params = {'learning_rate': 0.4,
          'max_depth': 5,  # 构建树的深度，越大越容易过拟合
          'num_boost_round': 1,
          'objective': 'multi:softprob',  # 多分类的问题
          'random_state': 7,
          'silent': 0,
          'num_class': 5,  # 类别数，与 multisoftmax 并用
          'eta': 0.  # 为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
          }
predictions,accuracy=m_xgboost.run(train_x, train_y, test_x, test_y,params,False)
common_util.cal_correct_rate(test_y,predictions,None,False,'xgb')
common_util.cal_mse(test_y,predictions,'xgb')


# save_df=pd.DataFrame()
# save_df['id']=test_df_with_id['id']
# save_df['pred']=predictions
# save_df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/pred_xgb.csv')
# Accuracy: 63.06%