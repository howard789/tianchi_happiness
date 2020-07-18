import xgboost as xgb
from mSklearn import m_xgboost
import pandas as pd
from common_kaggle import common_util
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
from common_kaggle import mathUtil

# df_tmp=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/combine_all.csv')
#
# df=pd.DataFrame()
# df[label]=df_tmp['true']
# df['cb_num']=df_tmp['cb_num']
# df['cb']=df_tmp['cb']
# df['xgb_num']=df_tmp['xgb_num']
# df['xgb']=df_tmp['xgb']
# df['lgb_num']=df_tmp['lgb_num']
# df['lgb']=df_tmp['lgb']
from tianchi_happiness import deal_data
from tianchi_happiness.analyze_linear import combine
label = 'true'
# df=combine.get_combine_raw(submit=False)

# mathUtil.mse_df(df,label)
df=combine.get_combine_amended(submit=False)
#
#

train_df, test_df = common_util.split_train_test(df, test_size=0.2)
train_x, train_y = common_util.split_df_to_array(train_df, label)
test_x, test_y = common_util.split_df_to_array(test_df, label)

try_best_param=False

if(try_best_param==False):
    params = {'learning_rate': 0.4,
              'max_depth': 5,  # 构建树的深度，越大越容易过拟合
              'num_boost_round': 1,
              'objective': 'multi:softprob',  # 多分类的问题
              'random_state': 7,
              'silent': 0,
              'num_class': 5,  # 类别数，与 multisoftmax 并用
              'eta': 0.  # 为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
              }


    predictions=m_xgboost.run(train_x, train_y, test_x, test_y,params,False)
    train_x['xgb_all']=predictions

    # df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/xgb_all.csv')
    # Accuracy: 63.06%

else:
    from mSklearn import m_xgboost
    m_xgboost.try_best_param_3(train_x, train_y, test_x, test_y)