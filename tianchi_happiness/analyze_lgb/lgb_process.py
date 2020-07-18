import lightgbm as lgb
import pandas as pd
from common_kaggle import common_util
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
from tianchi_happiness import deal_data

label = 'happiness'
train_df, test_df, test_df_with_id ,features= deal_data.getData(submit=False)






train_x, train_y = common_util.split_df_to_array(train_df, label)
test_x, test_y = common_util.split_df_to_array(test_df, label)

# combine all
COMBINE_ALL = False
if (COMBINE_ALL):
    from tianchi_happiness import deal_data

    df = deal_data.get_combine_all(submit=False)
    train_df, test_df = common_util.split_train_test(df, test_size=0.2)
    label = 'true'
    train_x, train_y = common_util.split_df_to_array(train_df, label)
    test_x, test_y = common_util.split_df_to_array(test_df, label)
    # 正确率60.62%

kfolder = KFold(n_splits=5, shuffle=True, random_state=2019)
# oof_cb = np.zeros(len(train_x))
# predictions_cb = np.zeros(len(test_x))
# kfold = kfolder.split(train_x, train_y)
fold_ = 0

param = {'boosting_type': 'gbdt',
         'num_leaves': 20,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 6,
         'learning_rate': 0.01,
         "min_child_samples": 30,

         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8,
         "bagging_seed": 11,
         "metric": 'mse',
         "lambda_l1": 0.1,
         "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train_x))
predictions_lgb = np.zeros(len(test_x))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
    print("fold n°{}".format(fold_ + 1))
    # print(trn_idx)
    # print(".............x_train.........")
    # print(X_train[trn_idx])
    #  print(".............y_train.........")
    #  print(y_train[trn_idx])
    trn_data = lgb.Dataset(train_x[trn_idx], train_y[trn_idx])

    val_data = lgb.Dataset(train_x[val_idx], train_y[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(train_x[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits

print(predictions_lgb)

pred = []
for i in range(len(predictions_lgb)):
    pred.append(round(predictions_lgb[i]))
print(predictions_lgb)
print(pred)
common_util.cal_correct_rate(test_y, pred)
# 正确率62.18%
#

#---------------特征重要性
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


df = pd.DataFrame(features, columns=['feature'])

print(len(features))
importance=list(clf.feature_importance())
print(len(importance))

df['importance']=list(clf.feature_importance())
df = df.sort_values(by='importance',ascending=False)
plt.figure(figsize=(14,28))
sns.barplot(x="importance", y="feature", data=df.head(50))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.show()













if (COMBINE_ALL == False):
    save_df = pd.DataFrame()
    save_df['id'] = test_df_with_id['id']
    save_df['pred'] = predictions_lgb
    save_df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/num_pred_lgb.csv')

    save_df2 = pd.DataFrame()
    save_df2['id'] = test_df_with_id['id']
    save_df2['pred'] = pred
    save_df2.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/pred_lgb.csv')
