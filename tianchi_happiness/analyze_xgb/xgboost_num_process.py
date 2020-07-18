import xgboost as xgb
from mSklearn import m_xgboost
import pandas as pd
from common_kaggle import common_util
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
from sklearn.metrics import mean_squared_error

from tianchi_happiness import deal_data
from tianchi_happiness.analyze_linear import combine
train_df,test_df,test_df_with_id,features=deal_data.getData(submit=False)
label = 'happiness'
X_train, y_train = common_util.split_df_to_array(train_df, label)
X_test, test_y = common_util.split_df_to_array(test_df, label)

#combine all
COMBINE_ALL=True
if(COMBINE_ALL):
    # df=combine.get_combine_raw(submit=False)
    # 正确率64.06% MSE 0.4625


    df=combine.get_combine_amended(submit=False)
    #正确率61.88% MSE 0.5844 正确率61.88% MSE 0.5188


    train_df, test_df = common_util.split_train_test(df, test_size=0.2)
    label = 'true'
    X_train, y_train = common_util.split_df_to_array(train_df, label)
    X_test, test_y = common_util.split_df_to_array(test_df, label)
    #正确率64.38%




xgb_params = {"booster": 'gbtree', 'eta': 0.005, 'max_depth': 5, 'subsample': 0.7,
              'colsample_bytree': 0.8, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(X_train))
predictions_xgb = np.zeros(len(X_test))

#自定义评价函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label,preds)
    return 'myFeval',score


for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params, feval=myFeval)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))

# CV score: 0.46450970
pred=[]
for i in range(len(predictions_xgb)):
    pred.append(round(predictions_xgb[i]))
print(predictions_xgb)
print(pred)
common_util.cal_correct_rate(test_y,pred)
common_util.cal_mse(test_y,pred)
# 正确率62.24%


# save_df=pd.DataFrame()
# save_df['id']=test_df_with_id['id']
# save_df['pred']=predictions_xgb
# save_df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/num_pred_xgb.csv')
#

# save_df2=pd.DataFrame()
# save_df2['id']=test_df_with_id['id']
# save_df2['pred']=pred
# save_df2.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/pred_cb.csv')
