import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, RepeatedKFold
from scipy import sparse
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
from datetime import datetime

from common_kaggle import common_util
from common_kaggle import mathUtil
from tianchi_happiness import deal_data
train_df,test_df,test_df_with_id,features=deal_data.getData(submit=False)
train=train_df
test=test_df
test_sub=pd.DataFrame()
test_sub['id']=test_df_with_id['id']
v=test['happiness']
test_sub['true']=test['happiness']

X_train,y_train=common_util.split_df_to_array(train,'happiness')
X_test,y_test=common_util.split_df_to_array(test,'happiness')
X_train_=pd.DataFrame(X_train)
y_train_=pd.DataFrame(y_train)
X_test_=pd.DataFrame(X_test)
y_test_=pd.DataFrame(y_test)

#自定义评价函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label,preds)
    return 'myFeval',score


##### xgb

xgb_params = {"booster": 'gbtree', 'eta': 0.005, 'max_depth': 5, 'subsample': 0.7,
              'colsample_bytree': 0.8, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params, feval=myFeval)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

predictions_xgb2 = clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit)

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train_)))
rate1,incorrect_list1=common_util.cal_correct_rate(np.array(test_sub['true']),predictions_xgb,None,True,'xgb')
rate2,incorrect_list2=common_util.cal_correct_rate(np.array(test_sub['true']),predictions_xgb2,None,True,'xgb2')
mse_value1=common_util.cal_mse(test_sub['true'],predictions_xgb,'xgb')
mse_value2=common_util.cal_mse(test_sub['true'],predictions_xgb2,'xgb2')

results_dic={}
results_dic['xgb1_num_correct_rate']=rate1
results_dic['xgb2_num_correct_rate']=rate2
results_dic['xgb1_num_mse']=mse_value1
results_dic['xgb2_num_mse']=mse_value2


##### lgb

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
oof_lgb = np.zeros(len(X_train_))
predictions_lgb = np.zeros(len(X_test_))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n°{}".format(fold_ + 1))
    # print(trn_idx)
    # print(".............x_train.........")
    # print(X_train[trn_idx])
    #  print(".............y_train.........")
    #  print(y_train[trn_idx])
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])

    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=100)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
predictions_lgb2 = clf.predict(X_test, num_iteration=clf.best_iteration)

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, y_train_)))

rate1,incorrect_list1=common_util.cal_correct_rate(np.array(test_sub['true']),predictions_lgb,None,True,'lgb')
rate2,incorrect_list2=common_util.cal_correct_rate(np.array(test_sub['true']),predictions_lgb2,None,True,'lgb2')
mse_value1=common_util.cal_mse(test_sub['true'],predictions_lgb,'lgb')
mse_value2=common_util.cal_mse(test_sub['true'],predictions_lgb2,'lgb2')

results_dic['lgb1_num_correct_rate']=rate1
results_dic['lgb2_num_correct_rate']=rate2
results_dic['lgb1_num_mse']=mse_value1
results_dic['lgb2_num_mse']=mse_value2



from catboost import Pool, CatBoostRegressor
# cat_features=[0,2,3,10,11,13,15,16,17,18,19]
from sklearn.model_selection import train_test_split


#X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train_, y_train_, test_size=0.3, random_state=2019)
# train_pool = Pool(X_train_s, y_train_s,cat_features=[0,2,3,10,11,13,15,16,17,18,19])
# val_pool = Pool(X_test_s, y_test_s,cat_features=[0,2,3,10,11,13,15,16,17,18,19])
# test_pool = Pool(X_test_ ,cat_features=[0,2,3,10,11,13,15,16,17,18,19])


kfolder = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_cb = np.zeros(len(X_train_))
predictions_cb = np.zeros(len(X_test_))
kfold = kfolder.split(X_train_, y_train_)
fold_=0
#X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train, y_train, test_size=0.3, random_state=2019)
for train_index, vali_index in kfold:
    print("fold n°{}".format(fold_))
    fold_=fold_+1
    k_x_train = X_train[train_index]
    k_y_train = y_train[train_index]
    k_x_vali = X_train[vali_index]
    k_y_vali = y_train[vali_index]
    cb_params = {
         'n_estimators': 100000,
         'loss_function': 'RMSE',
         'eval_metric':'RMSE',
         'learning_rate': 0.05,
         'depth': 5,
         'use_best_model': True,
         'subsample': 0.6,
         'bootstrap_type': 'Bernoulli',
         'reg_lambda': 3
    }
    model_cb = CatBoostRegressor(**cb_params)
    #train the model
    model_cb.fit(k_x_train, k_y_train,eval_set=[(k_x_vali, k_y_vali)],verbose=100,early_stopping_rounds=50)
    oof_cb[vali_index] = model_cb.predict(k_x_vali, ntree_end=model_cb.best_iteration_)
    predictions_cb += model_cb.predict(X_test_, ntree_end=model_cb.best_iteration_) / kfolder.n_splits

predictions_cb2 = model_cb.predict(X_test_, ntree_end=model_cb.best_iteration_)



print("CV score: {:<8.8f}".format(mean_squared_error(oof_cb, y_train_)))

rate1,incorrect_list1=common_util.cal_correct_rate(np.array(test_sub['true']),predictions_cb,None,True,'cat')
rate2,incorrect_list2=common_util.cal_correct_rate(np.array(test_sub['true']),predictions_cb2,None,True,'cat2')
mse_value1=common_util.cal_mse(test_sub['true'],predictions_cb,'cat')
mse_value2=common_util.cal_mse(test_sub['true'],predictions_cb2,'cat2')

results_dic['cb1_num_correct_rate']=rate1
results_dic['cb2_num_correct_rate']=rate2
results_dic['cb1_num_mse']=mse_value1
results_dic['cb2_num_mse']=mse_value2

print('---------------------------------------------------------------------------------')




print('-------xgb one-hot--------------------------------------------------------------------------')

from mSklearn import m_xgboost
params = {'learning_rate': 0.4,
          'max_depth': 5,  # 构建树的深度，越大越容易过拟合
          'num_boost_round': 1,
          'objective': 'multi:softprob',  # 多分类的问题
          'random_state': 7,
          'silent': 0,
          'num_class': 5,  # 类别数，与 multisoftmax 并用
          'eta': 0.  # 为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
          }
onehot_xgb_predictions,accuracy=m_xgboost.run(X_train, y_train, X_test, y_test,params,False)
rate,incorrect_list=common_util.cal_correct_rate(np.array(test_sub['true']),onehot_xgb_predictions,None,True,'onehot_xgb_predictions')
mse=common_util.cal_mse(test_sub['true'],onehot_xgb_predictions,'onehot_xgb_predictions')


results_dic['xgb_oneHot_correct_rate']=rate
results_dic['xgb_oneHot_mse']=mse


print('------lgb round---------------------------------------------------------------------------')

predictions_lgb_round = []
for i in range(len(predictions_lgb)):
    predictions_lgb_round.append(round(predictions_lgb[i]))
rate,incorrect_list=common_util.cal_correct_rate(np.array(test_sub['true']),predictions_lgb_round,None,True,'predictions_lgb_round')
mse=common_util.cal_mse(test_sub['true'],predictions_lgb_round,'predictions_lgb_round')

results_dic['lgb_round_correct_rate']=rate
results_dic['lgb_round_mse']=mse

print('------cat round---------------------------------------------------------------------------')


predictions_cb_round = []
for i in range(len(predictions_cb)):
    predictions_cb_round.append(round(predictions_cb[i]))
rate,incorrect_list=common_util.cal_correct_rate(np.array(test_sub['true']),predictions_cb_round,None,True,'predictions_cb_round')
mse=common_util.cal_mse(test_sub['true'],predictions_cb_round,'predictions_cb_round')
results_dic['cb_round_correct_rate']=rate
results_dic['cb_round_mse']=mse

print('-----combine---------------------------------------------------------------------------')
combine_df1=test_sub.copy()

combine_df1['xgb_num'] =predictions_xgb
combine_df1['xgb'] = onehot_xgb_predictions

combine_df1['cb_round'] =predictions_cb_round
combine_df1['cb_num'] =predictions_cb
combine_df1['lgb_round'] =predictions_lgb_round
combine_df1['lgb_num'] =predictions_lgb

combine_df2=combine_df1.copy()
combine_df2['diff_cb_round'] = combine_df2['cb_num'] - combine_df2['cb_round']
combine_df2['diff_xgb_num'] = combine_df2['xgb_num'] - combine_df2['xgb']
combine_df2['diff_lgb_round'] = combine_df2['lgb_num'] - combine_df2['lgb_round']


print('-----combine df1 --------------------------------------------------------------------------')

train_df, test_df = common_util.split_train_test(combine_df1, test_size=0.2)
train_x_f, train_y_f = common_util.split_df_to_array(train_df, 'true')
test_x_f, test_y_f = common_util.split_df_to_array(test_df,  'true')

linear_xgb_predictions=m_xgboost.run_linear(train_x_f, train_y_f,test_x_f, test_y_f,params=None)
rate,incorrect_list=common_util.cal_correct_rate(test_y_f,linear_xgb_predictions,None,True,'linear_xgb_predictions')
mse=common_util.cal_mse(test_y_f,linear_xgb_predictions,'linear_xgb_predictions')

results_dic['xgb_combine_df1_correct_rate']=rate
results_dic['xgb_combine_df1_mse']=mse

# round之后再测一次mse
linear_xgb_predictions_round=common_util.round_arr(linear_xgb_predictions)
mse=common_util.cal_mse(test_y_f,linear_xgb_predictions_round,'linear_xgb_predictions_round')

results_dic['xgb_combine_df1_round_mse']=mse

print('-----combine df2 --------------------------------------------------------------------------')

train_df, test_df = common_util.split_train_test(combine_df2, test_size=0.2)
train_x_f, train_y_f = common_util.split_df_to_array(train_df, 'true')
test_x_f, test_y_f = common_util.split_df_to_array(test_df,  'true')

linear_xgb_predictions=m_xgboost.run_linear(train_x_f, train_y_f,test_x_f, test_y_f,params=None)
rate,incorrect_list=common_util.cal_correct_rate(test_y_f,linear_xgb_predictions,None,True,'linear_xgb_predictions df2')
mse=common_util.cal_mse(test_y_f,linear_xgb_predictions,'linear_xgb_predictions df2')

results_dic['xgb_combine_df2_correct_rate']=rate
results_dic['xgb_combine_df2_mse']=mse

# round之后再测一次mse
linear_xgb_predictions_round=common_util.round_arr(linear_xgb_predictions)
mse=common_util.cal_mse(test_y_f,linear_xgb_predictions_round,'linear_xgb_predictions_round df2')

results_dic['xgb_combine_df2_round_mse']=mse

print('---------------------------------------------------------------------------------')
from sklearn import linear_model

# 将lgb和xgb和ctb的结果进行stacking
train_stack = np.vstack([oof_lgb, oof_xgb, oof_cb]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb, predictions_cb]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2018)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]

    clf_3 = linear_model.BayesianRidge()
    # clf_3 =linear_model.Ridge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10
predictions2 = clf_3.predict(test_stack)
print("CV score: {:<8.8f}".format(mean_squared_error(oof_stack, y_train_)))


combine_linear_rate1,xx=common_util.cal_correct_rate(np.array(test_sub['true']),predictions,None,True,'final')
combine_linear_rate2,xx=common_util.cal_correct_rate(np.array(test_sub['true']),predictions2,None,True,'fina2')
mse1=common_util.cal_mse(test_sub['true'],predictions,'final')
mse2=common_util.cal_mse(test_sub['true'],predictions2,'fina2')
results_dic['combine_linear_rate1']=combine_linear_rate1
results_dic['combine_linear_rate2']=combine_linear_rate2
results_dic['combine_linear_mse1']=mse1
results_dic['combine_linear_mse2']=mse2

print(results_dic)
for key, value in results_dic.items():
    print("{}:{:.4f}".format(key, value))













# result=list(predictions)
# result=list(map(lambda x: x + 1, result))
# test_sub["happiness"]=result
# print(result)
# common_util.cal_correct_rate(np.array(test_sub['true']),result,None,True)
# mathUtil.mse_df(test_sub,'true')
#
#
# test_sub.to_csv("submit_20190515.csv", index=False)

