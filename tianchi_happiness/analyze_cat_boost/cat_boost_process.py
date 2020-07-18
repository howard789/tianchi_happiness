from catboost import Pool, CatBoostRegressor
import pandas as pd
from common_kaggle import common_util
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
from tianchi_happiness import deal_data

train_df,test_df,test_df_with_id=deal_data.getData(submit=False)

label = 'happiness'
train_x, train_y = common_util.split_df_to_array(train_df, label)
test_x, test_y = common_util.split_df_to_array(test_df, label)


#combine all
COMBINE_ALL=True
if(COMBINE_ALL):
    from tianchi_happiness import deal_data
    df=deal_data.get_combine_all(submit=False)
    train_df, test_df = common_util.split_train_test(df, test_size=0.2)
    label = 'true'
    train_x, train_y = common_util.split_df_to_array(train_df, label)
    test_x, test_y = common_util.split_df_to_array(test_df, label)
    #正确率64.38%






kfolder = KFold(n_splits=5, shuffle=True, random_state=2019)
oof_cb = np.zeros(len(train_x))
predictions_cb = np.zeros(len(test_x))
kfold = kfolder.split(train_x, train_y)
fold_=0
#X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train, y_train, test_size=0.3, random_state=2019)

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




for train_index, vali_index in kfold:
    print("fold n°{}".format(fold_))
    fold_=fold_+1
    k_x_train = train_x[train_index]
    k_y_train = train_y[train_index]
    k_x_vali = train_x[vali_index]
    k_y_vali = train_y[vali_index]

    #train the model
    model_cb.fit(k_x_train, k_y_train,eval_set=[(k_x_vali, k_y_vali)],verbose=100,early_stopping_rounds=50)
    oof_cb[vali_index] = model_cb.predict(k_x_vali, ntree_end=model_cb.best_iteration_)
    predictions_cb += model_cb.predict(test_x, ntree_end=model_cb.best_iteration_) / kfolder.n_splits
    print()

predictions_cb=model_cb.predict(test_x)



pred=[]
for i in range(len(predictions_cb)):
    pred.append(round(predictions_cb[i]))
print(predictions_cb)
print(pred)
common_util.cal_correct_rate(test_y,pred)
# print("CV score: {:<8.8f}".format(mean_squared_error(oof_cb, train_y)))
# 正确率62.30%
















print('存档区..................')






if(COMBINE_ALL==False):
    save_df=pd.DataFrame()
    save_df['id']=test_df_with_id['id']
    save_df['pred']=predictions_cb
    save_df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/num_pred_cb.csv')


    save_df2=pd.DataFrame()
    save_df2['id']=test_df_with_id['id']
    save_df2['pred']=pred
    save_df2.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/pred_cb.csv')


    save_df3=pd.DataFrame()
    save_df3['id']=test_df_with_id['id']
    save_df3['true']=test_df_with_id[label]
    save_df3.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/true_y.csv')