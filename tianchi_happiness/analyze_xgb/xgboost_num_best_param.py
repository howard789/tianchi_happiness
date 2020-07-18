import xgboost as xgb
from mSklearn import m_xgboost
import pandas as pd
from common_kaggle import common_util
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
from sklearn.metrics import mean_squared_error
from tianchi_happiness.analyze_linear import combine
from tianchi_happiness import deal_data
from common_kaggle import data_util
from common_kaggle import datetimeUtil

def get_data():
    train_df, test_df, test_df_with_id, features = deal_data.getData(submit=False)

    data_util.pd_drop_row_after(train_df, 3000)
    data_util.pd_drop_row_after(test_df, 600)
    data_util.pd_drop_row_after(test_df_with_id, 600)

    label = 'happiness'
    X_train, y_train = common_util.split_df_to_array(train_df, label)
    X_test, test_y = common_util.split_df_to_array(test_df, label)
    return X_train, y_train, X_test, test_y


def get_combine_data():
    df = combine.get_combine_raw(submit=False)
    df=combine.get_combine_amended(submit=False)

    train_df, test_df = common_util.split_train_test(df, test_size=0.2)

    data_util.pd_drop_row_after(train_df, 10)
    data_util.pd_drop_row_after(test_df, 2)


    test_df_with_id = test_df
    label = 'true'
    X_train, y_train = common_util.split_df_to_array(train_df, label)
    X_test, test_y = common_util.split_df_to_array(test_df, label)
    # 正确率64.38%
    return X_train, y_train, X_test, test_y


def get_param_list():
    Init = True
    if (Init):
        multiple=2
        eta_list = data_util.generate_list(start_num=0., end_num=0.2, step=0.01*multiple, round_d=None)
        print('eta_list',len(eta_list))
        # eta[默认0.3] 0.01-0.2 就是学习率
        max_depth_list = data_util.generate_list(start_num=2, end_num=10, step=1*multiple, round_d=None)
        # 默认6典型值：3 - 10
        print('max_depth_list', len(max_depth_list))
        gamma_list = data_util.generate_list(start_num=0, end_num=0.1, step=0.01*multiple, round_d=None)
        # 默认0 参数的值越大，算法越保守
        print('gamma_list', len(gamma_list))
        subsample_list = data_util.generate_list(start_num=0.5, end_num=1, step=0.1*multiple, round_d=None)
        # 默认1典型值：0.5-1 这个参数控制对于每棵树，随机采样的比例。
        print('subsample_list', len(subsample_list))
        colsample_bytree_list = data_util.generate_list(start_num=0.5, end_num=1, step=0.1*multiple, round_d=None)
        # 默认1 典型值：0.5-1 最大值不能超过1的样子
        print('colsample_bytree_list', len(colsample_bytree_list))
        # objective:默认reg:linear
        # multi: softmax 你还需要多设一个参数：num_class(类别数目)。

        # eval_metric
        # seed 随机数的种子 设置它可以复现随机数据的结果，也可以用于调整参数

        n_estimators_list = data_util.generate_list(start_num=200, end_num=2000, step=200*multiple, round_d=None)
        print('n_estimators_list', len(n_estimators_list))

    total_params_list = []
    # for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        for gamma in gamma_list:
            for subsample in subsample_list:
                for colsample_bytree in colsample_bytree_list:
                    for eta in eta_list:
                        xgb_params = {"booster": 'gbtree', 'eta': eta, 'max_depth': max_depth, 'gamma': gamma,
                                      'subsample': subsample,
                                      'colsample_bytree': colsample_bytree, 'objective': 'reg:linear',
                                      'eval_metric': 'rmse',"learning_rate": eta}

                        # xgb_params = {"booster": 'gbtree', 'eta': 0.005, 'max_depth': 5, 'subsample': 0.7,
                        #               'colsample_bytree': 0.8, 'objective': 'reg:linear', 'eval_metric': 'rmse',
                        #               'silent': True, 'nthread': 8}
                        total_params_list.append(xgb_params)
    print('总共的param组合', len(total_params_list))
    return total_params_list


# 自定义评价函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label, preds)
    return 'myFeval', score


def try_best_param_impl(X_train, y_train, X_test, test_y, params):
    folds = KFold(n_splits=5, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(X_train))
    predictions_xgb = np.zeros(len(X_test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        # print("fold n°{}".format(fold_ + 1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                        verbose_eval=100, params=params, feval=myFeval)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

    # print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train)))

    pred = []
    for i in range(len(predictions_xgb)):
        pred.append(round(predictions_xgb[i]))
    print(predictions_xgb)
    print(pred)
    rate, incorrect_list = common_util.cal_correct_rate(test_y, pred)
    mse_value = common_util.cal_mse(test_y, pred)
    return rate, mse_value


if __name__ == '__main__':
    # X_train, y_train, X_test, test_y = get_data()
    X_train, y_train, X_test, test_y=get_combine_data()

    print('X_train.shape',X_train.shape)
    print('y_train.shape',y_train.shape)
    print('X_test.shape',X_test.shape)
    print('test_y.shape',test_y.shape)

    total_params_list = get_param_list()
    max_3_rate = [-np.inf, -np.inf, -np.inf]
    min_3_mse = [np.inf, np.inf, np.inf]
    top_3_params = [None, None, None]
    startTime=datetimeUtil.Now()
    for i in range(len(total_params_list)):
        params=total_params_list[i]
        rate, mse_value = try_best_param_impl(X_train, y_train, X_test, test_y, params)
        index = np.argmax(min_3_mse)
        if (mse_value < min_3_mse[index]):
            min_3_mse[index] = mse_value
            max_3_rate[index] = rate
            top_3_params[index] = params
        datetimeUtil.calTimeLeft(i,len(total_params_list),startTime,datetimeUtil.Now(),i)

    print('完成')
    for i in range(len(top_3_params)):
        print(i)
        print(max_3_rate[i])
        print(min_3_mse[i])
        print(top_3_params[i])
        print('------------------------------------------------------')
