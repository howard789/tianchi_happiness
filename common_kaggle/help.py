import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
oof_lgb=[2.1,2.2,2.3]
oof_xgb=[3.1,3.2,3.3]
oof_cb =[4.1,4.2,4.3]
y_train=np.array([2,3,4])
predictions_lgb =[4.1,4.2,4.3]
predictions_xgb=[5.1,5.2,5.3]
predictions_cb=[6.1,6.2,6.3]
# 将lgb和xgb和ctb的结果进行stacking
train_stack = np.vstack([oof_lgb, oof_xgb, oof_cb]).transpose()

test_stack = np.vstack([predictions_lgb, predictions_xgb, predictions_cb]).transpose()


folds_stack = RepeatedKFold(n_splits=3, n_repeats=2, random_state=2018)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, y_train)):
    print("fold {}".format(fold_))

    trn_data, trn_y = train_stack[trn_idx,:], y_train[trn_idx]
    val_data, val_y = train_stack[val_idx], y_train[val_idx]

    clf_3 = linear_model.BayesianRidge()
    # clf_3 =linear_model.Ridge()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10



print(predictions)
print("CV score: {:<8.8f}".format(mean_squared_error(oof_stack, y_train)))
