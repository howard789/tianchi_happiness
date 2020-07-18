from sklearn import linear_model
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np

from tianchi_happiness.analyze_linear import combine
df=combine.get_combine_raw(submit=False)


oof_cb=df['cb_num']
oof_xgb=df['xgb_num']
oof_lgb=df['lgb_num']

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

print("CV score: {:<8.8f}".format(mean_squared_error(oof_stack, y_train_)))
