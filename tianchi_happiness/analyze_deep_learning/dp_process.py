from tianchi_happiness import deal_data
import tensorflow.keras.utils as utils
from common_kaggle import common_util
import numpy as np
import pandas as pd
import scipy.special as spc
import os
from tianchi_happiness.analyze_deep_learning import mModels
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.layers as layers
from tianchi_happiness.analyze_deep_learning import deep_learning_service

from tianchi_happiness import deal_data

train_df,test_df,test_df_with_id=deal_data.getData(submit=False)

label = 'happiness'
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
    # 正确率




train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
train_y = tf.one_hot(train_y, depth=5)

test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)

# 将x,y合并,并分成batch
train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
batch_num=32
train_data = train_data.batch(batch_num)
num_classes=5
dims = train_x.shape[1]


model = mModels.get_model_13(dims,num_classes)
rate_result = []
train_num=2
for i in range(train_num):
    if(i>=train_num/2):
        model=mModels.amend_adam_rate(model,0.001)
    for step, (x, y) in enumerate(train_data):
        with tf.GradientTape() as type:
            x = tf.reshape(x, (-1, dims))
        model.fit(x=x, y=y)

predict = model.predict(test_x)

# correct_rate,incorrect_list = deep_learning_service.cal_correct_rate(test_y, predict,None)
# rate_result.append((i, correct_rate))
#
# print(predict)
# print(rate_result)
pred=[]
for i in range(len(predict)):
    pred.append(np.argmax(predict[i]))
# print(predict)
print(pred)
common_util.cal_correct_rate(test_y,pred)

if (COMBINE_ALL == False):
    save_df=pd.DataFrame()
    save_df['id']=test_df_with_id['id']
    save_df['pred']=pred
    save_df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/pred_dp.csv')