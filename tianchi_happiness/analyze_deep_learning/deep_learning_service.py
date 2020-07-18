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

def cal_correct_rate(y_true_array,predict,id_list):
    predit_list = []
    for i in range(predict.shape[0]):
        pred_y = int(np.argmax(predict[i]))
        predit_list.append(pred_y)
    y_predict_array = np.array(predit_list)
    rate, incorrect_list= common_util.cal_correct_rate(y_true_array, y_predict_array,id_list)
    return rate, incorrect_list

def cal_correct_rate_round(y_true_array,predict,id_list):
    predit_list = []
    for i in range(predict.shape[0]):
        pred_y = round(predict[i])
        predit_list.append(pred_y)
    y_predict_array = np.array(predit_list)
    rate, incorrect_list= common_util.cal_correct_rate(y_true_array, y_predict_array,id_list)
    return rate, incorrect_list

# def wrong_run():
#     label = 'happiness'
#     train_df_with_id, test_df_with_id = deal_data.get_train_data_and_deal_data(submit=False, data_size=1000,test_size=0.2,keep_id=True)
#
#     #最终验证的
#     test_df=test_df_with_id.drop(columns=['id'])
#     test_x, test_y = common_util.split_df_to_array(test_df, label)
#
#     # 过程验证的
#     train_df_no_id=train_df_with_id.drop(columns=['id'])
#     test_tmp_x, test_tmp_y = common_util.split_df_to_array(train_df_no_id, label)
#     all_id_list=train_df_with_id['id'].tolist()
#
#     keep_id_list=all_id_list
#     rate_result = []
#     num_classes = 5
#     dims = 214
#     model = mModels.get_model_9(dims, num_classes)
#     batch_num=32
#     total_train_count=1000
#
#     train_count=0
#     while(True):
#
#         if(train_count>total_train_count/2):
#             model=mModels.amend_adam_rate(model,0.001)
#         selected_train_df_with_id =deal_data.get_selected_train_data(train_df_with_id, label,keep_id_list)
#         train_df=selected_train_df_with_id.drop(columns=['id'])
#         train_x, train_y = common_util.split_df_to_array(train_df, label)
#         train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
#         train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
#         train_y = tf.one_hot(train_y, depth=5)
#         # 将x,y合并,并分成batch
#         train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
#         train_data = train_data.batch(batch_num)
#
#         for step, (x, y) in enumerate(train_data):
#             with tf.GradientTape() as type:
#                 x = tf.reshape(x, (-1, dims))
#             model.fit(x=x, y=y)
#         predict = model.predict(test_tmp_x)
#         correct_rate, incorrect_list = cal_correct_rate(test_tmp_y, predict,all_id_list)
#         rate_result.append(correct_rate)
#         keep_id_list=incorrect_list #错误的继续训练
#         train_count +=1
#         if(train_count>total_train_count):
#             break
#     #最终测试
#     print(rate_result)
#     predict = model.predict(test_x)
#     correct_rate, incorrect_list = cal_correct_rate(test_y, predict,None)
#     print('final:',correct_rate)




def scale_data2(df,label):

    del_col_names=[]
    for column in df.columns:
        if(column==label):
            continue
        else:
            col_data=np.array(df[column])
            max_value=max(col_data)
            min_value=min(col_data)
            diff=max_value-min_value
            print(max_value)
            print(min_value)
            print(diff)
            print('-----------------------')
            if(diff==0):
                del_col_names.append(column)
            else:
                for i in range(df.shape[0]):
                    v=df.loc[i,column]
                    # v2=(v-min_value)/diff
                    v2=2*(v-min_value)/diff-1
                    df.loc[i,column]=v2
    return df,del_col_names




def standard_run():
    label = 'happiness'
    #
    data_size=-1
    load=False

    load=True
    #

    if(load==False):
        train_df, test_df = deal_data.get_train_data_and_deal_data(submit=False, data_size=data_size,keep_id=True)
        train_df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/train_df_minus.csv')
        test_df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/test_df_minus.csv')
    else:
        train_df=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/train_df_minus.csv')
        test_df=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/test_df_minus.csv')



    train_df, test_df=common_util.scale_data(train_df=train_df,max_limit=1,min_limit=-1,labels=[label,'id'],test_df=test_df)

    if(True):
        train_df,del_col_names=scale_data2(train_df,label)
        test_df,xx=scale_data2(test_df,label)
        train_df.drop(columns=del_col_names, inplace=True)
        test_df.drop(columns=del_col_names, inplace=True)


    train_df_with_id_=train_df.copy()
    test_df_with_id=test_df.copy()
    #
    train_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)

    train_x, train_y = common_util.split_df_to_array(train_df, label)
    test_x, test_y = common_util.split_df_to_array(test_df, label)


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

    # model = mModels.get_model_11(dims,num_classes)
    model = mModels.get_model_13(dims,num_classes)
    # model = mModels.get_model_10(dims,num_classes)
    # model = mModels.get_model_8(dims,num_classes)
    # model = mModels.get_model_9(dims,num_classes)
    # model = mModels.get_model_4(dims,num_classes)
    # model = mModels.get_model_5(dims,num_classes)
    # model = mModels.get_model_7(dims,num_classes)

    # model = mModels.get_model_1(num_classes, dims, dims)

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

        correct_rate,incorrect_list = cal_correct_rate(test_y, predict,None)
        rate_result.append((i, correct_rate))

    print(rate_result)
    print('batch_num:',batch_num)





if __name__ == '__main__':
    import sys
    standard_run()
    # wrong_run()
    sys.exit()