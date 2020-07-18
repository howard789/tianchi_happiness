import pandas as pd
import numpy as np
def get_dic(df,key_col='id',value_col='pred'):
    mDic={}
    for i in range(df.shape[0]):
       key=df.loc[i,key_col]
       value = df.loc[i, value_col]
       mDic[key]=value
    return mDic

def make_combine_data():
    num_pred_cb=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/num_pred_cb.csv')
    pred_cb=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/pred_cb.csv')
    num_pred_xgb=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/num_pred_xgb.csv')
    pred_xgb=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/pred_xgb.csv')
    num_pred_lgb=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/num_pred_lgb.csv')
    pred_lgb=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/pred_lgb.csv')
    pred_dp=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/pred_dp.csv')

    num_pred_cb_dic =get_dic(num_pred_cb)
    pred_cb_dic     =get_dic(pred_cb)
    num_pred_xgb_dic=get_dic(num_pred_xgb)
    pred_xgb_dic    =get_dic(pred_xgb)
    num_pred_lgb_dic=get_dic(num_pred_lgb)
    pred_lgb_dic    =get_dic(pred_lgb)
    pred_dp_dic     =get_dic(pred_dp)

    true_y=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/true_y.csv')
    true_y['cb_num']=''
    true_y['cb']=''
    true_y['xgb_num']=''
    true_y['xgb']=''
    true_y['lgb_num']=''
    true_y['lgb']=''
    true_y['dp']=''

    for i in range(true_y.shape[0]):
        id=true_y.loc[i,'id']
        true_y.loc[i,'cb_num'] = num_pred_cb_dic[id]
        true_y.loc[i,'cb'] = pred_cb_dic[id]
        true_y.loc[i,'xgb_num'] = num_pred_xgb_dic[id]
        true_y.loc[i,'xgb'] = pred_xgb_dic[id]
        true_y.loc[i,'lgb_num'] = num_pred_lgb_dic[id]
        true_y.loc[i,'lgb'] = pred_lgb_dic[id]
        true_y.loc[i,'dp'] = pred_dp_dic[id]

    true_y.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/combine_all.csv')

    print('combine_all success')
    return True




def get_combine_raw(submit=False):
    df_tmp = pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp/combine_all.csv')

    df = pd.DataFrame()
    label = 'true'
    df[label] = df_tmp['true']
    df['cb_num'] = df_tmp['cb_num']
    df['cb'] = df_tmp['cb']
    df['xgb_num'] = df_tmp['xgb_num']
    df['xgb'] = df_tmp['xgb']
    df['lgb_num'] = df_tmp['lgb_num']
    df['lgb'] = df_tmp['lgb']
    return df


def add_ave(df):
    df['ave']=''
    df['ave_num']=''

    for i in range(df.shape[0]):

        ave=np.average([df.loc[i,'cb'],df.loc[i,'lgb'],df.loc[i,'xgb']])
        ave_num=np.average([df.loc[i,'cb_num'],df.loc[i,'lgb_num'],df.loc[i,'xgb_num']])
        df.loc[i,'ave']=ave
        df.loc[i,'ave_num']=ave_num
    df['diff_ave'] = df['ave_num'] - df['ave']
    return df


def get_combine_amended(submit=False):
    df =get_combine_raw(submit)
    # df= add_ave(df)

    df['diff_cb'] = df['cb_num'] - df['cb']
    df['diff_xgb'] = df['xgb_num'] - df['xgb']
    df['diff_lgb'] = df['lgb_num'] - df['lgb']

    df['cb_round'] = df['cb_num'].apply(lambda x: 1 if round(x) > x else 0)
    df['xgb_round'] = df['xgb_num'].apply(lambda x: 1 if round(x) > x else 0)
    df['lgb_round'] = df['lgb_num'].apply(lambda x: 1 if round(x) > x else 0)
    return df


