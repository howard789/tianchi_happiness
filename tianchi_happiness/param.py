from common_kaggle import common_util
from common_kaggle import mathUtil
from common_kaggle import generate_test_data
from common_kaggle import data_util
from common_kaggle import area_util
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics




def get_delete_features():
    delet_list=[]

    # nan
    delet_list +=['survey_time', 'edu_other', 'join_party', 'property_other', 'invest_other']

    # delet_list +=provinceList()
    # delet_list +=generate_feature_list('nationality',1,8)
    #
    # delet_list +=generate_feature_list('hukou_loc', 1, 4)
    #
    # delet_list +=generate_feature_list('hukou', 1, 8)
    #
    # delet_list +=generate_feature_list('media', 1, 6)
    #
    # delet_list +=generate_feature_list('leisure', 1, 12)
    #
    # delet_list +=generate_feature_list('trust', 1, 13)
    #
    # delet_list +=generate_feature_list('public_service', 1, 9)
    #
    # delet_list +=generate_one_hot_feature_list('invest', 1, 8)

    return delet_list











def scale_data_at_the_end(df):
    df['age']=df['age'].apply(mathUtil.round_10)
    df['income_p']=df['income_p'].apply(mathUtil.round_50)
    df['edu_a'] = df['edu_a'].apply(mathUtil.round_10)
    df['income'] = df['income'].apply(mathUtil.round_10w)
    df['family_income'] = df['family_income'].apply(mathUtil.round_10w)
    df['income_diff']=df['income_diff'].apply(mathUtil.round_15w)
    df['trust_sum'] = df['trust_sum'].apply(mathUtil.round_1)
    df['pub_sum'] = df['pub_sum'].apply(mathUtil.round_1)
    df['out_leisure_p'] = df['out_leisure_p'].apply(mathUtil.round_10)
    df['house_percent_value']=df['house_percent_value'].apply(mathUtil.round_10w)
    df['total_asset'] = df['total_asset'].apply(mathUtil.round_50w)
    df['total_asset_age'] = df['total_asset_age'].apply(mathUtil.round_5w)
    df['health_score'] = df['health_score'].apply(mathUtil.round_10w)
    df['s_edu_diff'] = df['s_edu_diff'].apply(mathUtil.round_20)
    df['ma_income_m'] = df['ma_income_m'].apply(mathUtil.round_10w)
    df['ma_income_f'] = df['ma_income_f'].apply(mathUtil.round_10w)
    df['s_income_diff'] = df['s_income_diff'].apply(mathUtil.round_15w)
    df['revenue_assets'] = df['revenue_assets'].apply(mathUtil.round_50w)
    df['first_ma_years'] = df['first_ma_years'].apply(mathUtil.round_10)
    df['cur_ma_years'] = df['cur_ma_years'].apply(mathUtil.round_10)
    df['first_ma_age_m'] = df['first_ma_age_m'].apply(mathUtil.round_1)
    df['first_ma_age_f'] = df['first_ma_age_f'].apply(mathUtil.round_1)
    df['ma_age_m'] = df['ma_age_m'].apply(mathUtil.round_5)
    df['ma_age_f'] = df['ma_age_f'].apply(mathUtil.round_5)
    df['first_ma_age_diff'] = df['first_ma_age_diff'].apply(mathUtil.round_5)
    df['ma_age_diff'] = df['ma_age_diff'].apply(mathUtil.round_5)
    df['pub_sum'] = df['pub_sum'].apply(mathUtil.round_100)
    df['house_value'] = df['house_value'].apply(mathUtil.round_50w)
    df['house_percent_value'] = df['house_percent_value'].apply(mathUtil.round_50w)
    df['family_income_m'] = df['family_income_m'].apply(mathUtil.round_10w)

    pubs=['public_service_1','public_service_2','public_service_3','public_service_4','public_service_5','public_service_6','public_service_7','public_service_8','public_service_9']
    for pub in pubs:
        df[pub] = df[pub].apply(mathUtil.round_10)

    trust_list=['trust_1','trust_2','trust_3','trust_4','trust_5','trust_6','trust_7','trust_8','trust_9','trust_10','trust_11','trust_12','trust_13']
    for trust in trust_list:
        ""

def generate_feature_list(prefix,startNum,endNum):
    _list=[]
    for i in range(startNum,endNum+1):
        v=prefix+'_'+str(i)
        _list.append(v)
    return _list

def generate_one_hot_feature_list(prefix,startNum,endNum):
    _list=[]
    for i in range(startNum,endNum+1):
        v=prefix+'_'+str(i)+'_0'
        v1=prefix+'_'+str(i)+'_1'
        _list.append(v)
        _list.append(v1)
    return _list




def provinceList():
    list=['province_11', 'province_12', 'province_13', 'province_14', 'province_15', 'province_21', 'province_22', 'province_23', 'province_31', 'province_32', 'province_33', 'province_34', 'province_35', 'province_36', 'province_37', 'province_41', 'province_42', 'province_43', 'province_44', 'province_45', 'province_46', 'province_50', 'province_51', 'province_52', 'province_53', 'province_54', 'province_61', 'province_62', 'province_63', 'province_64', 'province_65']
    return list


if __name__ == '__main__':
    v=provinceList()
    print(v)