
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.layers as layers

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
from tianchi_happiness import service
from tianchi_happiness import param
from mSklearn import m_xgboost
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb



def _deal_data(df,drop_invalid_label=True,debug=True):
    #将 label -1
    label='happiness'



    features1=[] #用来存放原始的feature,里面是list
    features2=[] #用来存放处理后的feature,里面是list
    unique_features=[] #用来存放新建立的feature,里面是str

    # 第1个feature:id
    # 第2个feature:happiness
    if(drop_invalid_label):
        df = data_util.pd_drop_row_val(df, 'happiness', [-8])


    #调整label为0-4
    # s=df[label]
    # s=s.map(lambda x:x-1)
    # df[label]=s




    # 第3个feature:survey_type
    # 1 = 城市;
    # 2 = 农村;
    data_util.pd_one_hot(df, 'survey_type',data_util.get_list(1,2),features1,features2,True)

    #第4个feature:province
    df['province']=df['province'].apply(area_util.transfer_province_to_standard)
    tmpList = area_util.get_area_code_list()
    data_util.pd_one_hot(df, 'province', tmpList,features1,features2,True)

    # 第5个feature:city 不处理
    # 第6个feature:county 不处理
    # 第7个feature:survey_time 不处理

    # 第8个feature:gender
    data_util.pd_one_hot(df, 'gender',data_util.get_list(1,2),features1,features2,True)

    # birth 转成年龄
    df['age'] = 2015 - df['birth']
    unique_features.append('age')

    # nationality
    data_util.pd_one_hot(df, 'nationality', data_util.get_list(1,8),features1,features2,True)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError

    # religion 和 religion_freq 合并
    # religion 0表示有信仰,1表示没有
    data_util.replace_column(df, 'religion', 'religion_freq', {1:1,-8:1})
        # 1 = 从来没有参加过;
        # 2 = 一年不到1次;
        # 3 = 一年大概1到2次;
        # 4 = 一年几次;
        # 5 = 大概一月1次;
        # 6 = 一月2到3次;
        # 7 = 差不多每周都有;
        # 8 = 每周都有;
        # 9 = 一周几次;
    data_util.replace_column(df, 'religion_freq', 'religion_freq', {-8:1})
    service.append_both_features(features1, features2,'religion_freq')



    # edu

    # 1 = 没有受过任何教育;
    # 2 = 私塾、扫盲班;
    # 3 = 小学;
    # 4 = 初中;
    # 5 = 职业高中;
    # 6 = 普通高中;
    # 7 = 中专;
    # 8 = 技校;
    # 9 = 大学专科（成人高等教育）;
    # 10 = 大学专科（正规高等教育）;
    # 11 = 大学本科（成人高等教育）;
    # 12 = 大学本科（正规高等教育）;
    # 13 = 研究生及以上;
    # 14 = 其他;

    # 第14个feature: edu_other 不处理(夜校)
    # 第15个feature: edu_status
    df['edu_a']=df['edu']
    data_util.fillna(df,'edu_a',7)
    data_util.replace_column(df, 'edu', 'edu_a', {-8:7})
    df['edu_a'] =(df['edu_a']+ df['edu_status'].apply(service.minus_edu_status))*10
    data_util.fillna(df, 'edu_a', 7)
    features1.append('edu')
    features2.append('edu_a')

    # 第16个feature: edu_yr 暂时不处理

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError


    # 第18个feature:political
    # 1 = 群众;
    # 2 = 共青团员;
    # 3 = 民主党派;
    # 4 = 共产党员;
    data_util.pd_one_hot(df, 'political', data_util.get_list(1,4), features1,features2,True)
    # 第19个feature: join_party 不处理,参加政党年份



    # 第17个feature:income (年收入)
    df['income']=df['income'].apply(service.amend_income)
    service.append_both_features(features1,features2,'income')

    #收入分组
    # df['income_cut'] = df['income'].apply(service.income_cut)
    # unique_features.append('income_cut')

    #收入和当地的平均相比
    service.cal_income_compare_local(df,'income_p')
    unique_features.append('income_p')

    # 将收入折成现值 1w 为单位
    service.discount_future_revenue(df, 'revenue_assets')
    unique_features.append('revenue_assets')

    # 第20个feature: floor_area
    # 第21个feature: property_0
    #计算持有的房产比例 1%为单位
    # 第30个feature:property_other
    service.cal_house_percent(df, 'house_percent')
    unique_features.append('house_percent')

    #计算房产价值,如果持有0%就不用算了
    service.cal_house_value(df, 'house_value')
    unique_features.append('house_value')

    #计算持有的房产价值 和总资产
    df['house_percent_value']=df['house_value']*df['house_percent']
    df['total_asset']=df['house_percent_value']+df['revenue_assets']

    unique_features.append('house_percent_value')
    unique_features.append('total_asset')

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError

    #年龄与财富的比值,每年可以享受的财富金额
    service.cal_total_asset_age(df, 'total_asset_age')
    unique_features.append('total_asset_age')

    #身高
    column_name='height_cm'
    new_col_name='height_cm_a'
    data_util.standard_int_intervals(df,column_name,new_col_name,True,140,5)
    features1.append(column_name)
    features2.append(new_col_name)

    # 体重
    column_name='weight_jin'
    new_col_name='weight_jin_a'
    data_util.standard_int_intervals(df,column_name,new_col_name,True,70,5)
    features1.append(column_name)
    features2.append(new_col_name)

    #bmi
    service.cal_bmi(df, 'bmi')
    unique_features.append('bmi')


    # health
    # 1 = 很不健康;
    # 2 = 比较不健康;
    # 3 = 一般;
    # 4 = 比较健康;
    # 5 = 很健康;
    col_name='health'
    data_util.replace_column(df,col_name,col_name,{-8:3})
    service.append_both_features(features1, features2, 'health')

    # health_problem
    col_name='health_problem'
    data_util.replace_column(df,col_name,col_name,{-8:3})
    service.append_both_features(features1, features2, 'health_problem')

    # depression
    col_name='depression'
    data_util.replace_column(df,col_name,col_name,{-8:3})
    service.append_both_features(features1, features2, 'depression')

    #health_score
    service.health_score(df,'health_score')
    unique_features.append('health_score')

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError
    # hukou
    # 1 = 农业户口;
    # 2 = 非农业户口;
    # 3 = 蓝印户口;
    # 4 = 居民户口（以前是农业户口）; 5 = 居民户口（以前是非农业户口）; 6 = 军籍;
    # 7 = 没有户口;
    # 8 = 其他;
    data_util.pd_one_hot(df, 'hukou', data_util.get_list(1,8),features1,features2,True)

    # hukou_loc
    # 1 = 本乡（镇、街道）; 2 = 本县（市、区）其他乡（镇、街道）; 3 = 本区 / 县 / 县级市以外;
    # 4 = 户口待定;
    df.fillna(value={'hukou_loc': 1})
    data_util.pd_one_hot(df, 'hukou_loc', data_util.get_list(1,4),features1,features2,True)

    # media
    # 1 = 从不;
    # 2 = 很少;
    # 3 = 有时;
    # 4 = 经常;
    # 5 = 非常频繁;
    mediaCols=['media_1','media_2','media_3','media_4','media_5','media_6']
    features1.append(mediaCols)
    for col in mediaCols:
        data_util.replace_column(df,col,col, {-8: 1})
    service.append_both_features(features1,features2,mediaCols)

    service.add_media_type(df,'media_type',features1,features2)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError
    # leisure
    # 1 = 每天;
    # 2 = 一周数次;
    # 3 = 一月数次;
    # 4 = 一年数次或更少;
    # 5 = 从不;
    leisureCols=['leisure_1','leisure_2','leisure_3','leisure_4','leisure_5','leisure_6','leisure_7','leisure_8','leisure_9','leisure_10','leisure_11','leisure_12']
    for col in leisureCols:
        data_util.replace_column(df,col,col, {-8: 5})
    service.append_both_features(features1,features2,leisureCols)
    service.add_out_home_leasure(df, 'out_leisure_p',features1,features2)


    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError


    # socialize  relax  learn
    # 1 = 从不;
    # 2 = 很少;
    # 3 = 有时;
    # 4 = 经常;
    # 5 = 非常频繁;
    for col in ['socialize','relax','learn']:
        data_util.replace_column(df, col, col, {-8: 3})
        service.append_both_features(features1, features2,col)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError
    # social_neighbor social_friend
    # 1 = 几乎每天;
    # 2 = 一周1到2次;
    # 3 = 一个月几次;
    # 4 = 大约一个月1次;
    # 5 = 一年几次;
    # 6 = 一年1次或更少;
    # 7 = 从来不;
    for col in ['social_neighbor','social_friend']:
        data_util.fillna(df,col,-8)
        data_util.replace_column(df, col, col, {-8: 3})
        service.append_both_features(features1, features2,col)

    # socia_outing 过去一年，您有多少个晚上是因为出去度假或者探访亲友而没有在家过夜
    # 1 = 从未;
    # 2 = 1 - 5个晚上;
    # 3 = 6 - 10个晚上;
    # 4 = 11 - 20个晚上;
    # 5 = 21 - 30个晚上;
    # 6 = 超过30个晚上;
    col='socia_outing'
    data_util.replace_column(df, col, col, {-8: 1})
    service.append_both_features(features1, features2, col)


    # equity
    # 1 = 完全不公平;
    # 2 = 比较不公平;
    # 3 = 说不上公平但也不能说不公平;
    # 4 = 比较公平;
    # 5 = 完全公平;
    col='equity'
    data_util.replace_column(df, col, col, {-8: 3})
    service.append_both_features(features1, features2, col)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError

    # class
    class_list=['class','class_10_before','class_10_after','class_14']
    class_list_a=['class_a','class_10_before_a','class_10_after_a','class_14_a']
    for i in range(len(class_list)):
        class_name=class_list[i]
        class_name_a=class_list_a[i]
        df[class_name_a]=df[class_name]
        data_util.replace_column(df, class_name, class_name_a, {-8: 5})
        features1.append(class_name)
        features2.append(class_name_a)
    service.add_class_change(df,unique_features)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError

    # work_exper
    # 1 = 目前从事非农工作;
    # 2 = 目前务农，曾经有过非农工作;
    # 3 = 目前务农，没有过非农工作;
    # 4 = 目前没有工作，而且只务过农;
    # 5 = 目前没有工作，曾经有过非农工作;
    # 6 = 从未工作过;
    data_util.pd_one_hot(df, 'work_exper', data_util.get_list(1,6),features1,features2,True)

    # work_status
    # 1 = 自己是老板（或者是合伙人）; 2 = 个体工商户;
    # 3 = 受雇于他人（有固定雇主）; 4 = 劳务工 / 劳务派遣人员;
    # 5 = 零工、散工（无固定雇主的受雇者）; 6 = 在自己家的生意 / 企业中工作 / 帮忙，领工资;
    # 7 = 在自己家的生意 / 企业中工作 / 帮忙，不领工资;
    # 8 = 自由职业者;
    # 9 = 其他;
    col='work_status'
    data_util.fillna(df,col,-8)
    data_util.pd_one_hot(df,col, data_util.get_list(1,9),features1,features2,True)


    # work_yr
    column_name='work_yr'
    data_util.fillna(df,column_name,0)
    data_util.standard_int_intervals(df,column_name,column_name,False,0,5)
    service.append_both_features(features1, features2,column_name)

    # work_type
    # 1 = 全职工作;
    # 2 = 非全职工作;
    column_name = 'work_type'
    data_util.fillna(df, column_name, -8)
    data_util.pd_one_hot(df, column_name,data_util.get_list(1,2),features1,features2,True)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError

    # work_manage
    # 1 = 只管理别人，不受别人管理;
    # 2 = 既管理别人，又受别人管理;
    # 3 = 只受别人管理，不管理别人;
    # 4 = 既不管理别人，又不受别人管理;
    column_name = 'work_manage'
    data_util.fillna(df, column_name, -8)
    data_util.pd_one_hot(df, column_name,data_util.get_list(1,4),features1,features2,True)

    # insur 社会保障
    # 1 = 参加了;
    # 2 = 没有参加; 97 不适用
    cols=['insur_1','insur_2','insur_3','insur_4']
    for col in cols:
        data_util.pd_one_hot(df, column_name, data_util.get_list(1, 2), features1, features2, True)

    # family_income
    df['family_income'] = df['family_income'].apply(mathUtil.nonNegative)
    service.append_both_features(features1,features2,'family_income')

    # family_m
    # 您家目前住在一起的通常有几人（包括您本人）
    col='family_m'
    col_a='family_m_a'
    data_util.standard_int_intervals(df,col,col_a,True,1,1)
    features1.append(col)
    features2.append(col_a)

    # 添加平均值
    col='family_income_m'
    service.add_family_income_permember(df,col)
    unique_features.append(col)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError



    # family_status
    # 1 = 远低于平均水平;
    # 2 = 低于平均水平;
    # 3 = 平均水平;
    # 4 = 高于平均水平;
    # 5 = 远高于平均水平;
    data_util.pd_one_hot(df, 'family_status', data_util.get_list(1,5),features1,features2,True)

    # house
    # 您家现拥有几处房产
    # df['house'] = df['house'].apply(lambda x: 0 if x <0 else x)
    col='house'
    col_a='house_a'
    data_util.standard_int_intervals(df,col,col_a,True,0,1)
    features1.append(col)
    features2.append(col_a)

    # car
    # 1 = 有;
    # 2 = 没有;
    data_util.pd_one_hot(df, 'car', data_util.get_list(1,2),features1,features2,True)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError
    #invest
    # 0 = 否;
    # 1 = 是;
    cols=['invest_1','invest_2','invest_3','invest_4','invest_5','invest_6','invest_7','invest_8']
    for col in cols:
        data_util.pd_one_hot(df, col, data_util.get_list(0, 1), features1, features2, True)

    # invest_other 不处理

    # son daughter minor_child
    cols=['son','daughter','minor_child']
    cols_a=['son_a','daughter_a','minor_child_a']
    for i in range(len(cols)):
        col=cols[i]
        data_util.fillna(df,col,0)
        col_a=cols_a[i]
        data_util.standard_int_intervals(df, col, col_a,True, 0, 1)

    # marriage
    service.deal_marriage(df, features1, features2,unique_features)


    # parent
    service.deal_parent(df,features1,features2,unique_features)

    # status_peer
    # 与同龄人相比，您本人的社会经济地位怎样
    # 1 = 较高;
    # 2 = 差不多;
    # 3 = 较低;
    col='status_peer'
    data_util.replace_column(df, col, col, {-8: 2})
    service.append_both_features(features1, features2, col)

    # status_3_before
    # 与三年前相比，您的社会经济地位发生了什么变化
    # 1 = 上升了;
    # 2 = 差不多;
    # 3 = 下降了;
    col='status_3_before'
    data_util.replace_column(df, col, col, {-8: 2})
    service.append_both_features(features1, features2, col)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError

    # view
    # 根据您的一般印象您对一些重要事情所持的观点和看法与社会大众一致的时候有多少
    # 1 = 一致的时候非常少;
    # 2 = 一致的时候比较少;
    # 3 = 一般;
    # 4 = 一致的时候比较多;
    # 5 = 一致的时候非常多;
    col='status_3_before'
    data_util.replace_column(df, col, col, {-8: 3})
    service.append_both_features(features1, features2, col)

    # inc_ability
    # 考虑到您的能力和工作状况，您目前的收入是否合理
    # 1 = 非常合理;
    # 2 = 合理;
    # 3 = 不合理;
    # 4 = 非常不合理;
    col='inc_ability'
    data_util.replace_column(df, col, col, {-8: 3})
    service.append_both_features(features1, features2, col)

    # inc_exp 您认为您的年收入达到多少元，您才会比较满意
    df['inc_exp']=df['inc_exp'].apply(mathUtil.nonNegative)
    df['income_diff']=df['inc_exp']-df['income']
    df['income_diff']=df['income_diff'].apply(mathUtil.nonNegative)
    unique_features.append('income_diff')

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError
    #trust
    # 1 = 绝大多数不可信;
    # 2 = 多数不可信;
    # 3 = 可信者与不可信者各半;
    # 4 = 多数可信;
    # 5 = 绝大多数可信;
    trust_list=['trust_1','trust_2','trust_3','trust_4','trust_5','trust_6','trust_7','trust_8','trust_9','trust_10','trust_11','trust_12','trust_13']
    for trust in trust_list:
        data_util.replace_column(df, trust, trust, {-8: 3})
        service.append_both_features(features1, features2, trust)
    df['trust_sum']=df['trust_1']+df['trust_2']+df['trust_3']+df['trust_4']+df['trust_5']+df['trust_6']+df['trust_7']+df['trust_8']+df['trust_9']+df['trust_10']+df['trust_11']+df['trust_12']+df['trust_13']
    unique_features.append('trust_sum')

    # neighbor_familiarity
    # 您和邻居，街坊 / 同村其他居民互相之间的熟悉程度
    col='neighbor_familiarity'
    data_util.replace_column(df, col, col, {-8: 3})
    service.append_both_features(features1, features2, col)

    if(debug):
        if(len(features1)!=len(features2)):
            raise ValueError

    pubs=['public_service_1','public_service_2','public_service_3','public_service_4','public_service_5','public_service_6','public_service_7','public_service_8','public_service_9']
    for pub in pubs:
        data_util.replace_column(df, pub, pub, {-8: 50})
        df[pub]=df[pub].apply(mathUtil.nonNegative)
        service.append_both_features(features1, features2, pub)
    df['pub_sum']=df['public_service_1']+df['public_service_2']+df['public_service_3']+df['public_service_4']+df['public_service_5']+df['public_service_6']+df['public_service_7']+df['public_service_8']+df['public_service_9']

    unique_features.append('pub_sum')

    param.scale_data_at_the_end(df)


    return df,features1,features2,unique_features


# def softmax(predict):
#     return predict
#     array=np.array(predict[0])
#     _list=[]
#     for i in range(len(array)):
#         tmp=array[i]
#         if(tmp>5):
#             tmp=5
#         elif(tmp<-5):
#             tmp = -5
#
#         _list.append(np.exp(tmp))
#
#     _sum=np.sum(_list)
#     _rtnList=[]
#     for i in range(len(array)):
#         _rtnList.append(_list[i]/_sum)
#
#     _rtnList2=np.array(_rtnList).reshape((1,5))
#     rtn=tf.convert_to_tensor(_rtnList2,dtype=tf.float32)
#     return rtn
#     return rtn


# def get_selected_train_data(train_df_with_id, label,keep_id_list):
#     selected_train_df_with_id=train_df_with_id.copy()
#     deleteList=[]
#     for i in range(selected_train_df_with_id.shape[0]):
#         id=selected_train_df_with_id.loc[i,'id']
#         if id not in keep_id_list:
#             deleteList.append(i)
#     selected_train_df_with_id.drop(index=deleteList,inplace=True)
#     selected_train_df_with_id.reset_index(drop=True,inplace=True)
#     return selected_train_df_with_id






def getData(submit=False):
    parentPath=os.path.abspath(os.path.dirname(__file__))+'/tmp_data/'


    if(submit):
        path_train=parentPath+"train_df_complete.csv"
        path_test = parentPath + "test_df_submit.csv"
        # train_df=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp_data/train_df_complete.csv')
        # test_df=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp_data/test_df_submit.csv')
    else:
        path_train=parentPath+"train_df_testing.csv"
        path_test = parentPath + "test_df_testing.csv"
        # train_df=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp_data/train_df_testing.csv')
        # test_df=pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp_data/test_df_testing.csv')
    train_df=pd.read_csv(path_train)
    test_df=pd.read_csv(path_test)
    train_df, test_df = common_util.scale_data(train_df=train_df, max_limit=1, min_limit=-1, labels=['happiness', 'id'],
                                               test_df=test_df)

    features=train_df.columns.tolist()
    features.remove('id')
    features.remove('happiness')


    test_df_with_id = test_df.copy()
    train_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)
    return train_df,test_df,test_df_with_id,features

def save_dealed_data():
    train_df_complete, test_df_submit=get_train_data_and_deal_data(submit=True, data_size=None, test_size=None)
    train_df_complete.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp_data/train_df_complete.csv',index=False)
    test_df_submit.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp_data/test_df_submit.csv',index=False)
    train_df, test_df=get_train_data_and_deal_data(submit=False, data_size=-1, test_size=0.2)
    train_df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp_data/train_df_testing.csv',index=False)
    test_df.to_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/tmp_data/test_df_testing.csv',index=False)
    return True



def get_train_data_and_deal_data(submit=False,data_size=-1,test_size=0.2):
    train_df, test_df = _get_train_data(submit, data_size, test_size)
    return _deal_data_impl(train_df, test_df)

def _get_train_data(submit=False,data_size=-1,test_size=0.2):
    if(submit):
        train_df = pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/data/happiness_train_complete.csv', na_filter=True, encoding='gbk')
        test_df = pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/data/happiness_test_complete.csv', na_filter=True, encoding='gbk')
        test_df['happiness']=-1
    else:
        train_df = pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/data/happiness_train_complete.csv', na_filter=True, encoding='gbk')
        if (data_size > 0):
            data_util.pd_drop_row_after(train_df, data_size)
        train_df, test_df = common_util.split_train_test(train_df, test_size)
    return train_df, test_df





def _deal_data_impl(train_df, test_df):
    # train_df, test_df=get_train_data(submit,data_size,test_size)
    #
    # 特征工程
    train_df      ,features1, features2, unique_features = _deal_data(train_df, drop_invalid_label=True)
    test_df,xx, xx, xx                           = _deal_data(test_df, drop_invalid_label=True)

    # test_df_bk = test_df.copy()
    # 整理最终的feature
    src_features = features2 + unique_features
    delete_features = param.get_delete_features()
    add_features = ['id', 'happiness']
    f_feature = data_util.combine_list(src_features, delete_features, add_features)
    label = 'happiness'

    # 依照最终feature 整理df
    train_df = data_util.delete_column_before_train(train_df, f_feature, label)
    test_df = data_util.delete_column_before_train(test_df, f_feature, label)

    # 依照最终feature 整理df
    data_util.scan_one_df_one_features(train_df, f_feature, label, True)
    data_util.scan_one_df_one_features(test_df, f_feature, label, True)
    data_util.check_before_train(train_df, test_df, f_feature)

    train_df[label] = train_df[label].apply(lambda x: x - 1)
    test_df[label] = test_df[label].apply(lambda x: x - 1)
    # if(keep_id==False):
    #     train_df.drop(columns=['id'], inplace=True)
    #     test_df.drop(columns=['id'], inplace=True)


    # X_train, y_train = common_util.split_df_to_array(train_df, label)
    # X_test, y_test = common_util.split_df_to_array(test_df, label)

    return train_df, test_df




# def model_myGbdt_submit():
#     train_df, test_df = service.load_train_data(index=3, size=-1, test_size=0.2)
#
#     # 特征工程
#     train_df, features1, features2, unique_features = deal_data(train_df, drop_invalid_label=True)
#     test_df, xx, xx, xx = deal_data(test_df, drop_invalid_label=True)
#     # 特征工程后才能backUp
#     test_df_bk = test_df.copy()
#     # data_util.scan_three_df(train_df_bk,train_df,features1, features2,unique_features)
#
#     # 整理最终的feature
#     src_features = features2 + unique_features
#     delete_features = param.get_delete_features()
#     add_features = ['id', 'happiness']
#     f_feature = data_util.combine_list(src_features, delete_features, add_features)
#
#     label = 'happiness'
#     # 依照最终feature 整理df
#     train_df = data_util.delete_column_before_train(train_df, f_feature, label)
#     test_df = data_util.delete_column_before_train(test_df, f_feature, label)
#
#     # 依照最终feature 整理df
#     data_util.scan_one_df_one_features(train_df, f_feature, label, True)
#     data_util.scan_one_df_one_features(test_df, f_feature, label, True)
#     data_util.check_before_train(train_df, test_df, f_feature)
#
#
#     train_df[label] = train_df[label].apply(lambda x: x-1)
#     test_df[label] = test_df[label].apply(lambda x: x-1)
#     train_df.drop(columns=['id'], inplace=True)
#     test_df.drop(columns=['id'], inplace=True)
#
#     X_train, y_train = common_util.split_df_to_array(train_df, label)
#     X_test, y_test = common_util.split_df_to_array(test_df, label)
#
#     print(len(X_train))
#     print(len(y_train))
#     print(len(X_test))
#     print(len(y_test))
#
#
#     params = {'learning_rate': 0.4,
#               'max_depth': 5,  # 构建树的深度，越大越容易过拟合
#               'num_boost_round': 1,
#               'objective': 'multi:softprob',  # 多分类的问题
#               'random_state': 7,
#               'silent': 0,
#               'num_class': 5,  # 类别数，与 multisoftmax 并用
#               'eta': 0.  # 为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
#               }
#
#
#
#     print('X_test shape', X_test.shape)
#     mode=2
#     if(mode==1):
#         m_xgboost.run(X_train, y_train, X_test, y_test, params, False)
#     elif(mode==2):
#         predictions = m_xgboost.submit(X_train, y_train, X_test, params)
#         final_df = pd.DataFrame()
#         final_df['id'] = test_df_bk['id']
#         final_df[label] = predictions
#         final_df[label] = final_df[label].apply(lambda x: x + 1)
#         final_df.to_csv('data/my_happiness_test.csv', index=None)
#         df=pd.read_csv('data/my_happiness_test.csv')
#         y_true_array=test_df_bk[label]
#         y_predict_array=df[label]
#         common_util.cal_correct_rate(y_true_array,y_predict_array)
#     elif (mode == 3):
#         predictions=m_xgboost.submit(X_train, y_train, X_test, params)
#         final_df=pd.DataFrame()
#         final_df['id']=test_df_bk['id']
#         final_df[label]=predictions
#         final_df[label] = final_df[label].apply(lambda x: x+1)
#         final_df.to_csv('data/my_happiness_0.62.csv',index=None)
#         print('success')
#     return True



# def model_myGbdt():
#     from myGBDT import gbdt_main
#
#     train_df, test_df = service.load_train_data(3,size=50,test_size=0.1)
#     train_df_bk=train_df.copy()
#
#     #特征工程
#     train_df, features1, features2,unique_features=deal_data(train_df,drop_invalid_label=True)
#     test_df, xx, xx,xx=deal_data(test_df,drop_invalid_label=True)
#
#     # data_util.scan_three_df(train_df_bk,train_df,features1, features2,unique_features)
#
#     # 整理最终的feature
#     src_features=features2+unique_features
#     delete_features=param.get_delete_features()
#     add_features=['id','happiness']
#     f_feature=data_util.combine_list(src_features, delete_features, add_features)
#
#     label='happiness'
#     # 依照最终feature 整理df
#     train_df =data_util.delete_column_before_train(train_df,f_feature,label)
#     test_df  =data_util.delete_column_before_train(test_df,f_feature,label)
#
#     # 依照最终feature 整理df
#     data_util.scan_one_df_one_features(train_df, f_feature,label,True)
#     data_util.scan_one_df_one_features(test_df, f_feature,label,True)
#
#     data_util.check_before_train(train_df, test_df, f_feature)
#     # service.scanPlot(train_df)
#
#
#
#
#     runMode=2
#     if(runMode==1):
#         #gbdt
#         gbdt_main.run_happiness(train_df, test_df, f_feature,['id'],5,2)
#     elif(runMode==2):
#         train_df[label] = train_df[label].apply(lambda x: 0 if x == 5 else x)
#         test_df[label] = test_df[label].apply(lambda x: 0 if x == 5 else x)
#         train_df.drop(columns=['id'], inplace=True)
#         test_df.drop(columns=['id'], inplace=True)
#
#         X_train, y_train = common_util.split_df_to_array(train_df, label)
#         X_test, y_test = common_util.split_df_to_array(test_df, label)
#
#         params = {'learning_rate': 0.4,
#                   'max_depth': 5,  # 构建树的深度，越大越容易过拟合
#                   'num_boost_round': 1,
#                   'objective': 'multi:softprob',  # 多分类的问题
#                   'random_state': 7,
#                   'silent': 0,
#                   'num_class': 5,  # 类别数，与 multisoftmax 并用
#                   'eta': 0.  # 为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
#                   }
#
#         m_xgboost.run(X_train, y_train, X_test, y_test,params,False)
#         m_xgboost.try_best_param(X_train, y_train, X_test, y_test)
#
#     elif(runMode==3):
#         # 失败的尝试
#         params = {'learning_rate': 0.4,
#                   'max_depth': 20,  # 构建树的深度，越大越容易过拟合
#                   'num_boost_round': 2000,
#                   'objective': 'multi:softprob',  # 多分类的问题
#                   'random_state': 7,
#                   'silent': 0,
#                   'num_class': 2,  # 类别数，与 multisoftmax 并用
#                   'eta': 0.3  # 为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
#                   }
#
#         models=[]
#         for i in range(2,6):
#             train_df_tmp=train_df.copy()
#             train_df_tmp[label] = train_df_tmp[label].apply(lambda x: 1 if x >= i else 0)
#             train_df_tmp.drop(columns=['id'], inplace=True)
#             X_train, y_train = common_util.split_df_to_array(train_df_tmp, label)
#
#             model=m_xgboost.get_trainedModel(X_train, y_train,params)
#             models.append(model)
#         test_df.drop(columns=['id'], inplace=True)
#         X_test, y_test = common_util.split_df_to_array(test_df, label)
#         scores=np.zeros(len(y_test))
#         for model_index in range(len(models)):
#             model=models[model_index]
#             y_pred = model.predict(xgb.DMatrix(X_test))
#             for i in range(len(y_pred)):
#                 sec_class_prob=y_pred[i][1]
#                 if(sec_class_prob>0.5):
#                     scores[i] += 1
#             # predictions=m_xgboost.get_predict(X_test, model)
#         print(scores)
#         common_util.cal_correct_rate(y_true_array=y_test,y_predict_array=scores)
#             # print("Accuracy: %.2f%%" % (accuracy * 100.0))
#
#     elif (runMode == 4):
#         #这个是深度学习
#         train_df.drop(columns=['id'], inplace=True)
#         test_df.drop(columns=['id'], inplace=True)
#
#         train_X, train_Y = common_util.split_df_to_array(train_df, label)
#         test_X, test_Y = common_util.split_df_to_array(test_df, label)
#         print(train_X.shape)
#         print(train_Y.shape)
#
#         train_X = tf.convert_to_tensor(train_X, dtype=tf.float32)
#         train_Y = tf.convert_to_tensor(train_Y, dtype=tf.int32)
#         train_Y = tf.one_hot(train_Y, depth=5)
#
#         # 将x,y合并,并分成batch
#         train_data = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
#         train_data = train_data.batch(1)
#
#         model = keras.Sequential()
#         # batch(100) num=200 loss=0.226464 正确率为 0.881400
#         # batch(100) num=300 loss=179257 正确率为 0.910200
#
#         units=214
#         model.add(layers.Dense(units=units*0.8, activation='relu'))
#         model.add(layers.Dense(units=units*0.6, activation='relu'))
#         model.add(layers.Dense(units=units*0.4, activation='relu'))
#         model.add(layers.Dense(units=units*0.1, activation='relu'))
#         model.add(layers.Dense(units=units*0.1, activation='relu'))
#         # model.add(layers.Dense(units=5, activation='Softplus'))
#         model.add(layers.Dense(units=5))
#
#         optimizer1 = optimizers.SGD(learning_rate=0.01)
#         optimizer2 = optimizers.SGD(learning_rate=0.001)
#
#         num = 100
#         for i in range(num):
#             if (i < num / 2):
#                 optimizer = optimizer1
#             else:
#                 optimizer = optimizer2
#
#             # 每次出来一个切片,并不确定会有多少个
#             for step, (x, y) in enumerate(train_data):
#                 with tf.GradientTape() as types:
#                     x = tf.reshape(x, (-1, 214))
#                     predict = model(x)
#                     # print(predict)
#                     # mse计算
#                     # predict =softmax(predict)
#                     # print(predict2)
#
#                     mse = tf.reduce_sum(tf.square(predict - y)) / x.shape[0]
#                 grads = types.gradient(target=mse, sources=model.trainable_variables)
#                 optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
#                 # model.compile(loss=keras.losses.mean_squared_error, optimizer='sgd')
#                 if (step % 100 == 0):
#                     print("{},{},loss={:.6f}".format(i, step, mse.numpy()))
#
#         test_X = tf.convert_to_tensor(test_X, dtype=tf.float32) / 255.
#         test_X = tf.reshape(test_X, (-1, 214))
#         predict = model.predict(test_X)
#         count = 0
#         for i in range(predict.shape[0]):
#             pred_y = int(np.argmax(predict[i]))
#             true_y = test_Y[i]
#             if (pred_y == true_y):
#                 count += 1
#         print("训练后正确率为 %f" % (count / len(test_Y)))
#     return True


if __name__ == '__main__':
    # model_myGbdt()
    # model_myGbdt_submit()
    ""
    # save_dealed_data()

    getData()
    # print('save_dealed_data success')
    # print("===获取当前文件目录===")
    # 当前脚本工作的目录路径
    # print(os.getcwd())
    # os.path.abspath()获得绝对路径
    # print(os.path.abspath(os.path.dirname(__file__)))
#
