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
import seaborn as sns
import matplotlib.pyplot as plt
def scanPlot(df):
    df.info()
    for featrue in df.columns:
        sns.countplot(x=featrue, hue="happiness", data=df)
        plt.show()
        print()

def append_both_features(features1, features2, col_name):
    features1.append(col_name)
    features2.append(col_name)


def cal_bmi(df, col_name):
    # 體重(公斤) / 身高2(公尺2)
    df[col_name] = 0
    for i in range(df.shape[0]):
        weight_kg = df.loc[i, 'weight_jin'] / 2
        height_m = df.loc[i, 'height_cm'] / 100
        bmi = weight_kg / (height_m ** 2)
        v = 0
        if (bmi < 18.5):
            v = 1
        elif (bmi < 24):
            v = 2
        elif (bmi < 28):
            v = 3
        elif (bmi < 30):
            v = 4
        elif (bmi < 40):
            v = 5
        else:
            v = 6
        df.loc[i, col_name] = v
    return df


def cal_total_asset_age(df, col_name):
    df[col_name] = 0
    for i in range(df.shape[0]):
        total_asset = df.loc[i, 'total_asset']
        age = df.loc[i, 'age']

        if (age < 85):
            rest = 85 - age
        else:
            rest = 1
        ave_asset = total_asset / rest
        if (ave_asset > 10000000):
            ave_asset = 10000000
        df.loc[i, col_name] = ave_asset


def add_class_change(df, unique_features):
    df['class_forcast'] = 0
    df['class_review'] = 0
    df['class_change_total'] = 0
    # 1 = 1(最底层);
    # 10 = 10(最顶层);
    for i in range(df.shape[0]):
        class_now = df.loc[i, 'class']
        class_10_before = df.loc[i, 'class_10_before']
        class_10_after = df.loc[i, 'class_10_after']
        if (class_now > 0 and class_10_before > 0 and class_10_after > 0):
            class_forcast = class_10_after - class_now
            class_review = class_now - class_10_before
            class_change_total = class_10_after - class_10_before
            df.loc[i, 'class_forcast'] = class_forcast
            df.loc[i, 'class_review'] = class_review
            df.loc[i, 'class_change_total'] = class_change_total
        else:
            df.loc[i, 'class_forcast'] = 0
            df.loc[i, 'class_review'] = 0
            df.loc[i, 'class_change_total'] = 0
    unique_features.append('class_forcast')
    unique_features.append('class_review')
    unique_features.append('class_change_total')
    return True


def add_out_home_leasure(df, col_name, features1, features2):
    tmp = []
    tmp.append('leisure_1')
    tmp.append('leisure_2')
    tmp.append('leisure_3')
    tmp.append('leisure_4')
    tmp.append('leisure_5')
    tmp.append('leisure_6')
    tmp.append('leisure_7')
    tmp.append('leisure_8')
    tmp.append('leisure_9')
    tmp.append('leisure_10')
    tmp.append('leisure_11')
    tmp.append('leisure_12')
    features1.append(tmp)

    df[col_name] = 0
    features2.append(col_name)
    for i in range(df.shape[0]):
        leisure_1 = frequency_to_days(df.loc[i, 'leisure_1'])  # 看电视或看碟
        leisure_2 = frequency_to_days(df.loc[i, 'leisure_2'])  # 出去看电影
        leisure_3 = frequency_to_days(df.loc[i, 'leisure_3'])  # 逛街购物
        leisure_4 = frequency_to_days(df.loc[i, 'leisure_4'])  # 读书/报纸/杂志
        leisure_5 = frequency_to_days(df.loc[i, 'leisure_5'])  # 参加文化活动，比如听音乐会看演出或展览
        leisure_6 = frequency_to_days(df.loc[i, 'leisure_6'])  # 与不住在一起的亲戚聚会
        leisure_7 = frequency_to_days(df.loc[i, 'leisure_7'])  # 与朋友聚会
        leisure_8 = frequency_to_days(df.loc[i, 'leisure_8'])  # 在家听音乐
        leisure_9 = frequency_to_days(df.loc[i, 'leisure_9'])  # 参加体育锻炼
        leisure_10 = frequency_to_days(df.loc[i, 'leisure_10'])  # 观看体育比赛
        leisure_11 = frequency_to_days(df.loc[i, 'leisure_11'])  # 做手工
        leisure_12 = frequency_to_days(df.loc[i, 'leisure_12'])  # 上网
        sum = leisure_1 + leisure_2 + leisure_3 + leisure_4 + leisure_5 + leisure_6 + leisure_7 + leisure_8 + leisure_9 + leisure_10 + leisure_11 + leisure_12
        home = leisure_1 + leisure_4 + leisure_8 + leisure_10 + leisure_11 + leisure_12
        out = leisure_2 + leisure_3 + leisure_5 + leisure_6 + leisure_7 + leisure_9
        df.loc[i, col_name] = mathUtil.Round(out / sum * 100)


def add_media_type(df, col_name, features1, features2):
    df[col_name] = 0
    for i in range(df.shape[0]):
        media_1 = frequency_to_days(df.loc[i, 'media_1'])
        media_2 = frequency_to_days(df.loc[i, 'media_2'])
        media_3 = frequency_to_days(df.loc[i, 'media_3'])
        media_4 = frequency_to_days(df.loc[i, 'media_4'])
        media_5 = frequency_to_days(df.loc[i, 'media_5'])
        media_6 = frequency_to_days(df.loc[i, 'media_6'])
        sum = media_1 + media_2 + media_3 + media_4 + media_5
        media_paper = media_1 + media_2
        media_tv = media_3 + media_4
        media_net = media_5 + media_6
        if (media_net > media_tv and media_net > media_paper):
            df.loc[i, col_name] = 1
        elif (media_tv > media_net and media_tv > media_paper):
            df.loc[i, col_name] = 2
        else:
            df.loc[i, col_name] = 3

    tmpList = []
    data_util.pd_one_hot(df, col_name, data_util.get_list(1, 3), features1, features2, False)
    return df


def health_score(df, col_name):
    df[col_name] = 0.
    for i in range(df.shape[0]):
        health = df.loc[i, 'health']
        health_problem = df.loc[i, 'health_problem']
        depression = df.loc[i, 'depression']
        health_problem *= 0.2
        depression *= 0.2
        score = health * health_problem * depression * 25
        score = mathUtil.Round(score)
        df.loc[i, col_name] = score
    return True



def isMale(df,i):
    gender = df.loc[i, 'gender']
    if (gender == 1):
        return True
    else:
        return False
    return None

def deal_marriage_edu(df, features1, features2,unique_features):

    df['s_edu_diff'] = 0.
    unique_features.append('s_edu_diff')
    male_edu_score = 0
    female_edu_score = 0
    for i in range(df.shape[0]):
        if (isMale(df, i)):
            male_edu_score = df.loc[i, 'edu_a']
            female_edu_score = df.loc[i, 's_edu']
        else:
            female_edu_score = df.loc[i, 'edu_a']
            male_edu_score = df.loc[i, 's_edu']

        female_edu_score = mathUtil.Float(female_edu_score)
        male_edu_score = mathUtil.Float(male_edu_score)
        if (female_edu_score is not None and male_edu_score is not None):
            df.loc[i, 's_edu_diff'] = male_edu_score - female_edu_score


def deal_marriage_income(df, features1, features2,unique_features):
    df['s_income'] = df['s_income'].apply(amend_income)
    df['s_income_diff'] = 0.
    df['ma_income_m'] = 0.
    df['ma_income_f'] = 0.

    unique_features.append('s_income_diff')
    unique_features.append('ma_income_m')
    unique_features.append('ma_income_f')

    male_income = 0
    female_income = 0
    for i in range(df.shape[0]):
        if (isMale(df, i)):
            male_income = df.loc[i, 'income']
            female_income = df.loc[i, 's_income']
        else:
            female_income = df.loc[i, 'income']
            male_income = df.loc[i, 's_income']

        female_income = mathUtil.Float(female_income)
        male_income = mathUtil.Float(male_income)
        if (female_income is not None and male_income is not None):
            df.loc[i, 's_income_diff'] = male_income - female_income
            df.loc[i, 'ma_income_m'] = male_income
            df.loc[i, 'ma_income_f'] = female_income

    return df

def deal_marriage_years(df, features1, features2,unique_features):
    # 1 = 未婚;
    # 2 = 同居;
    # 3 = 初婚有配偶;
    # 4 = 再婚有配偶;
    # 5 = 分居未离婚;
    # 6 = 离婚;
    # 7 = 丧偶;

    df['first_ma_years'] = 0  # 第一次结婚至今
    df['cur_ma_years'] = 0  # 结婚至今
    unique_features.append('first_ma_years')
    unique_features.append('cur_ma_years')


    for i in range(df.shape[0]):
        status = df.loc[i, 'marital']
        if (status<3 or status>4 ):
            continue

        first_m_year = df.loc[i, 'marital_1st']  # marital_1st 您第一次结婚的时间

        # marital_now 您与目前的配偶是哪一年结婚的
        now_m_year = df.loc[i, 'marital_now']  # 本次结婚的时间
        first_m_year = mathUtil.Int(first_m_year)
        now_m_year = mathUtil.Int(now_m_year)
        if(first_m_year is not None):
            # 第一次结婚年龄 和经过了几年
            if (1900 < first_m_year < 2016):
                if (status == 3):
                    df.loc[i, 'first_ma_years'] = 2015 - first_m_year
        if (now_m_year is not None):
            # 目前结婚经过了几年
            if (1900 < now_m_year < 2016):
                df.loc[i, 'cur_ma_years'] = 2015 - now_m_year
    return df


def deal_marriage_age(df, features1, features2,unique_features):

    # 1 = 未婚;
    # 2 = 同居;
    # 3 = 初婚有配偶;
    # 4 = 再婚有配偶;
    # 5 = 分居未离婚;
    # 6 = 离婚;
    # 7 = 丧偶;
    df['first_ma_age_m'] = 0  # 第一次结婚年龄
    df['first_ma_age_f'] = 0  # 第一次结婚年龄
    df['ma_age_m'] = 0
    df['ma_age_f'] = 0
    df['first_ma_age_diff'] = 0
    df['ma_age_diff'] = 0

    unique_features.append('first_ma_age_m')
    unique_features.append('first_ma_age_f')
    unique_features.append('ma_age_m')
    unique_features.append('ma_age_f')
    unique_features.append('first_ma_age_diff')
    unique_features.append('ma_age_diff')

    for i in range(df.shape[0]):
        status = df.loc[i, 'marital']
        if (status < 3 or status > 4):
            continue

        birth = df.loc[i, 'birth']  # 生年
        # s_birth 您目前的配偶或同居伴侣是哪一年出生的
        s_birth = mathUtil.Int(df.loc[i, 's_birth'])  # 配偶生年
        first_m_year = df.loc[i, 'marital_1st']  # marital_1st 您第一次结婚的时间

        # marital_now 您与目前的配偶是哪一年结婚的
        now_m_year = df.loc[i, 'marital_now']  # 本次结婚的时间

        first_m_year = mathUtil.Int(first_m_year)
        now_m_year = mathUtil.Int(now_m_year)




        # 相差几岁
        # 男方结婚年龄
        # 女方结婚年龄
        theGuy=mathUtil.cal_difference_a_b(now_m_year,birth)
        spouse=mathUtil.cal_difference_a_b(now_m_year,s_birth)

        if(theGuy<18 or spouse<18):
            # first_m_year 或  now_m_year为负值所产生
            continue

        if(isMale(df,i)):
            df.loc[i, 'ma_age_m']=theGuy
            df.loc[i, 'ma_age_f']=spouse
            if(status==3):
                df.loc[i, 'first_ma_age_m']=theGuy
                df.loc[i, 'first_ma_age_f']=spouse
        else:
            df.loc[i, 'first_ma_age_f'] = theGuy
            df.loc[i, 'first_ma_age_m'] = spouse
            if (status == 3):
                df.loc[i, 'first_ma_age_f'] = theGuy
                df.loc[i, 'first_ma_age_m'] = spouse
        df.loc[i, 'first_ma_age_diff']=mathUtil.cal_difference_a_b(df.loc[i, 'first_ma_age_m'],df.loc[i, 'first_ma_age_f'])
        df.loc[i, 'ma_age_diff']=mathUtil.cal_difference_a_b(df.loc[i,'ma_age_m'],df.loc[i,'ma_age_f'])
    return df



def deal_marriage(df, features1, features2,unique_features):


    # marital
    # 1 = 未婚;
    # 2 = 同居;
    # 3 = 初婚有配偶;
    # 4 = 再婚有配偶;
    # 5 = 分居未离婚;
    # 6 = 离婚;
    # 7 = 丧偶;
    data_util.pd_one_hot(df, 'marital', data_util.get_list(1, 7), features1, features2, True)

    # 是否离过婚
    df['ever_divorce'] = df['marital'].apply(lambda x: 1 if 3 < x < 7 else 0)
    features1.append('na')
    data_util.pd_one_hot(df, 'ever_divorce', data_util.get_list(0,1),features1,features2,False)

    # 是否有伴侣
    df['stay_with_someone'] = df['marital'].apply(lambda x: 1 if 1 < x < 5 else 0)
    features1.append('na')
    data_util.pd_one_hot(df, 'stay_with_someone', data_util.get_list(0,1),features1,features2,False)



    # 学历差距
    # s_edu 您配偶或同居伴侣目前的最高教育程度
    deal_marriage_edu(df, features1, features2,unique_features)

    # 收入差距
    # s_edu 您配偶或同居伴侣目前的最高教育程度
    # s_income 您配偶或同居伴侣去年全年的总收入
    deal_marriage_income(df, features1, features2,unique_features)

    # 结婚年数
    deal_marriage_years(df, features1, features2,unique_features)

    # 结婚age
    deal_marriage_age(df, features1, features2,unique_features)

    # s_political 您配偶或同居伴侣的政治面貌 暂不处理
    # s_hukou 您配偶或同居伴侣目前的户口登记状况 暂不处理
    # s_political 您配偶或同居伴侣的政治面貌 暂不处理
    # s_work_exper 您配偶或同居伴侣的工作经历及状况 暂不处理
    # s_work_status 下列各种情形，哪一种更符合您配偶或同居伴侣目前的工作状况 暂不处理
    # s_work_type 您配偶或同居伴侣目前工作的性质 暂不处理


def add_family_income_permember(df, col_name):
    df[col_name] = 0.
    for i in range(df.shape[0]):
        income = df.loc[i, 'family_income']
        m = df.loc[i, 'family_m_a']
        income / m
        df.loc[i, col_name] = mathUtil.round_1w(income / m)


def cal_house_percent(df, col_name):
    df[col_name] = 0.
    for i in range(df.shape[0]):
        percent = cal_house_holding_per(df, i)
        percent = mathUtil.Round(percent, 2)
        df.loc[i, col_name] = percent


def cal_house_value(df, col_name):
    # 估计房产价值
    df[col_name] = 0
    average_gdp_dic = area_util.getDic_2015_gdp_div_by_person()
    house_salary_dic = area_util.get_house_salary_dic()

    for i in range(df.shape[0]):
        holding_percent = df.loc[i, 'house_percent']
        if (holding_percent < 1):
            df.loc[i, col_name] = 0
        else:
            area_code = df.loc[i, 'province']
            house_value_perM = v = area_util.get_average_house_price_perM(area_code, house_salary_dic, average_gdp_dic)
            floor_area = df.loc[i, 'floor_area']
            final_house_value = house_value_perM * floor_area * holding_percent

            # 农村城市
            if (df.loc[i, 'survey_type'] == 2):
                final_house_value *= 0.2

            final_house_value = mathUtil.round_10w(final_house_value)
            df.loc[i, col_name] = final_house_value


def discount_future_revenue(df, col_name):
    df[col_name] = 0
    for i in range(df.shape[0]):
        years = 55 - df.loc[i, 'age']
        income = df.loc[i, 'income']
        value = common_util.cal_periodic_value(income, years)
        df.loc[i, col_name] = value
    return df


def cal_income_compare_local(df, column_name):
    df[column_name] = 0
    IncomeDic = area_util.getDic_2015_gdp_div_by_person()
    for i in range(df.shape[0]):
        area_code = df.loc[i, 'province']
        localIncome = IncomeDic[area_code]
        personal_income = df.loc[i, 'income']
        p = mathUtil.Round(personal_income / localIncome * 100, 0)
        df.loc[i, column_name] = p

    return df


def frequency_to_days(value):
    # 1 = 每天;
    # 2 = 一周数次;
    # 3 = 一月数次;
    # 4 = 一年数次或更少;
    # 5 = 从不;
    if (value == 5):
        return 1
    elif (value == 4):
        return 5
    elif (value == 3):
        return 12 * 2
    elif (value == 2):
        return 52 * 2
    elif (value == 1):
        return 365
    elif (value == -8):
        return 1
    else:
        print(value)
        raise ValueError


def cal_house_holding_per(df, i):
    value = 0.
    if (df.loc[i, 'property_0'] == 1):  # 无法回答
        value = 0.
    elif (df.loc[i, 'property_1'] == 1):  # 自己所有
        value = 1
    elif (df.loc[i, 'property_2'] == 1):  # -配偶所有
        value = 0.5
    elif (df.loc[i, 'property_3'] == 1):  # 子女所有
        value = 0.25
    elif (df.loc[i, 'property_4'] == 1):  # 父母所有
        value = 0.75
    elif (df.loc[i, 'property_5'] == 1):  # 配偶父母所有
        value = 0.5 * 0.75
    elif (df.loc[i, 'property_6'] == 1):  # 子女配偶所有
        value = 0.25 * 0.5
    elif (df.loc[i, 'property_7'] == 1):  # 其他家人/亲戚所有
        value = 0.
    elif (df.loc[i, 'property_8'] == 1):  # 家人/亲戚以外的个人或单位所有，这所房子是租来的
        value = 0.
    else:
        value = 0.

    property_other = str(df.loc[i, 'property_other']).strip()
    if (value == 0 and '房产证' in property_other):
        value = 1
    elif (value > 0 and '小产权' in property_other):
        value *= 0.5

    return value

def income_cut(x):
    # 全部类型: [0   100000   200000   800000   600000   300000  1000000  5000000
    #        900000  2000000 10000000   500000   400000   700000  9000000]
    if x==0:
        return 0
    elif  0<x<=100000:
        return 1
    elif  100000<x<=200000:
        return 2
    elif  200000<x<=300000:
        return 3
    elif  300000<x<=500000:
        return 4
    elif  500000<x<=1000000:
        return 5
    else:
        return 6

def amend_income(row):
    # 9999996 = 个人全年总收入高于百万位数;
    # 9999997.不适用 9999998.不知道9999999.拒绝回答
    if (row == 9999996):
        return 10000000
    elif (row == 9999997 or row == 9999998 or row == 9999999 or row<0):
        return 0
    else:
        return row


def load_train_data(index,size,test_size):
    if (index == 1):
        ""
        # train_df=pd.read_csv('./data/happiness_train_short.csv',na_filter=True)
        train_df = pd.read_csv('./data/happiness_train_abbr.csv', na_filter=True)
    elif (index == 2):
        ""
    elif (index == 3):
        train_df = pd.read_csv(r'D:/wks/wks_ml_kaggle/tianchi_happiness/data/happiness_train_complete.csv', na_filter=True, encoding='gbk')
    elif (index == 4):
        train_df = pd.read_csv('data/happiness_test_complete.csv', na_filter=True, encoding='gbk')
    if(size>0):
        data_util.pd_drop_row_after(train_df, size)

    train_df, test_df = common_util.split_train_test(train_df, test_size)

    return train_df, test_df


# def predelete(df):
#     # df=pd.DataFrame()
#     row_index=[]
#     for i in range(df.shape[0]):
#         v=df.loc[i,'happiness']
#         try:
#             v=float(v)
#             if(v==-8):
#                 row_index.append(i)
#         except:
#             row_index.append(i)
#     df.drop(row_index,axis=0,inplace=True)
#     df.reset_index(drop=True,inplace=True)


def deal_survay_time():
    "survey_time"
    # birth转成age


def minus_edu_status(row):
    # 1 = 正在读;
    # 2 = 辍学和中途退学;
    # 3 = 肄业;
    # 4 = 毕业;
    if (row == 1):
        return -0.5
    elif (row == 2):
        return -0.8
    elif (row == 3):
        return -0.3
    elif (row == 4):
        return 0


def deal_parent(df,features1,features2,unique_features):
    # f_birth 暂不处理
    # f_edu 暂不处理
    # f_political 暂不处理
    # f_work_14 暂不处理
    # m_birth 暂不处理
    # m_edu 暂不处理
    # m_political 暂不处理
    # m_work_14 暂不处理
    pass


if __name__ == '__main__':
    df=pd.DataFrame
    # isMale(df,0)
