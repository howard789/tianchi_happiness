import pandas as pd
import numpy as np
from common_kaggle import mathUtil

def generate_param_list(criteria_dic_list):
    total_list_dic={}
    for criteria_dic in criteria_dic_list:
        name=criteria_dic['name']
        start_num=criteria_dic['start_num']
        end_num=criteria_dic['end_num']
        step=criteria_dic['step']
        round_d=criteria_dic['round_d']
        if(name is None or start_num is None or end_num is None or step is None or round_d is None):
            raise ValueError
        else:
            total_list_dic[name]=generate_list(start_num,end_num,step,round_d)
    return total_list_dic

def generate_list(start_num,end_num,step,round_d=None):
    mList=[start_num]
    while(start_num+step<=end_num):
        if(round_d):
            v=round(start_num+step,round_d)
        else:
            v = start_num + step
        mList.append(v)
        start_num=v
    if(end_num not in mList):
        mList.append(end_num)
    return mList


def take_one_row_to_df(df,row_num):
    df2 = df.loc[row_num, :].to_frame()
    return transform_df(df2)


def transform_df(df):
    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    df2=df2.reset_index(drop=True)
    return df2



def check_before_train(train_df,test_df,features):
    if(contain_nan(train_df)):
        raise ValueError('train_df 包含nan')
    if (contain_nan(test_df)):
        raise ValueError('test_df 包含nan')
    for feature in features:
        try:
            v=train_df.loc[0,feature]
        except:
            print('train缺少{}'.format(feature))
        try:
            v=test_df.loc[0,feature]
        except:
            print('test缺少{}'.format(feature))

def get_all_nan_cells(df):
    mlist=[]
    for col in df.columns:
        for i in range(df.shape[0]):
            if(None==mathUtil.Float(df.loc[i,col])):
               mlist.append((col,i))
    return mlist



def get_nan_cols(df):
    mlist=[]
    for col in df.columns:
        if(None==mathUtil.Float(df.loc[0,col])):
           mlist.append(col)
    return mlist

def contain_nan(df):
    mlist=get_nan_cols(df)
    if(len(mlist)>0):
        print(mlist)
        return True
    else:
        return False

def delete_column_before_train(df,features,label):
    for col in df.columns:
        if(col not in features and col!=label):
            try:
                df=df.drop(columns=[col])
            except:
                continue

    return df



def replace_column(df,col,target_col,criteriaDic):
    for key in criteriaDic.keys():
        value=criteriaDic[key]
        for i in range(df.shape[0]):
            v=df.loc[i,col]
            if(v is not None and v==key):
                df.loc[i, target_col]=value
    return df



def get_list(start_num,end_num):
    return list(range(start_num,end_num+1))

def fillna(df,column_name,fill_value):
    for i in range(df.shape[0]):
        v=mathUtil.Float(df.loc[i,column_name])
        if(v is None):
            df.loc[i, column_name]=fill_value
    # s=df[column_name]
    # s=s.fillna(value=fill_value)
    # df[column_name]=s
    return True


def standard_int_intervals(df,origin_col_name,new_col_name,is_newCol,start,interval):
    if(is_newCol):
        df[new_col_name]=0
    for i in range(df.shape[0]):
        value = df.loc[i,origin_col_name]
        value = mathUtil.standard_int_intervals(start,interval,value)
        df.loc[i,new_col_name] = value
    return df


def combine_list(src_features,delete_features,add_features):
    # 整理 src_features
    tmp=[]
    for item in src_features:
        try:
            if(type(item)==str):
                tmp.append(item)
            else:
                len(item)
                tmp+=item
        except:
            tmp.append(item)

    # delete
    tmp2=[]
    for item in tmp:
        if(item in delete_features and item not in add_features):
            continue
        else:
            tmp2.append(item)

    # add
    for item in add_features:
        if(item not in tmp2):
            tmp2.append(item)

    # 删除重复的
    final_list=[]
    added=set()
    for i in range(len(tmp2)):
        v=tmp2[i]
        if(v not in added):
            final_list.append(v)
            added.add(v)

    return final_list



def check_all_float(df):
    list=[]
    for column in df.columns:
        for i in range(df.shape[0]):
            if(None==mathUtil.Float(df.loc[i,column])):
                list.append(column)
                break
    if(len(list)>0):
        print(list)
        return False
    else:
        return True


def nan_to_zero(df,columns):
    for column in columns:
        for i in range(df.shape[0]):
            v=df.loc[i,column]
            try:
                v2=float(v)
                if(np.isnan(v2)):
                    df.loc[i, column] = 0

            except:
                df.loc[i, column]=0

    return df


def pd_drop_col(df, column_name):
    df2 = df.drop(columns=[column_name])
    df2.reset_index(drop=True, inplace=True)
    return df2

def pd_drop_row_after(df, num):
    # 删除指定的数据 float
    rows=df.shape[0]
    labels = []
    if(rows>=num):
         for i in range(num,rows):
             labels.append(i)
    df.drop(index=labels,inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df



def pd_drop_row_val(df, column_name, column_float_list):
    # 删除指定的数据 float

    row_index = []
    for i in range(df.shape[0]):
        v = df.loc[i, column_name]
        try:
            v = float(v)
            if (v in column_float_list):
                row_index.append(i)
        except:
            row_index.append(i)
    df2 = df.drop(row_index, axis=0)
    df2.reset_index(drop=True, inplace=True)
    return df2


def pd_one_hot(df, target_column, classes,features1,features2,add_featrue1):
    if(add_featrue1):
        features1.append(target_column)
    tmpList=[]
    for class_name in classes:
        if(float(class_name)>=0):
            label_name = target_column + '_' + str(class_name)
        else:
            label_name = target_column + '_m' + str(abs(class_name))
        df[label_name] = df[target_column].apply(lambda x: 1 if x == class_name else 0)
        tmpList.append(label_name)
    features2.append(tmpList)
    return df



def scan_nan(df,non_float_list=[],column_name=None,nan_list=[]):
    #检查缺失情况,不直接使用
    rowSet=set()
    for column in df.columns:
        if(column_name!=None):
            if(column!=column_name):
                continue
        else:
            count=0
            for i in range(df.shape[0]):
                v=df.loc[i,column]
                if(None==v or v in nan_list or (column not in non_float_list and None == mathUtil.Float(v))):
                    count +=1
                    rowSet.add(i)
            if (column_name != None or count>0):
                print('{}缺少率{:.2%}'.format(column,count/df.shape[0]))
    if (column_name == None):
        print('全部row缺少率{:.2%}'.format(len(rowSet) / df.shape[0]))

def concat_dataFrame(df1,df2):
    #train和test连在一起
    data = pd.concat([df1,df2],axis=0,ignore_index=True)
    return data

def scan_data(df):

    "快速查看数据情况"
    # df=pd.DataFrame()

    print("数据量:{}".format(df.shape[0]))
    features=df.columns
    print("特征数:{},{}".format(len(features),features))
    print('----------------------------------------------------')

    # 查看数据是否缺失
    df.info(verbose=True, null_counts=True)

    for i in range(len(features)):
        print("第{}个feature:{}".format(i+1,features[i]))
        scan_nan(df, column_name=features[i])
        s=df[features[i]]
        # 查看label分布
        print(s.value_counts())

        total_arr=s.tolist()
        unique_arr=s.unique()
        print("数据类型:{}".format(type(unique_arr[0])))
        print("数据种类数:{}".format(len(unique_arr)))
        minV, maxV, mean, contain_nan=min_max_mean_nan(total_arr)
        print("极小值:{}".format(minV))
        print("极大值:{}".format(maxV))
        print("包含nan:{}".format(contain_nan))
        unique_arr=sorted(unique_arr)
        if(len(unique_arr)<20):
            print("全部类型:{}".format(unique_arr))
        else:
            print("前5个:{}".format(get_not_nan_data(unique_arr,5)))
        print('----------------------------------------------------')


def scan_one_feature(df,feature,str_num=None):
    if(feature=='na'):
        print('占位符')

    print("第{}个feature:{}".format(str_num, feature))
    scan_nan(df, column_name=feature)
    s = df[feature]
    total_arr = s.tolist()
    unique_arr = s.unique()
    num_of_kinds=len(unique_arr)
    print("数据类型:{}".format(type(unique_arr[0])))
    print("数据种类数:{}".format(num_of_kinds))
    minV, maxV, mean, contain_nan = min_max_mean_nan(total_arr)
    print("极小值:{}".format(minV))
    print("极大值:{}".format(maxV))
    print("包含nan:{}".format(contain_nan))
    unique_arr = sorted(unique_arr)
    if (len(unique_arr) < 20):
        print("全部类型:{}".format(unique_arr))
    else:
        print("前20个非nan:{}".format(get_not_nan_data(unique_arr, 20)))
    return num_of_kinds

def __scan_one_feature_process(df,i,features):

    num_of_kinds=0
    feature=features[i]
    if(type(feature)==str):
        num_of_kinds+=scan_one_feature(df, feature, i)
    else:
        for j in range(len(feature)):
            one_compoment_featrue=feature[j]
            num_of_kinds+=scan_one_feature(df, one_compoment_featrue, str(i)+'-'+str(j))
    return num_of_kinds


def scan_one_df_one_features(df,features,label,need_Label):
    features=features.copy()
    "快速查看数据情况"
    num_of_kinds_10=[]
    num_of_kinds_20=[]
    num_of_kinds_20_more=[]

    if(need_Label):
        if (label not in features):
            features.append(label)
    else:
        if(label in features):
            features.remove(label)



    print("df 数据量:{}".format(df.shape[0]))
    print("df 特征数:{}".format(df.shape[1]))
    print("features 特征数:{},{}".format(len(features),features))


    print('----------------------------------------------------')
    count = 0
    for i in range(len(features)):
        num_of_kinds=__scan_one_feature_process(df, i, features)

        if(num_of_kinds<=10):
            num_of_kinds_10.append(features[i])
        elif(num_of_kinds<=20):
            num_of_kinds_20.append(features[i])
        else:
            num_of_kinds_20_more.append(features[i])

        print('----------------------------------------------------')
    print('1-10个特征:',num_of_kinds_10)
    print('11-20个特征:',num_of_kinds_20)
    print('20个以上特征:',num_of_kinds_20_more)
    return True


def scan_three_df(df1,df2,features1,features2,unique_features):
    if(len(features1) != len(features2)):
        raise ValueError

    "快速查看数据情况"
    # df=pd.DataFrame()

    print("df1 数据量:{}".format(df1.shape[0]))
    print("df2 数据量:{}".format(df2.shape[0]))

    print("df1 特征数:{},{}".format(len(features1),features1))
    print("df2 特征数:{},{}".format(len(features2),features2))

    print('----------------------------------------------------')
    count=0
    for i in range(len(features1)):
        __scan_one_feature_process(df1, i, features1)
        print('----------------------------------------------------')
        __scan_one_feature_process(df2, i, features2)
        print('----------------------------------------------------')
        count+=1
    print('unique_features ------------------------------------------------------------------------')
    for i in range(len(unique_features)):
        __scan_one_feature_process(df2, i, unique_features)
        print('----------------------------------------------------')

def get_not_nan_data(arr,num):
    rtn=[]
    count=0
    for i in range(len(arr)):
        if(count==num):
            break
        v=mathUtil.Float(arr[i])
        if(None!=v and v not in rtn):
            rtn.append(v)
            count+=1
    if(len(rtn)==0):
        rtn=arr[0:num]
    return rtn


def min_max_mean_nan(list):
    if(len(list)==0):
        return None,None,None,True
    contain_nan=False
    minV=np.inf
    maxV=-np.inf
    total=0.
    mean=0.
    count=0
    for i in range(len(list)):
        v=mathUtil.Float(list[i])
        if(v==None):
            contain_nan=True
        else:
            if(v>maxV):
                maxV=v
            if (v < minV):
                minV = v
            total+=v
            count+=1
    if(count==0):
        mean=None
    else:
        mean=total/count
    return minV, maxV, mean, contain_nan



if __name__ == '__main__':
    # src_list=[1,'ff',[2,3,4,4]]
    r=generate_list(1,8,1,0)
    print(r)