import numpy as np
import math
import pandas as pd

def mse_df(df,label):
    true=np.array(df[label])
    for column in df.columns:
        tmp = np.array(df[column])
        mse_v=mse(true,tmp)
        print("{}的mse:{:.4f}".format(column,mse_v))
    return True



def mse(arry_1,arry_2):
    if(len(arry_1)!=len(arry_2)):
        raise ValueError
    else:
        sum=0.
        for i in range(len(arry_1)):
            sum+=np.power((arry_1[i]-arry_2[i]),2)
        return sum/len(arry_1)


def round_1(value):
    return Round_Int(value,1)

def round_5(value):
    return Round_Int(value,5)

def round_10(value):
    return Round_Int(value,10)

def round_20(value):
    return Round_Int(value,20)

def round_50(value):
    return Round_Int(value,50)


def round_100(value):
    return Round_Int(value,100)

def round_1000(value):
    return Round_Int(value,1000)

def round_1w(value):
    return Round_Int(value,10000)

def round_5w(value):
    return Round_Int(value,50000)


def round_10w(value):
    return Round_Int(value,100000)

def round_15w(value):
    return Round_Int(value,150000)

def round_20w(value):
    return Round_Int(value,200000)

def round_25w(value):
    return Round_Int(value,250000)

def round_50w(value):
    return Round_Int(value,500000)






def Round_Int(value,_round):
    baseValue = 1
    baseValue *= _round
    v = Int(value)
    if (v is None):
        return None
    else:
        return round(v / baseValue) * baseValue


def Float(num):
    if (None == num):
        return None
    else:
        try:
           if(math.isnan(float(num))):
               return None
           else:
               return float(num)
        except:
            return None



def nonNegative(value):
    v=Float(value)
    if(v is None or v<0):
        return 0
    else:
        return v

def Round(num, dim=0):
    v=Float(num)
    if(None==v):
        return None
    else:
        return round(v, dim)


def Int(num):
    v=Float(num)
    if(None==v):
        return None
    else:
        return int(v)

def cal_difference_a_b(a,b):
    a=Float(a)
    b=Float(b)
    if(a is None or b is None):
        return 0
    else:
        return a-b



def standard_int_intervals(start,interval,value):
    start=int(start)
    if (value <= start):
        return start
    else:
        return round((value - start) / interval)*interval+start





if __name__ == '__main__':
    # df=pd.DataFrame([[1325,2],[3,4]],columns=['A','B'])
    # df['A']=df['A'].apply(Round_1000)
    # print(df)
    # v=Round_Int(23623,10000)
    # print(v)
    arr_1=[2,4]
    arr_2=[1,2]
    print(mse(arr_1,arr_2))


