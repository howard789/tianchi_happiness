

def cal_income_p(area_code,annual_income,average_gdp_dic):
    average_gdp=average_gdp_dic[area_code]
    return annual_income/(average_gdp*10000)






def get_average_house_price_perM(area_code,house_salary_dic,average_gdp_dic):
    # house_salary=get_house_salary_dic()
    # average_gdp=get_average_gdp()
    totalValue=average_gdp_dic[area_code] * house_salary_dic[area_code] * 2

    return totalValue/80


def getDic_2015_gdp_div_by_person():
    # 2015年中国各省人均GDP排名,年收入人民币万元
    dic = {}
    dic[11] = 10.58 *10000# '北京市（京）'
    dic[12] = 10.90 *10000# '天津市（津）'
    dic[13] = 4.15  *10000# '河北省（冀）'
    dic[14] = 3.56  *10000# '山西省（晋）'
    dic[15] = 7.30  *10000# '内蒙古（蒙）'

    dic[21] = 6.57 *10000# '辽宁省（辽）'
    dic[22] = 5.20 *10000# '吉林省（吉）'
    dic[23] = 3.94 *10000# '黑龙江省（黑）'

    dic[31]= 10.34 *10000# '上海市（沪）'
    dic[32] = 8.85 *10000# '江苏省（苏）'
    dic[33] = 7.64 *10000# '浙江省（浙）'
    dic[34] = 3.70 *10000# '安徽省（皖）'
    dic[35] = 7.04 *10000# '福建省（闽）'
    dic[36] = 3.71 *10000# '江西省（赣）'
    dic[37] = 6.58 *10000# '山东省（鲁）'

    dic[41] = 3.94 *10000# '河南省（豫）'
    dic[42] = 5.16 *10000# '湖北省（鄂）'
    dic[43] = 4.08 *10000# '湖南省（湘）'
    dic[44] = 6.98 *10000# '广东省（粤）'
    dic[45] = 3.65 *10000# '广西壮族（桂）'
    dic[46] = 4.27 *10000# '海南省（琼）'

    dic[50] = 5.21 *10000# '重庆市（渝）'
    dic[51] = 3.74 *10000# '四川省（川）'
    dic[52] = 3.02 *10000# '贵州省（贵）'
    dic[53] = 2.98 *10000# '云南省（云）'
    dic[54] = 3.65 *10000# '西藏（藏）',没有资料,取青海的值

    dic[61] = 4.87 *10000# '陕西省（陕）'
    dic[62] = 2.66 *10000# '甘肃省（甘）'
    dic[63] = 4.30 *10000# '青海省（青）'
    dic[64] = 4.62 *10000# '宁夏回族（宁）'
    dic[65] = 4.27 *10000# '新疆维吾尔（新）'
    return dic


def get_house_salary_dic():
    # 房价收入比
    "2015年全国30省商品住宅房价收入比排行榜 房价收入比=每户住房总价÷每户家庭年总收入"
    dic = {}
    dic[11] = 14.5#'北京市（京）'
    dic[12] = 10#'天津市（津）'
    dic[13] = 7.3#'河北省（冀）'
    dic[14] = 6.3#'山西省（晋）'
    dic[15] = 4.4 #'内蒙古（蒙）'

    dic[21] = 6.1#'辽宁省（辽）'
    dic[22] = 6.1#'吉林省（吉）'
    dic[23] = 6.9#'黑龙江省（黑）'

    dic[31] = 14#'上海市（沪）'
    dic[32] = 6.7#'江苏省（苏）'
    dic[33] = 8.5#'浙江省（浙）'
    dic[34] = 6.5 #'安徽省（皖）'
    dic[35] = 8.9 #'福建省（闽）'
    dic[36] = 6.6 #'江西省（赣）'
    dic[37] = 5.8#'山东省（鲁）'

    dic[41] = 5.8#'河南省（豫）'
    dic[42] = 7.2#'湖北省（鄂）'
    dic[43] = 4.8#'湖南省（湘）'
    dic[44] = 9.4#'广东省（粤）'
    dic[45] = 6.0#'广西壮族（桂）'
    dic[46] = 12.1#'海南省（琼）'

    dic[50] = 6.3 #'重庆市（渝）'
    dic[51] = 6.6 #'四川省（川）'
    dic[52] = 5.1#'贵州省（贵）'
    dic[53] = 6.3 #'云南省（云）'
    dic[54] = 6.0#'西藏（藏）',没有资料,取青海的值

    dic[61] = 6.6#'陕西省（陕）'
    dic[62] = 6.7 #'甘肃省（甘）'
    dic[63] = 6.0#'青海省（青）'
    dic[64] = 5.5 #'宁夏回族（宁）'
    dic[65] = 5.5#'新疆维吾尔（新）'
    return dic

def get_area_code_list():
    dic=get_area_code_dic()
    return list(dic.keys())


def get_area_code_dic():
    dic={}
    dic[11]='北京市（京）'
    dic[12]='天津市（津）'
    dic[13]='河北省（冀）'
    dic[14]='山西省（晋）'
    dic[15]='内蒙古（蒙）'

    dic[21]='辽宁省（辽）'
    dic[22]='吉林省（吉）'
    dic[23]='黑龙江省（黑）'

    dic[31]='上海市（沪）'
    dic[32]='江苏省（苏）'
    dic[33]='浙江省（浙）'
    dic[34]='安徽省（皖）'
    dic[35]='福建省（闽）'
    dic[36]='江西省（赣）'
    dic[37]='山东省（鲁）'

    dic[41]='河南省（豫）'
    dic[42]='湖北省（鄂）'
    dic[43]='湖南省（湘）'
    dic[44]='广东省（粤）'
    dic[45]='广西壮族（桂）'
    dic[46]='海南省（琼）'

    dic[50]='重庆市（渝）'
    dic[51]='四川省（川）'
    dic[52]='贵州省（贵）'
    dic[53]='云南省（云）'
    dic[54]='西藏（藏）'

    dic[61]='陕西省（陕）'
    dic[62]='甘肃省（甘）'
    dic[63]='青海省（青）'
    dic[64]='宁夏回族（宁）'
    dic[65]='新疆维吾尔（新）'
    return dic


def transfer_province_to_standard(v):
    # 1 = 上海市;
    # 2 = 云南省;
    # 3 = 内蒙古自治区;
    # 4 = 北京市;
    # 5 = 吉林省;
    if(v==1):
        return 31
    elif(v==2):
        return 53
    elif(v==3):
        return 15
    elif(v==4):
        return 11
    elif(v==5):
        return 22
    # 6 = 四川省;
    # 7 = 天津市;
    # 8 = 宁夏回族自治区;
    # 9 = 安徽省;
    # 10 = 山东省;
    elif(v==6):
        return 51
    elif(v==7):
        return 12
    elif(v==8):
        return 64
    elif(v==9):
        return 34
    elif(v==10):
        return 37
    # 11 = 山西省;
    # 12 = 广东省;
    # 13 = 广西壮族自治区;
    # 14 = 新疆维吾尔自治区;
    # 15 = 江苏省;
    elif(v==11):
        return 14
    elif(v==12):
        return 44
    elif(v==13):
        return 45
    elif(v==14):
        return 65
    elif(v==15):
        return 32
    # 16 = 江西省;
    # 17 = 河北省;
    # 18 = 河南省;
    # 19 = 浙江省;
    # 20 = 海南省;
    elif(v==16):
        return 36
    elif(v==17):
        return 13
    elif(v==18):
        return 41
    elif(v==19):
        return 33
    elif(v==20):
        return 46
    # 21 = 湖北省;
    # 22 = 湖南省;
    # 23 = 甘肃省;
    # 24 = 福建省;
    # 25 = 西藏自治区;
    elif(v==21):
        return 42
    elif(v==22):
        return 43
    elif(v==23):
        return 62
    elif(v==24):
        return 35
    elif(v==25):
        return 54
    # 26 = 贵州省;
    # 27 = 辽宁省;
    # 28 = 重庆市;
    # 29 = 陕西省;
    # 30 = 青海省;
    # 31 = 黑龙江省;
    elif(v==26):
        return 52
    elif(v==27):
        return 21
    elif(v==28):
        return 50
    elif(v==29):
        return 61
    elif(v==30):
        return 63
    elif(v==31):
        return 23









if __name__ == '__main__':
    v=list(range(1, 3))
    print(v)