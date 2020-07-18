
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import tzinfo
import time as time_c

def Sleep(sec):
    time_c.sleep(sec)

def getToday_iso():
    return date.today().isoformat()

def Now():
    return datetime.now()

def compareTwoDateStr(dateStr_ToBeCompared,DateStr_critera):
    date1=strToDateObj(DateStr_critera)
    date2=strToDateObj(dateStr_ToBeCompared)
    return (date2-date1).days


def getNextNDays(dateStr_iso,n,holidays):
    return _internalUse_getNextNDays(dateStr_iso, n, True, holidays)


def _internalUse_getNextNDays(dateStr_iso,n,tradeDayOnly=True,holidays=None):

    if(tradeDayOnly):
        if(holidays==None):
            raise ValueError
        else:
            count=0
            target=dateStr_iso
            while(count!=n):
                if(n>0):
                    target = strToDateObj(target) + timedelta(days=1)
                    if (isTradeDate(target.isoformat(), holidays)):
                        count+=1
                else:
                    target = strToDateObj(target) + timedelta(days=-1)
                    if (isTradeDate(target.isoformat(), holidays)):
                        count -= 1

    else:
        target = strToDateObj(dateStr_iso) + timedelta(days=n)
    return target.isoformat()

def getWeekDay(dateStr):
    return strToDateObj(dateStr).isoweekday()

def strToDateObj(dateStr):
    _date=str(dateStr).replace('\\',"").strip()
    _date=_date.split(' ')[0]
    _date=_date.split('-')

    if(len(_date)==3):
        _year=_date[0]
        _month=_date[1]
        _day=_date[2]
    elif(len(_date)==1):
        _date=str(_date).split("'")[1]
        _year = _date[0:4]
        _month = _date[4:6]
        _day = _date[6:8]
    else:
        raise ValueError

    return date(int(_year),int(_month),int(_day))

def datetimeObj_to_IsoDateStr(datetimeObj):
    return datetimeObj.isoformat()

def string_to_IsoDateStr(dateStr):
    return strToDateObj(dateStr).isoformat()

def isFutureDay(dateStr_iso):
    today=datetime.today()
    baseDay=strToDateObj(dateStr_iso)
    if(calDaysInterval(baseDay,today)>0):
        return True
    else:
        return False


def calDaysInterval(day1Str, day2Str):
    day1 = strToDateObj(day1Str)
    day2 = strToDateObj(day2Str)
    delta = day2 - day1
    return delta.days

def getCalendarTuple(dateStr):
    return strToDateObj.isocalendar()


def isWeekend(dateStr):
    if(getWeekDay(dateStr)>5):
        return True
    else:
        return False


def getHolidaySet():
    holiday=set()
    holiday.add("")
    holiday.add("2016-01-01")#, "元旦");
    holiday.add("2016-02-08")#, "春节");
    holiday.add("2016-02-09")#, "春节");
    holiday.add("2016-02-10")#, "春节");
    holiday.add("2016-02-11")#, "春节");
    holiday.add("2016-02-12")#, "春节");
    holiday.add("2016-04-04")#, "清明");
    holiday.add("2016-05-02")#, "劳动");
    holiday.add("2016-06-09")#, "端午");
    holiday.add("2016-06-10")#, "端午");
    holiday.add("2016-09-15")#, "中秋");
    holiday.add("2016-09-16")#, "中秋");
    holiday.add("2016-10-03")#, "国庆");
    holiday.add("2016-10-04")#, "国庆");
    holiday.add("2016-10-05")#, "国庆");
    holiday.add("2016-10-06")#, "国庆");
    holiday.add("2016-10-07")#, "国庆");

    holiday.add("2017-01-02")#, "元旦");
    holiday.add("2017-01-27")#, "春节");
    holiday.add("2017-01-30")#, "春节");
    holiday.add("2017-01-31")#, "春节");
    holiday.add("2017-02-01")#, "春节");
    holiday.add("2017-02-02")#, "春节");
    holiday.add("2017-04-03")#, "清明");
    holiday.add("2017-04-04")#, "清明");
    holiday.add("2017-05-01")#, "劳动");
    holiday.add("2017-05-29")#, "端午");
    holiday.add("2017-05-30")#, "端午");
    holiday.add("2017-10-02")#, "国庆");
    holiday.add("2017-10-03")#, "国庆");
    holiday.add("2017-10-04")#, "国庆");
    holiday.add("2017-10-05")#, "国庆");
    holiday.add("2017-10-06")#, "国庆");

    holiday.add("2018-01-01")#, "元旦");
    holiday.add("2018-02-15")#, "春节");
    holiday.add("2018-02-16")#, "春节");
    holiday.add("2018-02-19")#, "春节");
    holiday.add("2018-02-20")#, "春节");
    holiday.add("2018-02-21")#, "春节");
    holiday.add("2018-04-05")#, "清明");
    holiday.add("2018-04-06")#, "清明");
    holiday.add("2018-04-30")#, "劳动");
    holiday.add("2018-05-01")#, "劳动");
    holiday.add("2018-06-18")#, "端午");
    holiday.add("2018-09-24")#, "中秋");
    holiday.add("2018-10-01")#, "国庆");
    holiday.add("2018-10-02")#, "国庆");
    holiday.add("2018-10-03")#, "国庆");
    holiday.add("2018-10-04")#, "国庆");
    holiday.add("2018-10-05")#, "国庆");
    holiday.add("2018-12-31")#, "元旦");

    holiday.add("2019-01-01")#, "元旦");
    holiday.add("2019-02-04")#, "春节");
    holiday.add("2019-02-05")#, "春节");
    holiday.add("2019-02-06")#, "春节");
    holiday.add("2019-02-07")#, "春节");
    holiday.add("2019-02-08")#, "春节");
    holiday.add("2019-04-05")#, "清明");
    holiday.add("2019-05-01")#, "劳动");
    holiday.add("2019-05-02")#, "劳动");
    holiday.add("2019-05-03")#, "劳动");
    holiday.add("2019-06-07")#, "端午");
    holiday.add("2019-09-13")#, "中秋");
    holiday.add("2019-09-14")#, "中秋");
    holiday.add("2019-10-01")#, "国庆");
    holiday.add("2019-10-02")#, "国庆");
    holiday.add("2019-10-03")#, "国庆");
    holiday.add("2019-10-04")#, "国庆");
    holiday.add("2019-10-07")#, "重阳");


    holiday.add("2020-01-01")#, "元旦");
    holiday.add("2020-01-24")#, "春节");
    holiday.add("2020-01-27")#, "春节");
    holiday.add("2020-01-28")#, "春节");
    holiday.add("2020-01-29")#, "春节");
    holiday.add("2020-01-30")#, "春节");
    holiday.add("2020-01-31")#, "春节");
    holiday.add("2020-04-06")#, "清明");
    holiday.add("2020-05-01")#, "劳动");
    holiday.add("2020-05-04")#, "劳动");
    holiday.add("2020-05-05")#, "劳动");
    holiday.add("2020-06-25")#, "端午");
    holiday.add("2020-06-26")#, "端午");
    holiday.add("2020-09-13")#, "中秋");
    holiday.add("2020-10-01")#, "国庆");
    holiday.add("2020-10-02")#, "国庆");
    holiday.add("2020-10-05")#, "国庆");
    holiday.add("2020-10-06")#, "国庆");
    holiday.add("2020-10-07")#, "国庆");
    holiday.add("2020-10-08")#, "国庆");

    holiday.add("2020-05-06")  # , "忘记更新数据");
    return holiday


def isTradeDate(dateStr_iso,holiday):

    if(isWeekend(dateStr_iso) or (dateStr_iso in holiday)):
        return False
    else:
        return True

def getSeqTradeDateDic():
    holidays=getHolidaySet()
    _date=strToDateObj('2016-01-01')
    dicSeq_TradeDay={}
    dicTradeDay_Seq={}
    i = 1;
    while(_date<=date.today()):
        dateStr_iso=_date.isoformat()
        if(isTradeDate(dateStr_iso,holidays)):
            dicSeq_TradeDay[i]=dateStr_iso
            dicTradeDay_Seq[dateStr_iso]=i
            i+=1
        _date=_date+timedelta(days=1)
    return  dicSeq_TradeDay,dicTradeDay_Seq


def getCurrentDatetime_str():
    return time_c.strftime("%Y-%m-%d %H:%M:%S", time_c.localtime())



def calTimeLeft(currentNum,totalNum,startTime,currentTime,preText=None):
    currentNum+=1
    timeSpend=currentTime-startTime
    timeLeft=(timeSpend/currentNum)*(totalNum-currentNum)
    # print("完成了:",currentNum," 剩余:",totalNum-currentNum)
    print(preText,"花费时间:",timeSpend," 剩余时间:",timeLeft)
    return timeSpend,timeLeft


def getSeqTradeDays(startDay,endDay,holidays):
    if(holidays==None):
        holidays=getHolidaySet()
    startDay=strToDateObj(startDay)
    endDay=strToDateObj(endDay)
    list=[]
    targetDay=startDay
    while(targetDay<=endDay):
        if(isTradeDate(targetDay.isoformat(),holidays)):
            list.append(targetDay.isoformat())
        targetDay=getNextNDays(targetDay.isoformat(),1,holidays)
        targetDay=strToDateObj(targetDay)
    return list

def afterTraingTime():
    currentHr=datetime.now().hour
    currentMin=datetime.now().minute
    if 9<currentHr<15:
        return False
    elif(currentHr>16):
        return True
    elif(currentMin>30):
        return True



if __name__ == '__main__':
    afterTraingTime()
    # r=compareTwoDateStr('2020-04-21', '2020-04-22')
    # r=getCurrentDatetime_str()
    # print(r)
    # print(type(r))
    # print(len(r))

    # pass
    # t1=strToDateObj('20200411')
    # t2=strToDateObj('20200411')
    # print(t1<t2)

    # timedelta=t1+1
    # t1=datetime(t1)
    # print(t1+datetime.timedelta(days=1))
    # d=date(2010,1,1)
    # d.time
    # print(t.isocalendar())
    # print(type(t1.isoweekday()))

    # now = datetime.now()
    # date = now + timedelta(days = 1)

    # today = date.today()
    # date = today + timedelta(days = 1)

    # print(_isWeekend('20200410'))

    # dicSeq_TradeDay, dicTradeDay_Seq=getSeqTradeDateDic()
    # print(len(dicSeq_TradeDay))
    # print(len(dicTradeDay_Seq))
    # print(dicSeq_TradeDay)
    # _currentNum=1
    # _totalNum=15
    # _startTime=datetime.now()
    # time_c.sleep(1)
    # _currentTime=datetime.now()
    # timeSpend, timeLeft=calTimeLeft(_currentNum, _totalNum, _startTime, _currentTime)
    # print(timeSpend)
    # print(timeLeft)
    # holidays=getHolidaySet()
    # r=isTradeDate('2020-01-31',holidays)
    # r=getNextNDays("2020-02-03",-1,True,holidays)
    # print(r)
    pass
