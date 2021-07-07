import pandas as pd
import numpy as np
import time
from math import exp, sqrt


# 基于隐语义模型的推荐
class LFM:
    def __init__(self):
        self.rating = pd.read_csv('./recommend/data/ratings.csv')

    @classmethod
    def __getUserNegativeItem__(clf, frame, userId, ratio=1):
        '''
        获取用户负反馈物品：热门但是用户没有进行过评分 与正反馈数量相等
        :param frame: ratings数据
        :param userId:用户ID
        :param ratio: 正负反馈数目比率
        :return: 负反馈物品
        '''
        userItemlist = list(set(frame[frame['userId'] == userId]['movieId']))  # 用户评分过的物品
        otherItemList = [item for item in set(frame['movieId'].values) if item not in userItemlist]  # 用户没有评分的物品
        itemCount = [len(frame[frame['movieId'] == item]['userId']) for item in
                     otherItemList]  # 物品热门程度   (计数，userId未评分的物品被评分的次数)
        series = pd.Series(itemCount, index=otherItemList)  # 生成一个物品-热门程度的 元组 （itemid， ）
        series = series.sort_values(ascending=False)[:ratio * len(userItemlist)]  # 获取正反馈物品数量的负反馈物品 （正负反馈数目比率为ratio=1）
        negativeItemList = list(series.index)  # 获得负反馈物品的itemid 列表
        return negativeItemList

    @classmethod
    def __getUserPositiveItem__(cls, frame, userId):
        '''
        获取用户正反馈物品：用户评分过的物品
        :param frame: ratings数据
        :param userId: 用户ID
        :return: 正反馈物品
        '''
        series = frame[frame['userId'] == userId]['movieId']
        positiveItemList = list(series.values)
        return positiveItemList

    def __initUserItem__(self, frame, userId):
        '''
        初始化用户正负反馈物品,正反馈标签为1,负反馈为0
        :param frame: ratings数据
        :param userId: 用户ID
        :return: 正负反馈物品字典
        '''
        positiveItem = self.__getUserPositiveItem__(frame, userId)
        negativeItem = self.__getUserNegativeItem__(frame, userId, ratio=1)
        itemDict = {}
        for item in positiveItem: itemDict[item] = 1
        for item in negativeItem: itemDict[item] = 0
        return itemDict

    def __initUserItemPool__(self, frame, userId):
        '''
        初始化目标用户样本
        :param userId:目标用户
        :return:
        '''
        userItem = []
        for id in userId:
            itemDict = self.__initUserItem__(frame, userId=id)
            userItem.append({id: itemDict})
            print("userId = {}的user-Item完成".format(id))
        print("初始化目标用户样本完成")
        return userItem

    @classmethod
    def __initPara__(cls, userId, movieId, classCount):
        '''
        初始化参数q,p矩阵, 随机
        # :param userCount:用户ID
        # :param itemCount:物品ID
        :param classCount: 隐类数量
        :return: 参数p,q
        '''
        arrayp = np.random.rand(len(userId), classCount)  # 构造p矩阵，[0,1]内随机值
        arrayq = np.random.rand(classCount, len(movieId))  # 构造q矩阵，[0,1]内随机值
        p = pd.DataFrame(arrayp, columns=range(0, classCount), index=userId)
        q = pd.DataFrame(arrayq, columns=movieId, index=range(0, classCount))
        print("p,q 矩阵初始化完成")
        return p, q

    def initModel(self, classCount):
        '''
        初始化模型：参数p,q,样本数据
        :param frame: 源数据
        :param classCount: 隐类数量
        :return:
        '''
        frame = self.rating
        userId = list(set(frame['userId'].values))
        movieId = list(set(frame['movieId'].values))
        p, q = self.__initPara__(userId, movieId, classCount)  # 初始化p、q矩阵
        userItem = self.__initUserItemPool__(frame, userId)  # 建立用户-物品对应关系
        return p, q, userItem

    @classmethod
    def __sigmod__(cls, x):
        '''
        单位阶跃函数,将兴趣度限定在[0,1]范围内
        :param x: 兴趣度
        :return: 兴趣度
        '''
        y = 1.0 / (1 + exp(-x))
        return y

    def __lfmPredict__(self, p, q, userId, movieId):
        '''
        利用参数p,q预测目标用户对目标物品的兴趣度
        :param p: 用户兴趣和隐类的关系
        :param q: 隐类和物品的关系
        :param userId: 目标用户
        :param movieId: 目标物品
        :return: 预测兴趣度
        '''
        p = np.mat(p.loc[userId].values)
        q = np.mat(q[movieId].values).T
        r = (p * q).sum()
        r = self.__sigmod__(r)
        return r

    def latenFactorModel(self, p, q, userItem, classCount, iterCount, alpha, lamda):
        '''
        隐语义模型计算参数p,q
        :param frame: 源数据
        :param classCount: 隐类数量
        :param iterCount: 迭代次数
        :param alpha: 步长
        :param lamda: 正则化参数
        :return: 参数p,q
        '''

        for step in range(0, iterCount):
            for user in userItem:
                for userId, samples in user.items():
                    for movieId, rui in samples.items():
                        eui = rui - self.__lfmPredict__(p, q, userId, movieId)
                        for f in range(0, classCount):
                            print('step %d user %d class %d' % (step, userId, f))
                            p[f][userId] += alpha * (eui * q[movieId][f] - lamda * p[f][userId])
                            q[movieId][f] += alpha * (eui * p[f][userId] - lamda * q[movieId][f])
            alpha *= 0.9
        return p, q

    def rec(self, userId, TopN=50):
        '''
        推荐TopN个物品给目标用户
        :param frame: 源数据
        :param userId: 目标用户
        :param p: 用户兴趣和隐类的关系
        :param q: 隐类和物品的关系
        :param TopN: 推荐数量
        :return: 推荐物品
        '''
        frame = self.rating
        p, q = self.read_p_q()
        allItemList = set(frame['movieId'])
        predictList = [self.__lfmPredict__(p, q, userId, movieId) for movieId in allItemList]
        series = pd.Series(predictList, index=allItemList)
        series = series.sort_values(ascending=False)[:TopN]
        series = list(series.index)
        return series

    def Recall_Precision_Coverage(self, df_test, k):
        hit = 0
        all_p = 0
        all_r = 0
        rec_items = set()  # 推荐的物品总数 set（） # 集合
        df_userid = set(df_test['userid'])
        df_itemid = set(df_test['itemid'])
        for userid in df_userid:  # 610
            pre_item = self.rec(df_test, userid, TopN=k)  # 推荐给userid 的电影
            df_user_item = df_test.loc[df_test['userid'] == userid]
            true_item = df_user_item['itemid']  # userid 用户看过的所有电影
            print("userid:{}".format(userid))
            for itemid in pre_item:
                rec_items.add(itemid)  #
                if itemid in true_item:
                    hit += 1  # 推荐的电影在看过的电影里面的次数
            all_p += len(pre_item)  # 给Ueserid 电影长度 实际上就是k
            all_r += len(true_item)  # 给Ueserid 看过的电影的长度
        return hit / (all_r * 1.0), hit / (all_p * 1.0), len(rec_items) / (len(df_itemid) * 1.0)

    def keep_p_q(cls, p, q):
        p.to_csv('./recomend/data/LFM_p.csv', index=True)
        q.to_csv('./recomend/data/LFM_q.csv', index=True)

    @classmethod
    def read_p_q(cls):
        p = pd.read_csv('./recommend/data/LFM_p.csv', index_col=0)
        p.columns = list(map(int, p.keys()))
        q = pd.read_csv('./recommend/data/LFM_q.csv', index_col=0)
        q.columns = list(map(int, q.keys()))
        return p, q


# 基于用户的协同过滤推荐
class BasedUserRmd:
    def __init__(self):
        self.rating = pd.read_csv("./recommend/data/ratings.csv")

    def rel(self, id1, id2):
        i1 = self.rating.loc[self.rating['userId'] == id1]
        i2 = self.rating.loc[self.rating['userId'] == id2]
        nj = 0
        for mv1 in i1["movieId"]:
            if mv1 in i2["movieId"]:
                nj += 1
        m1 = i1.shape[0]
        m2 = i2.shape[0]
        return nj / sqrt(m1 + 1) / sqrt(m2 + 1)

    def find(self, id):
        res = np.zeros(max(self.rating["movieId"]))
        wt = 0
        for ids in range(1, 611):
            if ids == id:
                continue
            rl = self.rel(id, ids)
            wt += rl
            isd = self.rating.loc[self.rating["userId"] == ids]
            for i, dis in enumerate(isd["movieId"]):
                rting = isd["rating"]
                res[dis - 1] = res[dis - 1] + rl * rting.iloc[i]
        return res / wt

    def mx(self, id, n=50):
        ss = self.find(id)
        sr = [(i + 1, s) for i, s in enumerate(ss)]
        sr.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in sr[:n]]


# 基于统计的topn推荐
def topn(n, tag='all', indate="NULL", days=0):
    TAG = pd.read_csv("./recommend/data/tags.csv")
    if tag == 'all':
        tag = set(TAG['tag'])
    else:
        tag = [tag]
    # 根据tag搜寻点击排行前n部电影.
    if days != 0:
        # 输入需要以“年-月-日”的格式
        indate = time.strptime(indate, "%Y-%m-%d")
        indate = int(time.mktime(indate))
        before = indate - days * 60 * 60 * 24
        after = indate + days * 60 * 60 * 24
        data = [tag, before, after]
        # 考虑日期
        record_init = TAG[(TAG['tag'].isin(tag)) & (TAG['timestamp'] >= before) & (TAG['timestamp'] <= after)]
        record_init = record_init['movieId']
        records = np.array(record_init).tolist()
    else:
        # 不考虑日期
        record_init = TAG[TAG['tag'].isin(tag)]
        record_init = record_init['movieId']
        records = np.array(record_init).tolist()
    result = {}
    for i in records:
        result[i] = records.count(i)
    array = list(result.items())
    array.sort(key=lambda item: item[1], reverse=True)
    top = array[0:n]
    Top = list()
    for i in top:
        Top.append(i[0])
    return Top


class RmdAll(LFM, BasedUserRmd):
    def __init__(self):
        self.LINKS = pd.read_csv("./recommend/data/links.csv")
        self.MOVIE = pd.read_csv("./recommend/data/movies.csv")
        self.rating = pd.read_csv("./recommend/data/ratings.csv")

    def rec_all(self, userId, n1, n2, n3, tag='all', indate="2018-05-12", days=1080):
        topnfilm = topn(n=n1, tag=tag, indate=indate, days=days)
        lfmfilm = self.rec(userId=userId, TopN=n2)
        baseduserfilm = self.mx(id=userId, n=n3)
        res = topnfilm + lfmfilm + baseduserfilm
        return res

    def __findmovie__(self, movieid, count):
        count = count
        # 输入一个电影ID，返回电影的详细信息
        records1 = self.MOVIE[self.MOVIE["movieId"] == movieid]
        title = list(records1["title"])
        genres = list(records1["genres"])
        records2 = self.LINKS[self.LINKS["movieId"] == movieid]
        tmdbId = int(list(records2["tmdbId"])[0])
        result = (count, movieid, title, genres, tmdbId)
        return result

    def findallmovie(self, res):
        # 输入T电影IDlist，返回每部电影的详细信息
        infor = list()
        count = 0
        for i in res:
            count += 1
            infor.append(self.__findmovie__(i, count))
        return infor


rmdall = RmdAll()
