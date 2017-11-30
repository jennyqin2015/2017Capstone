import pandas as pd
import cvxopt
# import pandas_datareader.data as web
import datetime
import numpy as np
'''
% matplotlib
inline
import matplotlib.pyplot as plt
% pylab
inline
'''
# import statsmodels.api as sm
##from statsmodels.tsa.api import VAR, DynamicVAR
# import statsmodels.tsa.vector_ar.var_model
import time
from cvxopt.modeling import variable
from cvxopt.modeling import op, dot
from cvxopt import matrix
# import pandas_montecarlo
from cvxopt.modeling import op

# start = datetime.datetime(2012,1,1)
# end = datetime.datetime(2017,10,1)

###########################################################################################
'''
data
processing(test)

XIC = pd.read_csv('XIC.csv')
SHY = pd.read_csv('SHY.csv')
VTI = pd.read_csv('VTI.csv')
ZHY = pd.read_csv('ZHY.csv')
VUS = pd.read_csv('VUS.csv')
IEMG = pd.read_csv('IEMG.csv')
IEFA = pd.read_csv('IEFA.csv')
ZCS = pd.read_csv('ZCS.csv')
ZFM = pd.read_csv('ZFM.csv')
PDF = pd.read_csv('PDF.csv')
DGS10 = pd.read_csv('DGS10.csv')
SNP = pd.read_csv('SNP.csv')
T10 = pd.read_csv('T10.csv')
IWF = pd.read_csv('IWF.csv')
IVV = pd.read_csv('IVV.csv')
QQQ = pd.read_csv('QQQ.csv')
VUG = pd.read_csv('VUG.csv')

XIC = XIC['Adj Close']
SHY = SHY['Adj Close']
VTI = VTI['Adj Close']
ZHY = ZHY['Adj Close']
VUS = VUS['Adj Close']
IEMG = IEMG['Adj Close']
IEFA = IEFA['Adj Close']
ZCS = ZCS['Adj Close']
ZFM = ZFM['Adj Close']
PDF = PDF['Adj Close']
IWF = IWF['Adj Close']
IVV = IVV['Adj Close']
QQQ = QQQ['Adj Close']
#
DGS10 = DGS10['DGS10']
T10 = T10['T10']
SNP = SNP['Adj Close']
VUG = VUG['Adj Close']

XIC = XIC[0:1200]
VTI = VTI[0:1200]
ZHY = ZHY[0:1200]
VUS = VUS[0:1200]
IEMG = IEMG[0:1200]
IEFA = IEFA[0:1200]
ZCS = ZCS[0:1200]
ZFM = ZFM[0:1200]
# PDF = PDF[0:1200]
DGS10 = DGS10[0:1200]
T10 = T10[0:1200]
SNP = SNP[0:1200]
SHY = SHY[0:1200]
IWF = IWF[0:1200]
IVV = IVV[0:1200]
QQQ = QQQ[0:1200]
VUG = VUG[0:1200]

stock_r = pd.DataFrame({'VTI': VTI, 'QQQ': QQQ, 'IVV': IVV, 'VUG': VUG, 'IWF': IWF})
stock_k = pd.DataFrame({'XIC': XIC, 'VTI': VTI, 'SHY': SHY, 'VUS': VUS, 'IEFA': IEFA, 'ZCS': ZCS, 'ZFM': ZFM})
stock_k = stock_k.apply(lambda x: np.log(x) - np.log(x.shift(1)))
stock_r = stock_r.apply(lambda x: np.log(x) - np.log(x.shift(1)))
'''

############################################################################################
class stochastic:
    def __init__(self, initial, year, stock, goal, r, q, index, Inj):

        # Initial: A float, the initial wealth that an investor wants to invest in
        # year: A float, the time needed to achieve the goal
        # stock: A panda dataframe of assets return
        # goal: A float, the number of wealth that an investor wants to achieve
        # r: A float, reward surplus
        # q: A float, storage penalty
        # index: An array of market index prices for each quarter of each year.
        # example [ [1,2,3,4,5,6] ]

        # year 1 market index prices are 1(quarter1),2(quarter2),3(quarter3),4(quarter4)
        # year 2 market index prices are 5(quarter1),6(quarter2),7(quarter3),8(quarter4)

        self.stock = stock
        self.goal = goal
        self.year = year
        self.initial = initial
        self.T = 4
        self.r = r
        self.q = q
        self.index = index
        self.Inj = Inj

    def split_goal(self):
        # split a big goal into multiple small goals achieved at the end of each year


        I = self.initial
        lis = []
        a = (self.goal / I) ** (1 / self.year)

        for i in range(self.year):
            I *= a
            lis.append(I)

        return lis

    def mean(self):
        # stock is a data frame(maybe a dataseris) daily data
        # return an array(numpy array) of stock return


        mean = self.stock.mean()
        mean = mean * 252  # annulized return
        mean = np.array(mean)

        return mean

    def vio(self):
        # stock is a data frame(maybe a dataseris) daily data
        # return an array(numpy array) of stock return

        cov = self.stock.cov()
        c = np.matrix((252 ** (1 / 2)) * (cov) ** (1 / 2))
        m, m = c.shape
        lis = []
        for i in range(m):
            lis.append(c[i, i])

        vio = np.array(lis)

        return vio

    def var(self):
        cov = self.stock.cov()
        c = np.matrix(252 * (cov))
        m, m = c.shape
        lis = []
        for i in range(m):
            lis.append(c[i, i])

        var = np.array(lis)

        return var

    def scen(self):
        # stock is a data frame(maybe a dataseries) daily data
        # return an N*2 array. N is # of assets. 2 refers to up and down total return.
        m, N = self.stock.shape
        scen = np.zeros((N, 3))
        r = self.mean()
        sigma = self.vio()
        for i in range(N):
            for j in range(2):
                if j % 2 == 0:
                    scen[i, j] = np.exp(((r[i] ** 2) * ((1 / 4) ** 2) + (sigma[i] ** 2) * (1 / 4)) ** (1 / 2))
                if j % 2 == 1:
                    scen[i, j] = np.exp(-((r[i] ** 2) * ((1 / 4) ** 2) + (sigma[i] ** 2) * (1 / 4)) ** (1 / 2))
            scen[i, 2] = exp(r[i] * (1 / 4))
        return scen

    def WW(self):
        lis = []
        for k in range(2 ** self.T):
            lis.append(variable(1, 'x'))
        return lis

    def YY(self):
        lis = []
        for k in range(2 ** self.T):
            lis.append(variable(1, 'x'))
        return lis

    def vari(self):
        m, N = self.stock.shape
        lis = []
        liss = []
        for j in range(2 ** self.T - 1):
            for i in range(N):
                lis.append(variable(1, 'x'))

            liss.append(lis)
            lis = []

        return liss

    def norm(self):
        m, N = self.stock.shape

        scen = np.zeros((N, 2))
        r = self.mean()
        sigma = self.vio()
        for i in range(N):
            for j in range(2):
                if j % 2 == 0:
                    scen[i, j] = r[i] / 4
                if j % 2 == 1:
                    scen[i, j] = sigma[i] / 2

        return scen

    def constriant2(self, R, X, I, T, w, Y, G,
                    Inj):  # R return for each asset n*2 matrix # X each asset at each period ,#G, T each period
        # w
        # y


        cons = []
        a = 0
        N, r = R.shape
        # t = 1 (0)

        for i in range(N):
            a += X[0][i]

        # a = np.sum(np.array(X[0][0]))

        cons.append(a == I + Inj)
        a = 0
        b = 0
        c = 0
        d = 0

        for j in range(2 ** T - 1 - 2 ** (T - 1)):

            for i in range(N):
                for lll in range(10000):
                    if R[i, 0] > 0.01:
                        ss = np.random.normal(R[i, 0], R[i, 1], 1)
                        if 1.3 * R[i, 1] + R[i, 0] > ss[0] > 0.0025:
                            break
                        else:
                            continue
                    else:
                        ss = np.random.normal(R[i, 0], R[i, 1], 1)
                        break
                a += X[j][i] * float(np.exp(ss[0]))  # np.random.uniform(R[i,2],R[i,0])

                # a += X[0][s][i]*float(R[i,0])
                b += X[2 * j + 1][i]
                for lll in range(10000):
                    if R[i, 0] > 0.01:
                        kk = np.random.normal(R[i, 0], R[i, 1], 1)
                        if kk[0] < 0.0025:
                            break
                        else:
                            continue
                    else:
                        kk = np.random.normal(R[i, 0], R[i, 1], 1)
                        break
                c += X[j][i] * float(np.exp(kk[0]))
                # a += X[0][s][i]*float(R[i,1])
                d += X[2 * j + 2][i]
                cons.append(X[j][i] >= 0)

            cons.append(a + Inj == b)
            cons.append(c + Inj == d)

            a = 0
            b = 0
            c = 0
            d = 0

        for j in range(2 ** T - 1 - 2 ** (T - 1), 2 ** T - 1):

            k = 2 * j + 1 - (2 ** T - 1)
            l = 2 * j + 2 - (2 ** T - 1)

            for i in range(N):
                for lll in range(10000):
                    if R[i, 0] > 0.01:
                        ss = np.random.normal(R[i, 0], R[i, 1], 1)
                        if 1.3 * R[i, 1] + R[i, 0] > ss[0] > 0.0025:
                            break
                        else:
                            continue
                    else:
                        ss = np.random.normal(R[i, 0], R[i, 1], 1)
                        break
                a += X[j][i] * float(np.exp(ss[0]))  # np.random.uniform(R[i,2],R[i,0])

                for lll in range(10000):
                    if R[i, 0] > 0.01:
                        kk = np.random.normal(R[i, 0], R[i, 1], 1)
                        if kk[0] < 0.0025:
                            break
                        else:
                            continue
                    else:
                        kk = np.random.normal(R[i, 0], R[i, 1], 1)
                        break
                c += X[j][i] * float(np.exp(kk[0]))  # np.random.uniform(R[i,1],R[i,2])

                cons.append(X[j][i] >= 0)

            a += (-Y[k] + w[k])
            c += (-Y[l] + w[l])
            cons.append(Y[k] >= 0)
            cons.append(Y[l] >= 0)
            cons.append(w[k] >= 0)
            cons.append(w[l] >= 0)
            cons.append(a == G)
            cons.append(c == G)
            b = 0
            d = 0
            a = 0
            c = 0

        return cons

    def objective(self, w, Y, q, r, T):
        a = 0
        b = 0
        for i in range(len(w)):
            a += -r * w[i]
            b += q * Y[i]

            c = -(a + b) * ((1 / 2) ** T)

        return c

    def func(self, R, X, I, T, w, Y, G, q, r, Inj):
        N, m = R.shape
        step = 15
        res = np.zeros((2 ** T - 1, N))
        kk = np.zeros((2 ** T - 1, N))
        a = 0
        for i in range(step):
            cons = self.constriant2(R, X, I, T, w, Y, G, Inj)
            bbb = self.objective(w, Y, q, r, T)
            lp2 = op(bbb, cons)
            lp2.solve()
            for i in range(2 ** T - 1):
                for j in range(N):
                    kk[i][j] = np.array(X[i][j].value)[0][0]

            for i in range(2 ** T):
                a += np.array((Y[i].value - w[i].value))[0][0]

            res += kk

        a = a * ((1 / 2) ** T) * (1 / step) + G

        res = res * (1 / step)
        return [res, a]

    def fcs(self):
        # return all optimal strategies
        k_list = self.split_goal()
        I0 = self.initial - self.Inj
        R = self.norm()
        T = self.T
        w = self.WW()
        Y = self.YY()
        X = self.vari()
        q = self.q
        r = self.r
        m, N = R.shape
        res = []
        rar = np.array([])
        cur_I = I0
        for i in range(len(k_list)):
            cur_K = k_list[i]
            cur_S, cur_I = self.func(R, X, int(cur_I), T, w, Y, cur_K, q, r, self.Inj)
            res.append(cur_S)
            if i == len(k_list) - 1:
                res.append(cur_I)
        return res

    def c_array(self):
        a = self.index
        a = a.tolist()

        return [a[i:i + 4] for i in range(0, len(a), 4)]

    def get_node(self):
        # get a list of index price
        # return the optimal strategy index
        big_lis = self.c_array()
        print(big_lis)
        final_lis = []
        for j in range(len(big_lis)):
            family_list = [0]
            price_lis = big_lis[j]
            for i in range(1, len(price_lis)):
                cur_node = price_lis[i]
                pre_node = price_lis[i - 1]

                if cur_node > pre_node:
                    cur_node = family_list[i - 1] * 2 + 1
                else:
                    cur_node = family_list[i - 1] * 2 + 2

                family_list.append(cur_node)

            final_lis.append(family_list)

        return final_lis

    def strategy(self):
        # return the optimal strategy which is a list
        # the last element of the list is the estimated wealth achieved at the end of the investment period
        # others are lists of arrays containing the optimal strategy at each year
        # example: [[array(1),array(2),array(3),array(4)],[array(1),array(2),array(3),array(4)],[array(1),array(2),array(3),array(4)],1000]
        # array(i) includes all the strategy at each quarter(i)
        # 1000 is the final wealth achieved

        res = self.fcs()
        expected = res[-1]

        res.pop(-1)

        stra = res

        optimal_index = self.get_node()

        liss = []

        a = min(len(stra), len(optimal_index))
        for i in range(a):
            lis = []
            op_id = optimal_index[i]
            op_stra = stra[i]
            for j in range(len(op_id)):
                k = op_id[j]
                lis.append(op_stra[k])
            liss.append(lis)
        liss.append(expected)

        return liss

    def weights(self):
        lis = self.strategy()
        lis.pop(-1)
        nlis = lis
        print(len(nlis))
        for i in range(len(nlis)):
            for j in range(len(nlis[i])):
                nlis[i][j] = nlis[i][j] / np.sum(nlis[i][j])

        return nlis