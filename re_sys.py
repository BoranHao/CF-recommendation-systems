#!usr/bin/env python
# -*- coding: utf-8 -*-


#-------------------------------------------------------------------------------
"""Three collaborative filtering recommendation system algorithms: 
User/ItemCF, TSVD, PMF. Data set is MovieLens 1M"""

__author__      = "Boran Hao"
__email__       = "brhao@bu.edu"
#-------------------------------------------------------------------------------


import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from itertools import combinations
import math
from numpy import linalg as la
import copy
import scipy.io as sio
import heapq
import random
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def ExtractFullMat(num=1100):
    '''Extract a full matrix from MovieLens Rating data
    num: first extract users rates at least num movies
    then for those extraced users, further extract full columns'''
    ratings = pd.read_table(
        'ratings.dat',
        sep='::',
        header=None,
        engine='python')
    rnames = ['user_id', 'movie_id', 'rating', 'timestamps']
    ratings.columns = rnames
    data = np.array(ratings)
    data = data[:, 0:3]
    row = data[:, 0].T
    col = data[:, 1].T
    rate = data[:, 2].T
    mat = coo_matrix((rate, (row, col)), shape=(6041, 3953))
    matr = mat.todense()

    uid = []
    for i in range(6041):
        if (np.nonzero(matr[i]))[0].shape[0] > num:
            uid.append(i)
    tp = matr.tolist()
    tpi = []
    for i in uid:
        tpi.append(tp[i])
    ldata = np.array(tpi)
    fi = []
    mid = []
    for i in range(3953):
        hang = ldata[:, i].T
        if (np.nonzero(hang))[0].shape[0] == len(uid):
            # print(i)
            mid.append(i)
            fi.append(hang.tolist())
    yu = np.array(fi, dtype=float).T
    return yu


def Scosine(A, B):
    '''return the cosine similarity of the WHOLE vectors'''
    if la.norm(A) * la.norm(B) == 0:
        if la.norm(A) == 0 and la.norm(B) == 0:
            co = 1
        else:
            co = 0

    if la.norm(A) * la.norm(B) != 0:
        co = np.dot(A, B) / (la.norm(A) * la.norm(B))
    return co


class Data():
    '''Build this class to store data, result
    using class methods to finish analysis'''

    def __init__(self, mat):
        self.mat = mat
        self.meann = []
        self.NNindex = []
        self.Smat = []
        self.CF = False
        self.SVD = False
        self.PMF = False
        self.CFmat = []
        self.SVDmat = []
        self.PMFmat = []
        self.original = copy.deepcopy(mat)
        self.SVDcomp = [0, -10]
        self.PMFcomp = [0, -10]
        self.PMFcomp2 = [0, 1000]

    def NullIndexGenerator(self, lack, unif=True):
        '''General null position index randomly, record those index
        set null values 0, but 0 is just a symbol. NullIndex is what to
        be used
        lack: sparsity
        unif: decide if we remove values totally randomly, or
        using some pattern to make biased data. This part is our future work
        and not yet finished'''
        nullnum = int(self.mat.shape[0] * self.mat.shape[1] * lack)
        random.seed(22)
        ran = []
        while True:
            ran.append(
                (int(self.mat.shape[0] * random.random()), int(self.mat.shape[1] * random.random())))
            ran = list(set(ran))
            if len(ran) == nullnum:
                break
        self.nullind = ran
        for tu in self.nullind:
            self.mat[tu[0], tu[1]] = 0

    def Normalize(self):
        '''centralize the rating for each user'''
        mat = copy.deepcopy(self.mat)
        rmt = copy.deepcopy(self.original)
        meann = []
        stdd = []

        for i in range(mat.shape[0]):
            rmt[i, :] = rmt[i, :] - np.mean(rmt[i, :])
            tp = 0
            tn = 0
            for j in range(mat.shape[1]):
                if mat[i, j] != 0:
                    tp = tp + mat[i, j]
                    tn = tn + 1
            if tn == 0:
                meann.append(0)
            else:
                meann.append(tp / tn)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                # rmt[i,j]=rmt[i,j]-meann[i]
                if (i, j) not in self.nullind:
                    mat[i, j] = mat[i, j] - meann[i]
            # mat[i,:]=mat[i,:]/np.std(mat[i,:])

        self.Noriginal = rmt
        self.meann = meann
        self.mat = mat

    def Nindex(self, k=15, typee='cosine'):
        '''Build W matrix and find the nearest neighbors
        k: nearest neighbor number
        typee: similarity measurement. cosine(Pearson Correlation)
        distance(partial cosine)...'''
        mat = copy.deepcopy(self.mat)
        num = mat.shape[0]

        Usernullind = {}
        for tu in self.nullind:
            Usernullind[tu[0]] = []
        for tu in self.nullind:
            Usernullind[tu[0]].append(tu)

        self.UserNullInd = Usernullind
        # print(num)
        Dmat = np.zeros([num, num])
        for i in range(num):
            for j in range(num):
                if i == j:
                    Dmat[i, j] = -999999
                else:
                    if typee == 'cosine':
                        self.simtype = 'cosine'
                        Dmat[i, j] = Scosine(mat[i, :], mat[j, :])
                    if typee == 'distance':
                        self.simtype = 'distance'
                        delta = 0
                        sumd = 0
                        hang1 = []
                        hang2 = []
                        for lie in range(self.mat.shape[1]):
                            try:
                                if (i, lie) not in Usernullind[i]:
                                    if (j, lie) not in Usernullind[j]:
                                        hang1.append(mat[i, lie])
                                        hang2.append(mat[j, lie])
                                        delta = delta + 1
                            except KeyError:
                                continue
                        hang1 = np.array(hang1)
                        hang2 = np.array(hang2)
                        if delta == 0:
                            Dmat[i, j] = 0
                        else:
                            # dist=sumd/delta
                            Dmat[i, j] = Scosine(hang1, hang2)
                            # Dmat[i,j]=math.exp(-(la.norm(hang1-hang2)**2/delta))
                            # # Gaussian similarity

        ind = []
        for i in range(num):
            tp = []
            hang = {}
            for j in range(num):
                hang[j] = Dmat[i][j]
            while True:
                aa = max(hang, key=hang.get)
                tp.append(aa)
                del hang[aa]
                if len(tp) == k:
                    break
            ind.append(tp)
        self.NNindex = ind
        self.Smat = Dmat

    def GetCFresult(self, LeastSim=0):
        '''Using Nearest Neighbor CF to predict the null ratings
        LeastSim: the smallest similarity to be 'related'''
        tpmat = copy.deepcopy(self.mat)
        dic = {}
        for tu in self.nullind:
            j = tu[1]
            tp = []
            for i in range(self.mat.shape[0]):
                if i in self.NNindex[tu[0]]:
                    if (i, j) not in self.nullind:
                        if self.Smat[tu[0], i] > LeastSim:
                            tp.append((i, j))
            dic[tu] = tp
        for tu in self.nullind:
            if len(dic[tu]) != 0:
                summ = 0
                sunn = 0
                for pt in dic[tu]:
                    summ = summ + self.mat[pt[0], pt[1]
                                           ] * self.Smat[pt[0], tu[0]]
                    sunn = sunn + self.Smat[pt[0], tu[0]]
                pred = summ / sunn
                #print('Predict for '+str(tu)+':'+str(pred))
                #print('Real for '+str(tu)+':'+str(self.Noriginal[tu[0],tu[1]]))
                tpmat[tu[0], tu[1]] = pred
        self.dict = dic  # record those been rated
        self.CFmat = tpmat
        self.CF = True

    def GetSVDresult(self, Num=5, e=10):
        '''Truncated SVD to predict,
        Num: largest singular values number remained
        e: iterative TSVD, when data sparsity is low, using TSVD until
        prediction converge can improve the performance, the defalut setting
        is no iteration'''
        A = copy.deepcopy(self.mat)

        while True:

            del(self.SVDcomp[0])
            self.SVDcomp.append(A)

            ast = np.fabs(self.SVDcomp[0] - self.SVDcomp[1]) <= e
            # print(ast)
            if ast.all():
                break

            # print(A[7,40])
            U, sigma, VT = la.svd(A)
            tp = list(sigma)
            while True:
                if len(tp) == self.mat.shape[1]:
                    break
                tp.append(0)
            sigma = np.array(tp)

            sigma = np.diag(sigma)
            sigma = sigma[0:self.mat.shape[0], :]
            for i in range(Num, self.mat.shape[0]):
                sigma[i, i] = 0
            A = U@sigma@VT

            for i in range(self.mat.shape[0]):
                for j in range(self.mat.shape[1]):
                    if (i, j) not in self.nullind:
                        A[i, j] = self.mat[i, j]
                    if A[i, j] > 4:
                        A[i, j] = 4
                    if A[i, j] < -4:
                        A[i, j] = -4
        self.SVDmat = A
        self.SVD = True

    def GetPMFresult(self, k=1, e=0.00001, alpha=0.01, c1=0, c2=0):
        '''icremental gradient method to sovle PMF optimization problem
        k: latent factor number
        e: decide stop criterion
        alpha: stepsize
        c1,c2: regularization parameters'''
        m = self.mat.shape[0]
        n = self.mat.shape[1]
        self.time = 0

        indlist = []
        for i in range(m):
            for j in range(n):
                if (i, j) not in self.nullind:
                    indlist.append((i, j))
        # p=np.ones([m,k])*0.1
        # q=np.ones([k,n])*(-0.1)
        np.random.seed(5)
        u = np.random.rand(m, k) - 0.5
        v = np.random.rand(n, k) - 0.5
        self.train = []
        self.test = []
        self.ax = []

        while True:
            # for tt in range(300):   # we can also fix iteration times
            # self.ax.append(tt)

            tpp = u@v.T
            cost = 0
            nnn = 0
            for tu in indlist:
                cost = cost + (self.mat[tu[0], tu[1]] - tpp[tu[0], tu[1]])**2
                nnn = nnn + 1

            te = (cost / nnn)**0.5  # training RMSE for each iteration
            self.train.append(te)
            # print(te)

            PMFm = 0
            PMFn = 0
            for tu in self.nullind:
                PMFm = PMFm + \
                    math.fabs(
                        self.Noriginal[tu[0], tu[1]] - tpp[tu[0], tu[1]])**2
                PMFn = PMFn + 1
            tse = (PMFm / PMFn)**0.5  # test RMSE for each iteration
            self.test.append(tse)
            # print(tse)

            self.time += 1
            del(self.PMFcomp[0])
            self.PMFcomp.append(cost)

            if np.fabs(self.PMFcomp[0] - self.PMFcomp[1]) <= e:
                break

            for tu in indlist:
                flag = self.mat[tu[0], tu[1]] - np.dot(u[tu[0]], v[tu[1]])
                for r in range(k):
                    u[tu[0]][r] = u[tu[0]][r] + alpha * \
                        flag * (v[tu[1]][r] - c1 * u[tu[0]][r])
                    v[tu[1]][r] = v[tu[1]][r] + alpha * \
                        flag * (u[tu[0]][r] - c2 * v[tu[1]][r])

        fi = u@v.T
        for i in range(m):
            for j in range(n):
                if (i, j) not in self.nullind:
                    fi[i, j] = self.mat[i, j]
                if fi[i, j] > 4:
                    fi[i, j] = 4
                if fi[i, j] < -4:
                    fi[i, j] = -4
        self.PMFmat = fi
        self.PMF = True
        # print(fi)

    def GetRe(self, ReNum=3):
        '''Using the prediction to make recommendation
        ReNum: max item number to be recommended to each user'''

        # 1.SVD
        if self.SVD:
            SVDre = {}
            SVDtp = {}
            SVDlist = []
            for tu in self.nullind:
                SVDre[tu[0]] = []
                SVDtp[tu[0]] = []
            for tu in self.nullind:
                SVDtp[tu[0]].append(tu)
            for user in SVDtp.keys():
                tp1 = {}
                for tup in SVDtp[user]:
                    tp1[tup] = self.SVDmat[tup[0], tup[1]]
                while True:
                    if len(tp1.keys()) == 0:
                        break
                    aa = max(tp1, key=tp1.get)
                    if self.SVDmat[aa[0], aa[1]] > 0:
                        SVDre[user].append(aa)
                        del tp1[aa]
                        if len(SVDre[user]) == ReNum:
                            break
                    if self.SVDmat[aa[0], aa[1]] <= 0:
                        break
            for i in SVDre.keys():
                for tuu in SVDre[i]:
                    SVDlist.append(tuu)
            # print(SVDlist)
            self.SVDreList = SVDlist

        # 2.PMF
        if self.PMF:
            PMFre = {}
            PMFtp = {}
            PMFlist = []
            for tu in self.nullind:
                PMFre[tu[0]] = []
                PMFtp[tu[0]] = []
            for tu in self.nullind:
                PMFtp[tu[0]].append(tu)
            for user in PMFtp.keys():
                tp2 = {}
                for tup in PMFtp[user]:
                    tp2[tup] = self.PMFmat[tup[0], tup[1]]
                while True:
                    if len(tp2.keys()) == 0:
                        break
                    aa = max(tp2, key=tp2.get)
                    if self.PMFmat[aa[0], aa[1]] > 0:
                        PMFre[user].append(aa)
                        del tp2[aa]
                        if len(PMFre[user]) == ReNum:
                            break
                    if self.PMFmat[aa[0], aa[1]] <= 0:
                        break
            for i in PMFre.keys():
                for tuu in PMFre[i]:
                    PMFlist.append(tuu)
            # print(PMFlist)
            self.PMFreList = PMFlist

        # 3.CF
        if self.CF:
            CFre = {}
            CFtp = {}
            CFlist = []
            for tu in self.dict.keys():
                CFre[tu[0]] = []
                CFtp[tu[0]] = []
            for tu in self.dict.keys():
                CFtp[tu[0]].append(tu)
            for user in CFtp.keys():
                tp3 = {}
                for tup in CFtp[user]:
                    tp3[tup] = self.CFmat[tup[0], tup[1]]
                while True:
                    if len(tp3.keys()) == 0:
                        break
                    aa = max(tp3, key=tp3.get)
                    if self.CFmat[aa[0], aa[1]] > 0:
                        CFre[user].append(aa)
                        del tp3[aa]
                        if len(CFre[user]) == ReNum:
                            break
                    if self.CFmat[aa[0], aa[1]] <= 0:
                        break
            for i in CFre.keys():
                for tuu in CFre[i]:
                    CFlist.append(tuu)
            self.CFreList = CFlist
            self.RE = True

    def RMSE(self):
        '''Compute RMSE of the corresponding methods
        AVG: baseline RMSE'''
        RMSE = {'RMSE': '', 'CF': [], 'SVD': [], 'PMF': [], 'AVG': []}
        if self.CF:
            CFm = 0
            CFn = 0
            for tu in self.nullind:
                if len(self.dict[tu]) != 0:
                    CFm = CFm + \
                        math.fabs(
                            self.Noriginal[tu[0], tu[1]] - self.CFmat[tu[0], tu[1]])**2
                    CFn = CFn + 1
            RMSE['CF'] = (CFm / CFn)**0.5

        if self.SVD:
            SVDm = 0
            SVDn = 0
            for tu in self.nullind:
                SVDm = SVDm + \
                    math.fabs(
                        self.Noriginal[tu[0], tu[1]] - self.SVDmat[tu[0], tu[1]])**2
                SVDn = SVDn + 1
            RMSE['SVD'] = (SVDm / SVDn)**0.5

        if self.PMF:
            PMFm = 0
            PMFn = 0
            for tu in self.nullind:
                PMFm = PMFm + \
                    math.fabs(
                        self.Noriginal[tu[0], tu[1]] - self.PMFmat[tu[0], tu[1]])**2
                PMFn = PMFn + 1
            RMSE['PMF'] = (PMFm / PMFn)**0.5

        AVGm = 0
        AVGn = 0
        for tu in self.nullind:
            AVGm = AVGm + \
                math.fabs(self.Noriginal[tu[0], tu[1]
                                         ] - self.mat[tu[0], tu[1]])**2
            AVGn = AVGn + 1
        RMSE['AVG'] = (AVGm / AVGn)**0.5

        self.RMSE = RMSE
        print(RMSE)

    def CVG(self):
        '''Computing coverage of different methods, using the
        recommendation result given by function GetRe'''
        CVG = {'CVG': '', 'CF': [], 'SVD': [], 'PMF': []}
        total = self.mat.shape[1]
        if self.CF:
            tp = []
            for tu in self.CFreList:
                tp.append(tu[1])
            part = list(set(tp))
            CVG['CF'] = len(part) / total

        if self.SVD:
            tp = []
            for tu in self.SVDreList:
                tp.append(tu[1])
            part = list(set(tp))
            CVG['SVD'] = len(part) / total

        if self.PMF:
            tp = []
            for tu in self.PMFreList:
                tp.append(tu[1])
            part = list(set(tp))
            CVG['PMF'] = len(part) / total
        self.CVG = CVG
        print(CVG)


#data = sio.loadmat('yu.mat')
#yu = data['yuan']

def main():
    yu = ExtractFullMat()

    ob = Data(yu)
    ob.NullIndexGenerator(0.1)

    ob.Normalize()
    ob.Nindex(19, 'cosine')

    ob.GetCFresult(0)
    ob.GetSVDresult(2, 1)
    ob.GetPMFresult(1, 0.0001, 0.01, 0.01, 0.01)  # k e alpha

    ob.GetRe(2)
    ob.RMSE()
    ob.CVG()

    # sio.savemat('fSVD.mat',{'RMSE': RMSE})
    # sio.savemat('IC2.mat',{'CVG': CVG})
    # sio.savemat('UA2.mat',{'AVG': AVG})
    # sio.savemat('null.mat',{'null': nullrate})

    '''axx=[0,100,200,250,300]
    av=[ob.RMSE['AVG'],ob.RMSE['AVG'],ob.RMSE['AVG'],ob.RMSE['AVG'],ob.RMSE['AVG']]

    plt.figure()

    plt.plot(ob.ax,ob.train,color='blue',label='RMSE: Training')
    plt.plot(ob.ax,ob.test,color='green',label='RMSE: Testing')
    plt.plot(axx,av,color='red',linestyle='-.',label='Baseline')
    #plt.plot(nullrate,FSC2,color='blue',linestyle='-.',label='CCR: Partial Cosine')

    plt.xlabel("Iteration number, with sparsity 0.1, Lu=Lv=0")
    plt.ylabel("Performance Metrics")
    plt.title("Training and test error")
    plt.legend()
    plt.grid(True)
    plt.show()'''


if __name__ == "__main__":
    main()

