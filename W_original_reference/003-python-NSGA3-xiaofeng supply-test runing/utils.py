# -*- coding: utf-8 -*-
"""
程序功能：论文复现
论文信息：
An Evolutionary Many-Objective Optimization Algorithm Using Reference-point Based Non-dominated Sorting Approach, Part I: Solving Problems with Box Constraint
作者：(晓风)wangchao
最初建立时间：2019.03.26
最近修改时间：2019.03.27
最小化问题：DTLZ1,DTLZ2,DTLZ3
NSGA3的简单实现
"""
from scipy.special import comb
from itertools import combinations
import numpy as np
import copy
import math

def uniformpoint(N,M):
    H1=1
    while (comb(H1+M-1,M-1)<=N):
        H1=H1+1
    H1=H1-1
    W=np.array(list(combinations(range(H1+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H1+M-1,M-1)),1))
    W=(np.hstack((W,H1+np.zeros((W.shape[0],1))))-np.hstack((np.zeros((W.shape[0],1)),W)))/H1
    if H1<M:
        H2=0
        while(comb(H1+M-1,M-1)+comb(H2+M-1,M-1) <= N):
            H2=H2+1
        H2=H2-1
        if H2>0:
            W2=np.array(list(combinations(range(H2+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H2+M-1,M-1)),1))
            W2=(np.hstack((W2,H2+np.zeros((W2.shape[0],1))))-np.hstack((np.zeros((W2.shape[0],1)),W2)))/H2
            W2=W2/2+1/(2*M)
            W=np.vstack((W,W2))#按列合并
    W[W<1e-6]=1e-6
    N=W.shape[0]
    return W,N

def funfun(M,N,name):
    #种群初始化
    D=M+4#定义自变量个数为目标个数加4
    low=np.zeros((1,D))
    up=np.ones((1,D))
    pop = np.tile(low,(N,1))+(np.tile(up,(N,1))-np.tile(low,(N,1)))*np.random.rand(N,D)

    # N = 4, M = 3
    # pop=np.array([[0.915192859,0.028727049,0.607623827,0.479063179,0.206890535,0.3339074,0.625488188],
    #     [0.614069404,0.702568007,0.634910672,0.789036154,0.40688523,0.232214776,0.060136365],
    #     [0.556674654,0.088459678,0.780410954,0.457283871,0.021801791,0.263091797,0.241113907]])

    # N = 10, M = 8
    #test value
    pop=np.array([[0.536670256,0.964008663,0.328074671,0.795736011,0.611979939,0.416348378,0.721122666,0.591319176,0.148780267,0.470257077,0.33385828,0.357379456],
[0.330122966,0.467156812,0.464288412,0.594115071,0.971130391,0.04400558,0.295053079,0.731548924,0.892694749,0.844722789,0.820193364,0.721198203],
[0.50930279,0.574945265,0.839338753,0.336512317,0.571693052,0.724383644,0.397724115,0.474396526,0.677331281,0.742628266,0.241313572,0.362475023],
[0.990179757,0.101665098,0.095609143,0.991536875,0.938234005,0.943046877,0.847108978,0.599045578,0.851177758,0.238282134,0.573930608,0.120977838],
[0.728028879,0.502546086,0.853794269,0.201726141,0.923346543,0.252173433,0.76302695,0.208720562,0.910886964,0.568307209,0.151518037,0.352788854],
[0.631061767,0.495882541,0.843099449,0.073345594,0.199840554,0.299829067,0.099449554,0.312643064,0.259084465,0.973921013,0.477105295,0.124218861],
[0.471947378,0.295603416,0.507458351,0.737000875,0.633039016,0.653803646,0.508372405,0.564803018,0.33925357,0.443474945,0.302883491,0.339089051],
[0.166939806,0.782189015,0.667332862,0.546467366,0.969729699,0.763327228,0.339952496,0.918921422,0.809726944,0.452821296,0.058341928,0.297880213]])

    #计算PF
    if name=='DTLZ1':
        #g=np.transpose(np.mat(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))))
        g=np.array(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))).reshape(N,1)
        popfun=np.multiply(0.5*np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),pop[:,:(M-1)]))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),1-np.fliplr(pop[:,:(M-1)])))))
        P,nouse = uniformpoint(N,M)
        P=P/2
    elif name=='DTLZ2':
        #g=np.transpose(np.mat(np.sum((pop[:,(M-1):]-0.5)**2,1)))
        g=np.array(np.sum((pop[:,(M-1):]-0.5)**2,1)).reshape(N,1)
        popfun=np.multiply(np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),np.cos(pop[:,:(M-1)]*(np.pi/2))))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),np.sin(np.fliplr(pop[:,:(M-1)])*(np.pi/2))))))  #去掉1-
        P,nouse = uniformpoint(N,M)        
        #P = P/np.tile(np.transpose(np.mat(np.sqrt(np.sum(P**2,1)))),(1,M))
        P = P/np.tile(np.array(np.sqrt(np.sum(P**2,1))).reshape(P.shape[0],1),(1,M))
    elif name=='DTLZ3':
        g=np.array(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))).reshape(N,1)
        popfun=np.multiply(np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),np.cos(pop[:,:(M-1)]*(np.pi/2))))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),np.sin(np.fliplr(pop[:,:(M-1)])*(np.pi/2))))))   #去掉1-
        P,nouse = uniformpoint(N,M)        
        #P = P/np.tile(np.transpose(np.mat(np.sqrt(np.sum(P**2,1)))),(1,M))
        P = P/np.tile(np.array(np.sqrt(np.sum(P**2,1))).reshape(P.shape[0],1),(1,M))
        
    return pop,popfun,P,D

def cal(pop,name,M,D):
    N = pop.shape[0]
    if name=='DTLZ1':
        g=np.array(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))).reshape(N,1)
        popfun=np.multiply(0.5*np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),pop[:,:(M-1)]))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),1-np.fliplr(pop[:,:(M-1)])))))

    elif name=='DTLZ2':
        g=np.array(np.sum((pop[:,(M-1):]-0.5)**2,1)).reshape(N,1)
        popfun=np.multiply(np.tile(1+g,(1,M)),(np.fliplr((np.hstack((np.ones((g.shape[0],1)),np.cos(pop[:,:(M-1)]*(np.pi/2))))).cumprod(1))))
        popfun=np.multiply(popfun,(np.hstack((np.ones((g.shape[0],1)),np.sin(np.fliplr(pop[:,:(M-1)])*(np.pi/2)))))) #去掉1-

    elif name=='DTLZ3':
        g=np.array(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))).reshape(N,1)
        popfun=np.multiply(
            np.tile(1+g,(1,M)),
            (np.fliplr(
                (np.hstack(
                    (np.ones((g.shape[0],1)),
                     np.cos(pop[:,:(M-1)]*(np.pi/2)))
                )).cumprod(1)
            ))
        )
        popfun=np.multiply(
            popfun,
            (np.hstack(
                (np.ones((g.shape[0],1)),
                 np.sin(np.fliplr(pop[:,:(M-1)])*(np.pi/2)))
            ))
        )
    return popfun

def GO(pop,t1,t2,pc,pm):
    pop1 = copy.deepcopy(pop[0:int(pop.shape[0]/2),:])
    pop2 = copy.deepcopy(pop[(int(pop.shape[0]/2)):(int(pop.shape[0]/2)*2),:])
    N,D = pop1.shape[0],pop1.shape[1]
    #模拟二进制交叉
    beta=np.zeros((N,D))
    mu=np.random.random_sample([N,D])

    beta[mu<=0.5]=(2*mu[mu<=0.5])**(1/(t1+1))
    beta[mu>0.5]=(2-2*mu[mu>0.5])**(-1/(t1+1))
    beta=beta*((-1)**(np.random.randint(2, size=(N,D))))
    beta[np.random.random_sample([N,D])<0.5]=1
    beta[np.tile(np.random.random_sample([N,1])>pc,(1,D))]=1
    # test value
    beta=np.array([[1, -0.878870713, 1, -0.920930292, 1.005392461, -1.096583436, 1.083902817, 1, 1, -1.011287183, 0.992816033, 1],
     [-1.110349385, -0.873485845, 1, -0.962959982, -0.963243454, -1.063336868, 0.886432583, 0.951896223, 1, 1, 1, 1],
     [1, 1, -1.04691477, 1, 0.951538268, 1.007108557, -0.998326021, 1.002432312, 1, 1, -1.063349371, -1.041392457],
     [1, 1, 1, 0.900501563, 1, 1.033636659, 0.903264628, 0.971148024, 1, -0.946106543, 1, 1]])

    off = np.vstack(((pop1+pop2)/2+beta*(pop1-pop2)/2,(pop1+pop2)/2-beta*(pop1-pop2)/2))
    #多项式变异
    low=np.zeros((2*N,D))
    up=np.ones((2*N,D))
    site=np.random.random_sample([2*N,D]) < pm/D
    # test value
    site=np.array([[True,True,False,False,False,False,False,True,False,False,False,False],
[False,False,False,False,False,False,False,False,False,False,False,False],
[False,False,False,False,False,False,False,False,False,False,False,False],
[False,False,False,False,False,False,False,False,False,False,False,False],
[False,False,False,False,False,False,False,False,False,False,False,False],
[False,False,False,False,False,False,False,False,False,True,False,True],
[False,False,False,False,False,False,True,False,False,False,True,False],
[True,False,False,False,False,False,False,False,False,False,False,False]])
    mu = np.random.random_sample([2*N,D])
    # test value
    mu=np.array([[0.624839298,0.369995778,0.10970764,0.219856621,0.533680366,0.931944542,0.284911832,0.209206592,0.325568356,0.644241396,0.962398359,0.826637025],
[0.016991584,0.197278359,0.749724825,0.562153788,0.564333458,0.109185912,0.86834697,0.320817053,0.183212354,0.496115902,0.942216114,0.403104491],
[0.184094407,0.630484457,0.033729,0.566771511,0.302863316,0.976188427,0.964296963,0.542763046,0.816640317,0.456764777,0.153972744,0.564214618],
[0.038243251,0.603972472,0.609786042,0.895609427,0.626072884,0.992929495,0.472212724,0.000528076,0.780945301,0.821961692,0.843344089,0.058997741],
[0.709967537,0.462649651,0.012472086,0.048386347,0.960229536,0.639894086,0.660292141,0.796967948,0.866355124,0.428343185,0.236795931,0.228880588],
[0.526860769,0.292149839,0.259385915,0.597298454,0.790664812,0.287689763,0.519944039,0.260642325,0.81215641,0.091940903,0.896178825,0.002919548],
[0.575588584,0.451375377,0.416451222,0.41813774,0.890301028,0.342187696,0.71041813,0.528475246,0.546380853,0.245318474,0.621473305,0.62492191],
[0.93013724,0.992170714,0.091569941,0.372750828,0.383556775,0.0230959,0.073962309,0.715610665,0.079024727,0.610889619,0.194665362,0.825140308]])
    temp = site & (mu<=0.5)
    off[off<low]=low[off<low]
    off[off>up]=up[off>up]
    off[temp]=off[temp]+(up[temp]-low[temp])*((2*mu[temp]+(1-2*mu[temp])*((1-(off[temp]-low[temp])/(up[temp]-low[temp]))**(t2+1)))**(1/(t2+1))-1)
    temp = site & (mu>0.5)
    off[temp]=off[temp]+(up[temp]-low[temp])*(1-(2*(1-mu[temp])+2*(mu[temp]-0.5)*((1-(up[temp]-off[temp])/(up[temp]-low[temp]))**(t2+1)))**(1/(t2+1)))
    
    return off

def NDsort(mixpop,N,M):
    nsort = N#排序个数
    N,M = mixpop.shape[0],mixpop.shape[1]
    Loc1=np.lexsort(mixpop[:,::-1].T)#loc1为新矩阵元素在旧矩阵中的位置，从第一列依次进行排序
    mixpop2=mixpop[Loc1]
    Loc2=Loc1.argsort()#loc2为旧矩阵元素在新矩阵中的位置
    frontno=np.ones(N)*(np.inf)#初始化所有等级为np.inf
    #frontno[0]=1#第一个元素一定是非支配的
    maxfno=0#最高等级初始化为0
    while (np.sum(frontno < np.inf) < min(nsort,N)):#被赋予等级的个体数目不超过要排序的个体数目
        maxfno=maxfno+1
        for i in range(N):
            if (frontno[i] == np.inf):
                dominated = 0
                for j in range(i):
                    if (frontno[j] == maxfno):
                        m=0
                        flag=0
                        while (m<M and mixpop2[i,m]>=mixpop2[j,m]):
                            if(mixpop2[i,m]==mixpop2[j,m]):#相同的个体不构成支配关系
                                flag=flag+1
                            m=m+1 
                        if (m>=M and flag < M):
                            dominated = 1
                            break
                if dominated == 0:
                    frontno[i] = maxfno
    frontno=frontno[Loc2]
    return frontno,maxfno
#求两个向量矩阵的余弦值,x的列数等于y的列数
def pdist(x,y):
    x0=x.shape[0]
    y0=y.shape[0]
    xmy=np.dot(x,y.T)#x乘以y
    xm=np.array(np.sqrt(np.sum(x**2,1))).reshape(x0,1)
    ym=np.array(np.sqrt(np.sum(y**2,1))).reshape(1,y0)
    xmmym=np.dot(xm,ym)
    cos = xmy/xmmym
    return cos

def lastselection(popfun1,popfun2,K,Z,Zmin):
    #选择最后一个front的解
    popfun = copy.deepcopy(np.vstack((popfun1, popfun2)))-np.tile(Zmin,(popfun1.shape[0]+popfun2.shape[0],1))
    N,M = popfun.shape[0],popfun.shape[1]
    N1 = popfun1.shape[0]
    N2 = popfun2.shape[0]
    NZ = Z.shape[0]
    
    #正则化
    extreme = np.zeros(M)
    w = np.zeros((M,M))+1e-6+np.eye(M)
    for i in range(M):
        extreme[i] = np.argmin(np.max(popfun/(np.tile(w[i,:],(N,1))),1))
    
    #计算截距
    extreme = extreme.astype(int)#python中数据类型转换一定要用astype
    try:   #修改
        temp = np.linalg.inv(np.mat(popfun[extreme,:]))       #逆矩阵
        # temp = np.linalg.pinv(np.mat(popfun[extreme,:]))   #广义逆矩阵 原来是求广义逆矩阵的，所以导致多个目标的时候可以求到广义逆矩阵。
    except:
        print("矩阵不存在逆矩阵")
        temp=np.full((M,M),np.nan)
    hyprtplane = np.array(np.dot(temp,np.ones((M,1))))
    a = 1/hyprtplane
    if np.any(np.isnan(a)):   #修改 原来的形式是：sum的形式
        a = np.max(popfun,0)
    np.array(a).reshape(M,1)#一维数组转二维数组
    a = a.T - Zmin    #修改，原来是下面那一行的形式
    # a=a.T
    popfun = popfun/(np.tile(a,(N,1)))
    
    ##联系每一个解和对应向量
    #计算每一个解最近的参考线的距离
    cos = pdist(popfun,Z)
    distance = np.tile(np.array(np.sqrt(np.sum(popfun**2,1))).reshape(N,1),(1,NZ))*np.sqrt(1-cos**2)
    #联系每一个解和对应的向量
    d = np.min(distance.T,0)
    pi = np.argmin(distance.T,0)
    
    #计算z关联的个数
    rho = np.zeros(NZ)
    for i in range(NZ):
        rho[i] = np.sum(pi[:N1] == i)
    
    #选出剩余的K个
    choose = np.zeros(N2)
    choose = choose.astype(bool)
    zchoose = np.ones(NZ)
    zchoose = zchoose.astype(bool)
    while np.sum(choose) < K:
        #选择最不拥挤的参考点
        temp = np.ravel(np.array(np.where(zchoose == True)))
        jmin = np.ravel(np.array(np.where(rho[temp] == np.min(rho[temp]))))
        j = temp[jmin[np.random.randint(jmin.shape[0])]]
#        I = np.ravel(np.array(np.where(choose == False)))
#        I = np.ravel(np.array(np.where(pi[(I+N1)] == j)))
        # 修改，将下面的一行替换为两行
        kkk = np.where(pi[N1:] == j, 1, 0) * np.where(choose == 0, 1, 0)
        I = np.ravel(np.where(kkk == 1))
        # I = np.ravel(np.array(np.where(pi[N1:] == j)))
        I = I[choose[I] == False]
        if (I.shape[0] != 0):
            if (rho[j] == 0):
                s = np.argmin(d[N1+I])
            else:
                s = np.random.randint(I.shape[0])
            choose[I[s]] = True
            rho[j] = rho[j]+1
        else:
            zchoose[j] = False
    return choose

def envselect(mixpop,N,Z,Zmin,name,M,D):
    #非支配排序
    mixpopfun = cal(mixpop,name,M,D)
    frontno,maxfno = NDsort(mixpopfun,N,M)
    Next = frontno < maxfno
    #选择最后一个front的解
    Last = np.ravel(np.array(np.where(frontno == maxfno)))
    choose = lastselection(mixpopfun[Next,:],mixpopfun[Last,:],N-np.sum(Next),Z,Zmin)
    Next[Last[choose]] = True
    #生成下一代
    pop = copy.deepcopy(mixpop[Next,:])
    return pop

def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A,BT)
    # print(vecProd)
    SqA =  A**2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)
 
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0   
    ED = np.sqrt(SqED)
    return ED

def IGD(popfun,PF):
    distance = np.min(EuclideanDistances(PF,popfun),1)
    score = np.mean(distance)
    return score