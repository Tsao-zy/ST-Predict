# -*- coding: utf-8 -*-
"""
@author: Ziyang Cao
copyright@2020
"""
import random
import math
import time
import warnings
import pandas as pd
import numpy as np
from scipy import special
import datetime
from sympy import Symbol, diff
warnings.filterwarnings("ignore")
#from cvxopt import matrix, solvers
import json
import multiprocessing
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from scipy.linalg import pinv as pinverse
def mergeData(file_path, behavior="in"):
    '''
    Parameters
    ----------
    file_path :the location where your ".csv" stored.
    behavior : string, where is "in" or "out".The default is "in".

    Returns
    -------
    Dataframe, 81 location and 26 days flows per 10 mins.

    '''
    move = behavior + "_"
    year = 2019
    month = 1
    df = None
    
    for day in range(1, 26):
        ddate = datetime.datetime(year, month, day).strftime("%Y-%m-%d")
        ddate = move + ddate
        path =  file_path + ddate + ".csv"
        day_data = pd.read_csv(path)
        if df is None:
            df = day_data
        else:
            df = pd.concat([df, day_data], ignore_index=True)
    df = df[["start_clock", "stationID", "flows", "lat", "lng"]]
    df.reset_index(inplace=True)
    return df

def cut(file_path, behavior="in"):
    start_tik = time.mktime(time.strptime("2019-01-01 06:00:00","%Y-%m-%d %H:%M:%S"))
    left_start = int(start_tik // 600)
    right_start = left_start + 103
    
    string = []
    for i in range(25):
        for j in range(left_start, right_start):
            string.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(j*600)))
        left_start += 144
        right_start += 144
    df_time = pd.DataFrame(string, columns=["start_clock"])
    df_all = mergeData(file_path)
    df = pd.merge(df_time, df_all, on="start_clock", how="inner")
    log_flow = np.log(df["flows"])
    log_flow[np.isneginf(log_flow)] = 0
    df["flows"] = log_flow
    df.drop(["index"],axis=1, inplace=True)
    return df

def estimateMeanANOVA(st_data):
    '''
    Parameters
    ----------
    st_data : spatio-temporal data

    Returns
    -------
    global mean, mean of time effort, mean of space effort.

    '''
    mu = st_data["flows"].mean()

    temporal_level = st_data["start_clock"].unique()
    spatio_level = st_data["stationID"].unique()
    
    temporal_effort = dict()
    spatio_effort = dict()
    
    for ttime in temporal_level:
        part_data = st_data[st_data["start_clock"] == ttime]
        temporal_effort[ttime] = part_data["flows"].mean()

    for location in spatio_level:
        part_data = st_data[st_data["stationID"] == location]
        spatio_effort[location] = part_data["flows"].mean()
    return mu, temporal_effort, spatio_effort

        

    
def getDerivativeSymbol(h_nonzero=True):
    '''

    Parameters
    ----------
    h : boolean, h is True means h>0, False means h=0.
        The default is True.

    Returns
    -------
    parameters_derivative_dict: dictionary, derivative about parameters (a,b,beta,sigma)
    C_h_u : string, C(h,u) of string format.
    comx : string, K_v of string format.The default is True.

    '''
    a = Symbol("a")
    b = Symbol("b")
    beta = Symbol("beta")
    sigma = Symbol("sigma")
    mu = Symbol("mu")
    h = Symbol("h")
    nu = Symbol("nu")
    gamma = Symbol("gamma")
    parameters_list = [a, b, beta, sigma]
  
    para1 = a * a * mu * mu + 1
    para2 = a * a * mu * mu + beta
    if h_nonzero is True:
        C_h_u = sigma * 2 * beta * (b / 2 * (para1/para2)**(1/2) * h)**nu / (para1)**(nu) / para2 / gamma
        comx = b * (para1/para2)**(1/2) * h
    else:
        C_h_u = sigma * 2 * beta / (para1)**(nu) / para2
        comx = 1
    parameters_derivative_dict = {}
    
    for p in parameters_list:
        parameters_derivative_dict[str(p)] = (str(diff(C_h_u, p)), str(diff(comx, p)))

    return parameters_derivative_dict, str(C_h_u), str(comx) 


def getConvFunc(a, b, beta, sigma, mu, h, nu, gamma):
    '''

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.
    nu : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.

    Returns
    -------
    C_h_u : TYPE
        DESCRIPTION.

    '''
    para1 = a * a * mu * mu + 1
    para2 = a * a * mu * mu + beta
    if h == 0:
        C_h_u = sigma * 2 * beta * (b / 2 * (para1/para2)**(1/2) * h)**nu / (para1)**(nu) / para2 / gamma
    else:
        C_h_u = sigma * 2 * beta / (para1)**(nu) / para2
    return C_h_u

        
def gradient_g(s1, t1, s2, t2, a_hat, b_hat, beta_hat, sigma_hat):
    '''
    

    Parameters
    ----------
    s1 : Array, space information about datapoint 1.
    t1 : timestamp, time information about datapoint 1.
    s2 : Array, space information about datapoint 2.
    t2 : timestamp, time information about datapoint 1.
    a_hat : float, real value about parameter a.
    b_hat : float, real value about parameter b.
    beta_hat : float, real value about parameter beta.
    sigma_hat : float, real value about parameter sigma.

    Returns
    -------
    Array, derivative about parameter.

    '''
    h_zero = True
    if (s1 == s2).all():
        h_zero = False
    parameters_derivative_dict, C_h_u, comx = getDerivativeSymbol(h_zero)
    h = np.sqrt(np.sum((s1 - s2)**2))
    mu = abs((t1 - t2).seconds // 60)
    a = a_hat
    b = b_hat
    beta = beta_hat
    sigma = sigma_hat
    nu = 3
    gamma = special.gamma(nu)
    f1 = eval(C_h_u)
    x = eval(comx)
    
    gradient_a = eval(parameters_derivative_dict["a"][0]) * special.kv(nu, x) + eval(parameters_derivative_dict["a"][1]) * f1 * (1/2 * (special.kv(nu - 1, x) - special.kv(nu + 1, x)))
    gradient_b = eval(parameters_derivative_dict["b"][0]) * special.kv(nu, x) + eval(parameters_derivative_dict["b"][1])* f1 *(1/2 * (special.kv(nu - 1, x) - special.kv(nu + 1, x)))
    gradient_beta = eval(parameters_derivative_dict["beta"][0]) * special.kv(nu, x) + eval(parameters_derivative_dict["beta"][1])* f1 *(1/2 * (special.kv(nu - 1, x) - special.kv(nu + 1, x)))
    gradient_sigma = eval(parameters_derivative_dict["sigma"][0]) * special.kv(nu, x) - 2
    gradient_sigma_e = 2
    return np.array([-gradient_a, -gradient_b, -gradient_beta, -gradient_sigma, gradient_sigma_e])



def distance_k(s1, s2, t1, t2, flow1, flow2):
    '''
    Parameters
    ----------
    s1 : Array, space information about datapoint 1.
    t1 : timestamp, time information about datapoint 1.
    s2 : Array, space information about datapoint 2.
    t2 : timestamp, time information about datapoint 1.
    flow1 : integer or float, flow at time t1 and space s1.
    flow2 : integer or float, flow at time t2 and space s2.

    Returns
    -------
    integer or float, the flow difference between (s1, t1) and (s2, t2), where
    equals to X(s1,t1) - X(s2, t2)

    '''
    flow1 = flow1
    flow2 = flow2
    return flow1 - flow2


def caculateCEF(s1, t1, s2, t2, a, b, beta, sigma, sigma_e, flow1, flow2):
    '''

    Parameters
    ----------
    s1 : Array, space information about datapoint 1.
    t1 : timestamp, time information about datapoint 1.
    s2 : Array, space information about datapoint 2.
    t2 : timestamp, time information about datapoint 1.
    a : float, real value about parameter a.
    b : float, real value about parameter b.
    beta : float, real value about parameter beta.
    sigma : float, real value about parameter sigma.
    sigma_e : float, real value about parameter sigma_epsilon.
    flow1 : integer or float, flow at time t1 and space s1.
    flow2 : integer or float, flow at time t2 and space s2.

    Returns
    -------
    CEF : float, Composite Estimating Function about (s1,t1) and (s2, t2).
    '''
    h = np.sqrt(np.sum((s1 - s2)**2))
    mu = abs((t1 - t2).seconds // 60)
    para1 = a * a * mu * mu + 1
    para2 = a * a * mu * mu + beta
    if h == 0:
        C_h_u = (sigma * 2 * beta) / math.sqrt(para1) / para2
    else:
        x = b * math.sqrt(para1 / para2) * h
        C_h_u = (sigma * 2 * beta) * math.sqrt(x/2) * special.kv(0.5, x) / math.sqrt(para1) / para2 / special.gamma(0.5)
    variance_dk = 4 * sigma + 4 * sigma_e - 2 * C_h_u
    Gamma_k = variance_dk / 2
    CEF = gradient_g(s1, t1, s2, t2, a, b, beta, sigma) / 2 / Gamma_k * (1 - distance_k(s1, s2, t1, t2,flow1, flow2)**2 / 2 / Gamma_k)
    return CEF


def caculatePhin(dataset, a, b, beta, sigma, sigma_e):
    '''

    Parameters
    ----------
    dataset : array, subset about spatio-temporal datasets.
    a : float, real value about parameter a.
    b : float, real value about parameter b.
    beta : float, real value about parameter beta.
    sigma : float, real value about parameter sigma.
    sigma_e : float, real value about parameter sigma_epsilon.

    Returns
    -------
    phi_n :  array, Composite Estimating Function in dataset.
    count : integer, length of dataset.

    '''
    phi_n = np.zeros(5)
    count = len(dataset)
     
    for i in range(0, count, 2):
        s1 = np.array([dataset[i, 3], dataset[i, 4]])
        s2 = np.array([dataset[i + 1, 3], dataset[i + 1, 4]])
        t1 = datetime.strptime(dataset[i, 0], "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(dataset[i + 1, 0], "%Y-%m-%d %H:%M:%S")
        flow1 = dataset[i, 5]
        flow2 = dataset[i + 1, 5]
        phi_n += caculateCEF(s1, t1, s2, t2, a, b, beta, sigma, sigma_e, flow1, flow2)
    
    return phi_n, count
    



def subregionIndex(dataset):
    '''
    Parameters
    ----------
    dataset : Array, spatio-temporal data

    Returns
    -------
    space_index : array, spatio datapoint index
    time_index : array, temporal datapoint index
    '''

    space_index = list()
    for i in set(dataset[:, 1]):
        if i == 54:
            continue
        index_array = np.where(dataset[:, 1] == i)[0]
        space_index.append(index_array)
    time_index = list()
    for t in set(dataset[:, 0]):
        index_array = np.where(dataset[:, 0] == t)[0]
        time_index.append(index_array)
    return space_index, time_index


def sampling(dataset, num, ratio, index_array, inter=False):
    '''
    
    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    num : TYPE
        DESCRIPTION.
    ratio : TYPE
        DESCRIPTION.
    index_array : TYPE
        DESCRIPTION.
    inter : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    sample : TYPE
        DESCRIPTION.

    '''
    length, width = len(index_array), min(map(len, index_array))
    sample = None
    passed = False
    while not passed:
        if sample is not None and len(sample) >= num * ratio:
            passed = True
            break
        idx_d1 = random.randint(0, length - 1)
        if inter is True:
            idx_d2 = random.randint(0, length - 1)
            while idx_d2 == idx_d1:
                idx_d2 = random.randint(0, length - 1)
        else:
            idx_d2 = idx_d1
        
        idx_d3, idx_d4 = random.randint(0, width - 1), random.randint(0, width - 1)
        if idx_d3 == idx_d4:
            continue
        if sample is None:
            sample = np.vstack((dataset[index_array[idx_d1][idx_d3]], dataset[index_array[idx_d2][idx_d4]]))
        else:
            sample = np.vstack((sample, dataset[index_array[idx_d1][idx_d3]], dataset[index_array[idx_d2][idx_d4]]))
    return sample

def subregion(dataset, ratio={"T": 0.25, "S": 0.25, "C": 0.5}, num=5000):
    '''

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    ratio : TYPE, optional
        DESCRIPTION. The default is {"T": 0.25, "S": 0.25, "C": 0.5}.
    num : TYPE, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    space_sample : TYPE
        DESCRIPTION.
    time_sample : TYPE
        DESCRIPTION.
    interact_sample : TYPE
        DESCRIPTION.

    '''
    space_index_array, time_index_array = subregionIndex(dataset)
 
    time_sample = sampling(dataset, num, ratio["T"], time_index_array)
    space_sample = sampling(dataset, num, ratio["S"], space_index_array)
    interact_sample = sampling(dataset, num, ratio["C"], time_index_array, inter=True)
    
    return space_sample, time_sample, interact_sample

def evaluateGammaCov(dataset, a, b, beta, sigma, sigma_e, size=5):
    '''
    

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    sigma_e : TYPE
        DESCRIPTION.
    size : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    res = np.zeros((size, 15))
    cnt = np.zeros((size, 3))
    piece_len = len(dataset) // size
    for i in range(size):
        timeset_sample, spaceset_sample, interactset_sample = subregion(dataset[(piece_len*i):(piece_len*(i+1)), :], 
                                                                        ratio={"T": 0.35, "S": 0.15, "C": 0.5},num=int(piece_len*0.05))
        phi_time, T_count = caculatePhin(timeset_sample, a, b, beta, sigma, sigma_e)
        phi_space, S_count = caculatePhin(spaceset_sample, a, b, beta, sigma, sigma_e)
        phi_interact, C_count = caculatePhin(interactset_sample, a, b, beta, sigma, sigma_e)
        
        Gamma_n = np.hstack((phi_time / T_count, phi_space / S_count, phi_interact / C_count))
        res[i] = Gamma_n
        cnt[i, 0] += T_count
        cnt[i, 1] += S_count
        cnt[i, 2] += C_count
    return np.sum(cnt) * (size - 1) * np.cov(res.T), np.sum(cnt[:, 0]), np.sum(cnt[:, 1]), np.sum(cnt[:, 2])

def evaluateGammaCovBoots(dataset, a, b, beta, sigma, sigma_e, kn=10):
    '''

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    sigma_e : TYPE
        DESCRIPTION.
    kn : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    ans = np.zeros((15, 15))
    Tcount = 0
    Scount = 0
    Ccount = 0
    for j in range(kn):
        size = np.random.randint(4, 9)
        tmp, tsize, ssize, csize = evaluateGammaCov(dataset, a, b, beta, sigma, sigma_e, size=size)
        ans += tmp
        Tcount += tsize
        Scount += ssize
        Ccount += csize
    cov_hat = ans/ kn
    diag_cnt = np.diag([Scount, Tcount, Ccount])
    sqrt_N = np.kron(diag_cnt, np.eye(5))
    return sqrt_N.dot(cov_hat).dot(sqrt_N)
    # avg_gamma = np.mean(res, axis=0)
    # cov_hat = (((res - avg_gamma).T).dot(np.diag(cnt)).dot((res - avg_gamma) ))/ size
    # #cov_hat = np.average(res, weights=cnt, axis=0) * np.average(cnt)
    # Tcount = 5000 * 0.25
    # Scount = 5000 * 0.25
    # Ccount = 5000 * 0.5
    # diag_cnt = np.diag([Scount, Tcount, Ccount])
    # sqrt_N = np.kron(diag_cnt, np.eye(5))
    # return sqrt_N.dot(cov_hat).dot(sqrt_N)
    

    
def caculateQn(timeset, spaceset, interactset, a, b, beta, sigma, sigma_e, W_inv=np.eye(15)):
    '''

    Parameters
    ----------
    timeset : array, subset of time.
    spaceset : array, subset of space.
    interactset : array, subset of interaction.
    a : float, real value about parameter a.
    b : float, real value about parameter b.
    beta : float, real value about parameter beta.
    sigma : float, real value about parameter sigma.
    sigma_e : float, real value about parameter sigma_epsilon.
    W_inv : matrix, optional, inverse matrix of weight. The default is np.eye(15).

    Returns
    -------
    Q_n : float, Q_n on dataset entirely.

    '''

    phi_time, T_count = caculatePhin(timeset, a, b, beta, sigma, sigma_e)
    phi_space, S_count = caculatePhin(spaceset, a, b, beta, sigma, sigma_e)
    phi_interact, C_count = caculatePhin(interactset, a, b, beta, sigma, sigma_e)
    
    Gamma_n = np.hstack((phi_time / T_count, phi_space / S_count, phi_interact / C_count))
    Q_n = Gamma_n.dot(W_inv).dot(np.transpose(Gamma_n))
    return Q_n


# def optim(timeset, spaceset, interactset,
#           init_a=1, init_b=1, init_beta=1, init_sigma=1, 
#           init_sigma_e=1, setw=np.eye(15),
#           a_range=np.arange(0, 5, 0.1), 
#           b_range=np.arange(0, 4, 0.01), 
#           beta_range=np.arange(0.5, 10.5, 0.5),
#           sigma_range=np.arange(1, 21), 
#           sigma_e_range=np.arange(1, 21)):
#     best_estimator = np.array([init_a, init_b, init_beta, init_sigma, init_sigma_e])
#     qn = caculateQn(timeset, spaceset, interactset, init_a, 
#                     init_b, init_beta, init_sigma, init_sigma_e, setw)
#     for a in a_range:
#         for b in b_range:
#             for beta in beta_range:
#                 for sigma in sigma_range:
#                     for sigma_e in sigma_e_range:
#                         new_qn = caculateQn(timeset, spaceset, interactset, 
#                                             a, b, beta, sigma, sigma_e, setw)
#                         if qn > new_qn:
#                             qn = new_qn
#                             best_estimator = np.array([a, b, beta, sigma, sigma_e])
#     return best_estimator


def optimal_pso(all_in_st_arr, W_inv=np.eye(15)):
    '''

    Parameters
    ----------
    all_in_st_arr : TYPE
        DESCRIPTION.
    W_inv : TYPE, optional
        DESCRIPTION. The default is np.eye(15).

    Returns
    -------
    dict
        DESCRIPTION.
    fitLists : TYPE
        DESCRIPTION.

    '''
    g_bests = []
    fit_g = []
    fitLists = []
    spaceset, timeset, interactset = subregion(all_in_st_arr, num=5000)
    my_pso = PSO(pN=40, dim=5, max_iter=10, timeset=timeset, spaceset=spaceset, interactset=interactset,W_inv=W_inv)
    my_pso.init_population()
    fitList = my_pso.iterator()
    fit_g.append(my_pso.fit)
    g_bests.append(my_pso.g_best)
    fitLists.append(fitList)
    return {"g_best": g_bests, "g_fit": fit_g}, fitLists




def predict(a, b, beta, sigma, tik, loc, gis, kdt, cdist, arr, timeSpan=103):
    '''
    

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    tik : TYPE
        DESCRIPTION.
    loc : TYPE
        DESCRIPTION.
    gis : TYPE
        DESCRIPTION.
    kdt : TYPE
        DESCRIPTION.
    cdist : TYPE
        DESCRIPTION.
    arr : TYPE
        DESCRIPTION.
    timeSpan : TYPE, optional
        DESCRIPTION. The default is 103.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    k = len(kdt.query_radius(gis[loc].reshape(1, -1), r=0.03)[0])
    distance_k, station_k = kdt.query(gis[loc].reshape(1, -1), k=k)

    tiks = []
    for i in range(144):
        moment = datetime.datetime.strptime(tik, "%Y-%m-%d %H:%M:%S") - datetime.timedelta(minutes=144* 10 - 10 * i)
        if (moment.hour >= 23 and moment.minute > 0) or moment.hour < 6:
            continue
        else:
            tiks.append(moment.strftime("%Y-%m-%d %H:%M:%S"))
    tiks_stamp = list(map(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")), tiks))
    now_stamp = time.mktime(time.strptime(tik, "%Y-%m-%d %H:%M:%S"))
    c_star = np.zeros((1, timeSpan*k))
    C_mid = 2 * sigma * np.eye(timeSpan*k, timeSpan*k)
    flow_value = np.zeros((timeSpan*k, 1))
    nu = 3
    gamma = special.gamma(nu)
    for i in range(timeSpan*k):
        loc1 = i // timeSpan
        tik1 = i % timeSpan
        for j in range(timeSpan*k):
            loc2 = j // timeSpan
            tik2 = j % timeSpan
            h = cdist[station_k[0, loc1], station_k[0, loc2]]
            mu = abs(tiks_stamp[tik1] - tiks_stamp[tik2]) // 60
            C_mid[i, j] = getConvFunc(a, b, beta, sigma, mu, h, nu, gamma)
    for i in range(timeSpan*k):
        loc_diff = i // timeSpan
        tik_diff = abs(tiks_stamp[i%k] - now_stamp)
        h = distance_k[0, loc_diff]
        mu = tik_diff // 60
        c_star[0, i] = getConvFunc(a, b, beta, sigma, mu, h, nu, gamma)
    
    for i in range(k):
        flow_value[(timeSpan*i):(timeSpan*(i+1)), 0] = arr[np.isin(arr[:,0], tiks) & np.isin(arr[:,1], loc), -1].astype("float")
  
    return float(c_star.dot(pinverse(C_mid)).dot(flow_value))





def predictAll(a, b, beta, sigma, all_in_st_arr, start_tik="2019-01-26 06:00:00"):
    '''
    

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    beta : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    all_in_st_arr : TYPE
        DESCRIPTION.
    start_tik : TYPE, optional
        DESCRIPTION. The default is "2019-01-26 06:00:00".

    Returns
    -------
    local_time_arr : TYPE
        DESCRIPTION.

    '''
    tiks_range = [(datetime.datetime.strptime(start_tik, "%Y-%m-%d %H:%M:%S") - 
                   datetime.timedelta(minutes=144* 10 - 10 * i)).strftime("%Y-%m-%d %H:%M:%S")
                  for i in range(103)]
    local_time_arr = all_in_st_arr[np.isin(all_in_st_arr[:, 0], tiks_range)]
    tiks_list = [(datetime.datetime.strptime(start_tik, "%Y-%m-%d %H:%M:%S") + 
                 datetime.timedelta(minutes=10 * i)).strftime("%Y-%m-%d %H:%M:%S")
                for i in range(103)]
    gis = local_time_arr[:80, 3:5].astype("float")
    gis = np.vstack((gis[:54], np.array([float("Inf"),float("Inf")]), gis[54:]))
    kdt = KDTree(gis, leaf_size=10)
    cdists = cdist(gis, gis)
    
    for tik in tiks_list:
        for loc in range(81):
            if loc == 54:
                continue
            log_pred = predict(a, b, beta, sigma, tik, loc, gis, kdt, cdists, local_time_arr, timeSpan=103)
            new_info = np.array([tik, loc, 0, gis[loc, 0], gis[loc, 1], log_pred], dtype="object")
            local_time_arr = np.vstack((local_time_arr, new_info))
    return local_time_arr





# ----------------------PSO参数设置---------------------------------
class PSO:
    def __init__(self, timeset, spaceset, interactset, pN, dim, max_iter,
                 W_inv=np.eye(15), w=0.8, c1=1.5, c2=1.5, fit=float("Inf")):  # 初始化类  设置粒子数量  位置信息维度  最大迭代次数
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置（还要确定取值范围）
        self.a_range = (0, 10)
        self.b_range = (0, 5)
        self.beta_range = (0.01, 10)
        self.sigma_range = (0.01, 5)
        self.sigma_e_range = (0.01, 5)
        self.V = np.zeros((self.pN, self.dim))  # 所有粒子的速度（还要确定取值范围）
        self.a_v_range = (0.001, 0.5)
        self.b_v_range = (0.0001, 0.1)
        self.beta_v_range = (0.01, 1)
        self.sigma_v_range = (0.001, 0.01)
        self.sigma_e_v_range = (0.01, 0.5)
        self.p_best = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置
        self.g_best = np.zeros((1, self.dim))  # 全局最佳位置
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = fit # 全局最佳适应值
        self.timeset = timeset
        self.spaceset = spaceset
        self.interactset = interactset
        self.W_inv = W_inv
    # ---------------------目标函数Sphere函数-----------------------------

    # ---------------------初始化种群----------------------------------
    def init_population(self):

        for i in range(self.pN):  # 遍历所有粒子

            #for j in range(self.dim):  # 每一个粒子的纬度
            self.X[i][0] = random.uniform(self.a_range[0], self.a_range[1])
            self.X[i][1] = random.uniform(self.b_range[0], self.b_range[1])
            self.X[i][2] = random.uniform(self.beta_range[0], self.beta_range[1])
            self.X[i][3] = random.uniform(self.sigma_range[0], self.sigma_range[1])
            self.X[i][4] = random.uniform(self.sigma_e_range[0], self.sigma_e_range[1])# 给每一个粒子的位置赋一个初始随机值（在一定范围内）

            self.V[i][0] = random.uniform(self.a_v_range[0], self.a_v_range[1])
            self.V[i][1] = random.uniform(self.b_v_range[0], self.b_v_range[1])
            self.V[i][2] = random.uniform(self.beta_v_range[0], self.beta_v_range[1])
            self.V[i][3] = random.uniform(self.sigma_v_range[0], self.sigma_v_range[1])
            self.V[i][4] = random.uniform(self.sigma_e_v_range[0], self.sigma_e_v_range[1])
            #self.V[i][j] = random.uniform(-0.1, 0.1)  # 给每一个粒子的速度给一个初始随机值（在一定范围内）

            self.p_best[i] = self.X[i]  # 把当前粒子位置作为这个粒子的最优位置

            #tmp = self.function(self.X[i])  # 计算这个粒子的适应度值
            tmp = abs(caculateQn(self.timeset, self.spaceset, self.interactset,
                                 self.X[i][0], self.X[i][1], self.X[i][2],
                                 self.X[i][3], self.X[i][4], W_inv=self.W_inv))
            self.p_fit[i] = tmp  # 当前粒子的适应度值作为个体最优值

            if tmp < self.fit:  # 与当前全局最优值做比较并选取更佳的全局最优值

                self.fit = tmp
                self.g_best = self.X[i]

     # ---------------------更新粒子位置----------------------------------

    def iterator(self):

        fitness = []

        for t in range(self.max_iter):

            for i in range(self.pN):

                # 更新速度
                self.V[i] = self.w * self.V[i] + self.c1 * random.uniform(0,1) * (self.p_best[i] - self.X[i]) + \
                            (self.c2 * random.uniform(0,1) * (self.g_best - self.X[i]))

                self.V[i][0] = min(self.a_v_range[1], self.V[i][0])
                self.V[i][0] = max(self.a_v_range[0], self.V[i][0])
                self.V[i][1] = min(self.b_v_range[1], self.V[i][1])
                self.V[i][1] = max(self.b_v_range[0], self.V[i][1])
                self.V[i][2] = min(self.beta_v_range[1], self.V[i][2])
                self.V[i][2] = max(self.beta_v_range[0], self.V[i][2])
                self.V[i][3] = min(self.sigma_v_range[1], self.V[i][3])
                self.V[i][3] = max(self.sigma_v_range[0], self.V[i][3])
                self.V[i][4] = min(self.sigma_e_v_range[1], self.V[i][4])
                self.V[i][4] = max(self.sigma_e_v_range[0], self.V[i][4])

                # 更新位置
                self.X[i] = self.X[i] + self.V[i]

                self.X[i][0] = min(self.a_range[1], self.X[i][0])
                self.X[i][0] = max(self.a_range[0], self.X[i][0])
                self.X[i][1] = min(self.b_range[1], self.X[i][1])
                self.X[i][1] = max(self.b_range[0], self.X[i][1])
                self.X[i][2] = min(self.beta_range[1], self.X[i][2])
                self.X[i][2] = max(self.beta_range[0], self.X[i][2])
                self.X[i][3] = min(self.sigma_range[1], self.X[i][3])
                self.X[i][3] = max(self.sigma_range[0], self.X[i][3])
                self.X[i][4] = min(self.sigma_e_range[1], self.X[i][4])
                self.X[i][4] = max(self.sigma_e_range[0], self.X[i][4])

            for i in range(self.pN):  # 更新gbest\pbest

                #temp = self.function(self.X[i])
                temp = abs(caculateQn(self.timeset, self.spaceset, self.interactset,
                                      self.X[i][0], self.X[i][1], self.X[i][2],
                                      self.X[i][3], self.X[i][4], W_inv=self.W_inv))

                if temp < self.p_fit[i]:  # 更新个体最优
                    self.p_best[i] = self.X[i]
                    self.p_fit[i] = temp

                if temp < self.fit:  # 更新全局最优
                    self.g_best = self.X[i]
                    self.fit = temp

            fitness.append(self.fit)
            print(t, self.fit)  # 输出最优值

        return fitness



if __name__ == "__main__":
    
    # file_path = "G:/2-Graduate Document/thesis/dataset/datasets_Oper/"
    # #file_path = 
    # all_in_st_data = cut(file_path)
    # mu, temporal_effort, spatio_effort = estimateMeanANOVA(all_in_st_data)
    # all_in_st_data["flows_no_mean"] = 0
    # all_in_st_arr = np.array(all_in_st_data)
    
    # for i in range(len(all_in_st_arr)):
    #     time_key = all_in_st_arr[i, 0]
    #     time_mu = temporal_effort[time_key]
    #     location_key = all_in_st_arr[i, 1]
    #     location_mu = spatio_effort[location_key]
    #     flows = all_in_st_arr[i, 2]
    #     all_in_st_arr[i, 5] = flows - (time_mu + location_mu - mu)
    # #filepath = "/Users/ls_stat/CZY/"
    # filepath = "/cluster/home/ls_stat/CZY/all_in_st_arr.npy"
    # filepath = "D:/czy/testing.npy"
    # all_in_st_arr = np.load(filepath, allow_pickle=True)
    # #all_in_st_arr = np.load(filepath)
    # space_sample, time_sample, interact_sample = subregion(all_in_st_arr)
    #bst_est = optim(timeset=space_sample,
    #                spaceset=space_sample,
    #                interactset=interact_sample)
    #bst_est_dict = dict(zip(["a", "b", "beta", "sigma", "sigma_e"], bst_est))
    #est_js = json.dumps(bst_est_dict, indent=4)  # indent参数是换行和缩进
   
    #fo = open('/Users/ls_stat/CZY/best_estimator_for_weight.json', 'w')
    #fo.write(est_js)
    #fo.close()
    
    
    # est_weight_info = optimal_pso(timeset=time_sample, spaceset=space_sample, interactset=interact_sample)
    # np.save("/cluster/home/ls_stat/CZY/est_info.npy", est_weight_info)
    
    
    
    # caculateQn(timeset=time_sample, spaceset=space_sample, 
    #            interactset=interact_sample, a=1, b=2, 
    #            beta=3, sigma=9, sigma_e=1)
    # t1 = datetime.strptime(all_in_st_arr[1, 0], "%Y-%m-%d %H:%M:%S")
    # t2 = datetime.strptime(all_in_st_arr[100, 0], "%Y-%m-%d %H:%M:%S")
    # gradient_g(np.array([10,12]), t1, np.array([9,21]), t2, 0.3, 2, 3, 4)
    # caculateCEF(np.array([10,12]), t1, np.array([9,21]), t2, 0.3, 2, 3, 4, 10, 23, 3324)
 
    
 
    # w = evaluateGammaCovBoots(dataset=all_in_st_arr, a=4.78858005, 
    #                      b=4.31989921, beta=3.97813599, 
    #                      sigma=4.81582039, sigma_e=5) 
    # np.save("D:/czy/weight.npy", w)
    
    
    filepath = "D:/czy/testing.npy"
    all_in_st_arr = np.load(filepath, allow_pickle=True)
    #w = np.load("D:/czy/weight.npy")
    #W_inv = np.linalg.pinv(w)
    
    
    
    start = time.time()

    #result, fitness = optimal_pso(all_in_st_arr, W_inv)
    
    #pools = multiprocessing.Pool(processes=6)
    #results = [pools.apply_async(optimal_pso, args=(all_in_st_arr, W_inv)) for i in range(6)]
    #result = [p.get() for p in results]

    #pools.close()
    #pools.join()
 
    #est_weight_info = optimal_pso(timeset=time_sample, spaceset=space_sample, interactset=interact_sample, W_inv=W_inv)
    #np.save("D:/czy/est_info.npy", result)
    #np.save("D:/czy/est_fitness.npy", fitness)


    a = 1.415470645138922290e+00
    b = 1.848971147251750935e+00
    beta = 7.013283637640848056e+00
    sigma = 4.458254386223678978e+00
    sigma_e = 1.480545408869228918e+00
    pred_arr = predictAll(a, b, beta, sigma, all_in_st_arr)
    np.save("D:/czy/pred_arr.npy", pred_arr)
    
    end = time.time()
    
    print(end - start)
    
    
        