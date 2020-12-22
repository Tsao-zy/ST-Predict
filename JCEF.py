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
import multiprocessing
from sympy import Symbol, diff
warnings.filterwarnings("ignore")
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from scipy.linalg import pinv as pinverse

random.seed(100)





def strDate2Timestamp(date_time, formats="%Y-%m-%d %H:%M:%S"):
    '''

    Parameters
    ----------
    date_time : TYPE
        DESCRIPTION.
    formats : TYPE, optional
        DESCRIPTION. The default is "%Y-%m-%d %H:%M:%S".

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    timestamp : TYPE
        DESCRIPTION.

    '''
    if type(date_time) is not str:
        raise Exception("date time format is not a string.")
    struct_time = time.strptime(date_time, formats)
    timestamp = time.mktime(struct_time)
    return int(timestamp)




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
    df["start_clock"] = df["start_clock"].map(strDate2Timestamp)
    df.reset_index(inplace=True)
    return df




def cutZeroTime(file_path, behavior="in", start_time="2019-01-01 06:00:00"):
    '''

    Parameters
    ----------
    file_path : TYPE
        DESCRIPTION.
    behavior : TYPE, optional
        DESCRIPTION. The default is "in".
    start_time : TYPE, optional
        DESCRIPTION. The default is "2019-01-01 06:00:00".

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    start_tik = strDate2Timestamp(start_time)
    start_10_minutes = int(start_tik // 600)
    end_10_minutes = start_10_minutes + 103
    
    timestamp_list = []
    for day in range(25):
        for Min_per10 in range(start_10_minutes, end_10_minutes):
            timestamp_list.append(Min_per10*600)
        start_10_minutes += 144
        end_10_minutes += 144
    df_timestamp = pd.DataFrame(timestamp_list, columns=["start_clock"])
    df_all = mergeData(file_path, behavior=behavior)
    df = pd.merge(df_timestamp, df_all, on="start_clock", how="inner")
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
    st_data["hour_minute"] = st_data["start_clock"].map(lambda x:datetime.datetime.fromtimestamp(x).strftime("%H:%M:%S"))

    temporal_level = st_data["hour_minute"].unique()
    spatio_level = st_data["stationID"].unique()
    
    temporal_effort = dict()
    spatio_effort = dict()
    
    for ttime in temporal_level:
        part_data = st_data[st_data["hour_minute"] == ttime]
        temporal_effort[ttime] = part_data["flows"].mean()

    for location in spatio_level:
        part_data = st_data[st_data["stationID"] == location]
        spatio_effort[location] = part_data["flows"].mean()
    return mu, temporal_effort, spatio_effort




def ANOVAProcessing(st_data):
    '''

    Parameters
    ----------
    st_data : TYPE
        DESCRIPTION.

    Returns
    -------
    st_arr : TYPE
        DESCRIPTION.

    '''
    mu, temporal_effort, spatio_effort = estimateMeanANOVA(st_data)
    st_arr = np.array(st_data.values)
    
    for i in range(len(st_arr)):
        times = st_arr[i, 0]
        time_key = datetime.datetime.fromtimestamp(times).strftime("%H:%M:%S")
        time_mu = temporal_effort[time_key]
        location_key = st_arr[i, 1]
        location_mu = spatio_effort[location_key]
        flows = st_arr[i, 2]
        st_arr[i, 5] = flows - (time_mu + location_mu - mu)
    return st_arr

    


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
    mu = abs((t1 - t2) // 60)
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
    mu = abs((t1 - t2) // 60)
    para1 = a * a * mu * mu + 1
    para2 = a * a * mu * mu + beta
    if h == 0:
        C_h_u = (sigma * 2 * beta) / math.sqrt(para1) / para2
    else:
        x = b * math.sqrt(para1 / para2) * h
        C_h_u = (sigma * 2 * beta) * math.sqrt(x/2) * special.kv(0.5, x) / math.sqrt(para1) / para2 / special.gamma(0.5)
    variance_dk = 4 * sigma + 4 * sigma_e - 2 * C_h_u
    Gamma_k = variance_dk / 2
    CEF = gradient_g(s1, t1, s2, t2, a, b, beta, sigma) / 2 / Gamma_k * (1 - (flow1 - flow2)**2 / 2 / Gamma_k)
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
        t1 = dataset[i, 0]
        t2 = dataset[i + 1, 0]
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



def optimalPso(st_arr, W_inv=np.eye(15)):
    '''

    Parameters
    ----------
    st_arr : TYPE
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
    spaceset, timeset, interactset = subregion(st_arr, num=5000)
    my_pso = PSO(pN=40, dim=5, max_iter=10, timeset=timeset, spaceset=spaceset, interactset=interactset,W_inv=W_inv)
    my_pso.init_population()
    fitList = my_pso.iterator()
    fit_g.append(my_pso.fit)
    g_bests.append(my_pso.g_best)
    fitLists.append(fitList)
    return {"g_best": g_bests, "g_fit": fit_g}, fitLists




def getTiksRange(tik, span=103):
    
    tiks_stamp = []
    for i in range(144, 0, -1):
        moment = tik - 600 * i
        dtform = datetime.datetime.fromtimestamp(moment)
        if (dtform.hour >= 23 and dtform.minute > 0) or dtform.hour < 6:
            continue
        else:
            tiks_stamp.append(moment)
    return tiks_stamp

    


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
    tiks_stamp = getTiksRange(tik)
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
        tik_diff = abs(tiks_stamp[i%k] - tik)
        h = distance_k[0, loc_diff]
        mu = tik_diff // 60
        c_star[0, i] = getConvFunc(a, b, beta, sigma, mu, h, nu, gamma)
    
    for i in range(k):
        flow_value[(timeSpan*i):(timeSpan*(i+1)), 0] = arr[np.isin(arr[:,0], tiks_stamp) & np.isin(arr[:,1], loc), -1].astype("float")
  
    return float(c_star.dot(pinverse(C_mid)).dot(flow_value))





def predictAll(a, b, beta, sigma, st_arr, start_tik=1548453600):
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
    st_arr : TYPE
        DESCRIPTION.
    start_tik : TYPE, optional
        DESCRIPTION. The default is "2019-01-26 06:00:00".

    Returns
    -------
    local_time_arr : TYPE
        DESCRIPTION.

    '''
    tiks_range = getTiksRange(start_tik)
    
    local_time_arr = st_arr[np.isin(st_arr[:, 0], tiks_range)]
    

    tiks_list = [start_tik + 600 * i for i in range(103)]
    
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




class PSO:
    def __init__(self, timeset, spaceset, interactset, pN, dim, max_iter,
                 W_inv=np.eye(15), w=0.8, c1=1.5, c2=1.5, fit=float("Inf")):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pN = pN
        self.dim = dim
        self.max_iter = max_iter
        self.X = np.zeros((self.pN, self.dim))
        self.a_range = (0, 10)
        self.b_range = (0, 5)
        self.beta_range = (0.01, 10)
        self.sigma_range = (0.01, 5)
        self.sigma_e_range = (0.01, 5)
        self.V = np.zeros((self.pN, self.dim))
        self.a_v_range = (0.001, 0.5)
        self.b_v_range = (0.0001, 0.1)
        self.beta_v_range = (0.01, 1)
        self.sigma_v_range = (0.001, 0.01)
        self.sigma_e_v_range = (0.01, 0.5)
        self.p_best = np.zeros((self.pN, self.dim))
        self.g_best = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)
        self.fit = fit
        self.timeset = timeset
        self.spaceset = spaceset
        self.interactset = interactset
        self.W_inv = W_inv


    def init_population(self):

        for i in range(self.pN):

            self.X[i][0] = random.uniform(self.a_range[0], self.a_range[1])
            self.X[i][1] = random.uniform(self.b_range[0], self.b_range[1])
            self.X[i][2] = random.uniform(self.beta_range[0], self.beta_range[1])
            self.X[i][3] = random.uniform(self.sigma_range[0], self.sigma_range[1])
            self.X[i][4] = random.uniform(self.sigma_e_range[0], self.sigma_e_range[1])

            self.V[i][0] = random.uniform(self.a_v_range[0], self.a_v_range[1])
            self.V[i][1] = random.uniform(self.b_v_range[0], self.b_v_range[1])
            self.V[i][2] = random.uniform(self.beta_v_range[0], self.beta_v_range[1])
            self.V[i][3] = random.uniform(self.sigma_v_range[0], self.sigma_v_range[1])
            self.V[i][4] = random.uniform(self.sigma_e_v_range[0], self.sigma_e_v_range[1])


            self.p_best[i] = self.X[i]

            tmp = abs(caculateQn(self.timeset, self.spaceset, self.interactset,
                                 self.X[i][0], self.X[i][1], self.X[i][2],
                                 self.X[i][3], self.X[i][4], W_inv=self.W_inv))
            self.p_fit[i] = tmp

            if tmp < self.fit:

                self.fit = tmp
                self.g_best = self.X[i]



    def iterator(self):

        fitness = []

        for t in range(self.max_iter):

            for i in range(self.pN):

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

            for i in range(self.pN):

                temp = abs(caculateQn(self.timeset, self.spaceset, self.interactset,
                                      self.X[i][0], self.X[i][1], self.X[i][2],
                                      self.X[i][3], self.X[i][4], W_inv=self.W_inv))

                if temp < self.p_fit[i]:
                    self.p_best[i] = self.X[i]
                    self.p_fit[i] = temp

                if temp < self.fit:
                    self.g_best = self.X[i]
                    self.fit = temp

            fitness.append(self.fit)
            print(t, self.fit)

        return fitness



if __name__ == "__main__":
    
    
    """
    estimate parameters for weight matrix
    """
    start = time.time()
    # file_path = "D:/datasets/"
    # st_data = cutZeroTime(file_path)
    # st_arr = ANOVAProcessing(st_data)
    # space_sample, time_sample, interact_sample = subregion(st_arr)
    # W_inv=np.eye(15)
    
    # cpu_num = 8 
    # pools = multiprocessing.Pool(processes=cpu_num)
    # results = [pools.apply_async(optimalPso, args=(st_arr, W_inv)) for i in range(cpu_num)]
    # result = [p.get() for p in results]

    # pools.close()
    # pools.join()
    # np.save("D:/czy/est_parameter_for_weight.npy", result)


    """
    estimate weight matrix using upon parameters
    
    """
    # idx = min((result[i, 0]["g_fit"], i) for i in range(cpu_num))
    # a, b, beta, sigma, sigma_e = result[1, 0]["g_best"][0]
    
    # w = evaluateGammaCovBoots(dataset=all_in_st_arr, a=a, b=b, beta=beta, 
    #                           sigma=sigma, sigma_e=sigma_e)
    # np.save("D:/czy/est_weight.npy", w)
    
    
    """
    re-estimated parameters by using upon weight matrix
    """
    # W_inv = pinverse(w)
    
    # cpu_num = 8 
    # pools = multiprocessing.Pool(processes=cpu_num)
    # results = [pools.apply_async(optimalPso, args=(st_arr, W_inv)) for i in range(cpu_num)]
    # result = [p.get() for p in results]

    # pools.close()
    # pools.join()
    # np.save("D:/czy/est_parameter_using_estimated_weight.npy", result)
    
    
    """
    predict by using upon re-estimated parameters
    """
    # idx = min((result[i, 0]["g_fit"], i) for i in range(cpu_num))
    # a, b, beta, sigma, sigma_e = result[1, 0]["g_best"][0]
    # pred_arr = predictAll(a, b, beta, sigma, st_arr)
    # np.save("D:/czy/prediction_array.npy", pred_arr)


    end = time.time()
    print(end - start)
    
    
        