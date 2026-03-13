from tqdm import tqdm
import os
import numpy as np
from methods import *
import pickle
import pandas as pd

def find_positions(arr1, arr2):
    positions = []
    for elem in arr1:
        if elem in arr2:
            positions.append(arr2.index(elem)) # 找到元素在arr2中的位置
        else:
            positions.append(-1) # 如果找不到该元素，返回-1
    return positions

# parameters
paradigms = ['WN10', 'WN15', 'WN20']
shuffle_num = 10
seed_ori = 2024
chn19 = ['P7','C5','PO7','P3','Cb1','P1','C1','O1','Pz','POz','Oz','O2','PO4','P2','Cb2','P4','PO8','P6','P8']
pix = {'WN10':1,
       'WN15':1.5,
       'WN20':2}
noisename = os.path.join(os.getcwd(), 'noise')
stimulus = {}
for noise in os.listdir(noisename):
    noisepath = os.path.join(noisename, noise)
    with open(noisepath, 'rb') as fp:
        stimulus[os.path.splitext(noise)[0]] = pickle.load(fp)

filename = os.path.join(os.getcwd(), 'highdensity')
STRFs = {}
frames = []
for sub in os.listdir(filename):
    # read data
    subpath = os.path.join(filename, sub)
    date = os.listdir(subpath)[0]
    getpath = os.path.join(subpath, date)
    with open(getpath, 'rb') as fp:
        data = pickle.load(fp)

    print('Subject '+sub+':')

    strfs = {}
    for p in tqdm(paradigms, desc='Paradigms'):
        shuffle_data = []
        data_p = data[p]
        stim_p = stimulus[p]
    
        strf = STRF()
        compSTRF = strf.fit_tdca(data_p, stim_p, None)
        shuffle_data.append(compSTRF)
        
        for seed in range(shuffle_num):
            strf = STRF()
            compSTRF = strf.fit_tdca(data_p, stim_p, seed+seed_ori)
            shuffle_data.append(compSTRF)
        
        strfs[p] = shuffle_data

    STRFs[sub] = strfs


with open(os.path.join('results/STRF_shuffle66.pickle'), "wb+") as fp:
    pickle.dump(STRFs, fp, protocol=pickle.HIGHEST_PROTOCOL)


    
