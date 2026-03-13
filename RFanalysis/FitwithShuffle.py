from tqdm import tqdm
import os
import numpy as np
from methods import *
import pickle
import pandas as pd

# parameters
paradigms = ['WN10', 'WN15', 'WN20']
shuffle_num = 10
seed_ori = 2024
pix = {'WN10':1,
       'WN15':1.5,
       'WN20':2}
noisename = os.path.join(os.getcwd(), 'noise')
stimulus = {}
for noise in os.listdir(noisename):
    noisepath = os.path.join(noisename, noise)
    with open(noisepath, 'rb') as fp:
        stimulus[os.path.splitext(noise)[0]] = pickle.load(fp)

filename = os.path.join(os.getcwd(), 'datasets')
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
        chnName = data_p['channel']
        # aligned
        strf = STRF()
        chnSTRF = strf.fit_only(data_p, stim_p, None)
        shuffle_data.append(chnSTRF)
        
        for seed in range(shuffle_num):
            strf = STRF()
            chnSTRF = strf.fit_only(data_p, stim_p, seed+seed_ori)
            shuffle_data.append(chnSTRF)
        
        strfs[p] = shuffle_data

    STRFs[sub] = strfs


with open(os.path.join('results/STRF_shuffle.pickle'), "wb+") as fp:
    pickle.dump(STRFs, fp, protocol=pickle.HIGHEST_PROTOCOL)


    
