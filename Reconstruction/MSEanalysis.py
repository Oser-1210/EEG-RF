from tqdm import tqdm
import os
import numpy as np
from methods import *
import pickle
import pandas as pd

# parameters
paradigms = ['WN15', 'WN20']
shuffle_num = 20
k_length = 0.4
noisename = os.path.join(os.getcwd(), 'noise')
stimulus = {}
for noise in os.listdir(noisename):
    noisepath = os.path.join(noisename, noise)
    with open(noisepath, 'rb') as fp:
        stimulus[os.path.splitext(noise)[0]] = pickle.load(fp)

filename = os.path.join(os.getcwd(), 'datasets')
MSEs = {}

for sub in os.listdir(filename):
    # read data
    subpath = os.path.join(filename, sub)
    date = os.listdir(subpath)[0]
    getpath = os.path.join(subpath, date)
    with open(getpath, 'rb') as fp:
        data = pickle.load(fp)

    print('Subject '+sub+':')

    mses = {}
    for p in tqdm(paradigms, desc='Paradigms'):
        data_p = data[p]
        stim_p = stimulus[p]
        # aligned
        strf = STRF()
        mse, kernals = strf.mse_train(data_p, stim_p, shuffle_num, k_length)
     
        mses[p] = [mse, kernals]
    MSEs[sub] = mses


with open(os.path.join('results/MSEtest.pickle'), "wb+") as fp:
    pickle.dump(MSEs, fp, protocol=pickle.HIGHEST_PROTOCOL)


    
