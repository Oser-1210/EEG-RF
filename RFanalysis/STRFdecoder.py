from tqdm import tqdm
import os
import numpy as np
from methods import *
import pickle
import pandas as pd

# parameters
paradigms = ['WN10', 'WN15', 'WN20']
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
RMSmap = {}
Gauss = {}
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
    rmsmap = {}
    gauss = {}
    for p in tqdm(paradigms, desc='Paradigms'):
        data_p = data[p]
        stim_p = stimulus[p]
        chnName = data_p['channel']
        strf = STRF()
        chnSTRF, chnRMSmap, chnGauss = strf.fit(data_p, stim_p)
        strfs[p] = chnSTRF
        rmsmap[p] = chnRMSmap
        for ch in chnName:
            if chnGauss[ch] == []: continue
            loc_x, loc_y, _, _, size = chnGauss[ch]
            if size*pix[p] > 8:
                chnGauss[ch] = []
            else:
                frame = pd.DataFrame({
                    'subject':[sub],
                    'channel':[ch],
                    'size':[size*pix[p]],
                    'x':[loc_x*pix[p]],
                    'y':[loc_y*pix[p]],
                    'paradigm':[p]
                })
                frames.append(frame)

        gauss[p] = chnGauss
       
    Gauss[sub] = gauss
    STRFs[sub] = strfs
    RMSmap[sub] = rmsmap

with open(os.path.join('results/STRF.pickle'), "wb+") as fp:
    pickle.dump(STRFs, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join('results/RMSmap.pickle'), "wb+") as fp:
    pickle.dump(RMSmap, fp, protocol=pickle.HIGHEST_PROTOCOL)   
with open(os.path.join('results/Gauss.pickle'), "wb+") as fp:
    pickle.dump(Gauss, fp, protocol=pickle.HIGHEST_PROTOCOL)   
df = pd.concat(frames, axis=0, ignore_index=True)
df.to_csv('results/RFsize.csv')

    
