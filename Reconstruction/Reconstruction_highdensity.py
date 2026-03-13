from tqdm import tqdm
import os
import numpy as np
from methods import *
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


def stablizeSTRF(data, k_l, multi):

    _, row, column = data[0].shape
    rf = np.zeros((k_l, row, column))
    W = np.zeros((row, column))
    for i in range(row):
        for j in range(column):
            shuffline = []
            for shf in range(len(data)):
                if shf == 0: 
                    aligned = np.flip(data[shf][:,row-i-1,j], axis=0)
                    aligned = aligned[:k_l]
                    rf[:,row-i-1,j] = aligned
                else:
                    kernal = np.flip(data[shf][:,row-i-1,j], axis=0)
                    kernal = kernal[:k_l]
                    shuffline.append(kernal)
            shuffline = np.stack(shuffline)
            overall_mean = np.mean(shuffline, axis=0)
            overall_std_dev = np.std(shuffline, axis=0, ddof=1)
            for idx in range(aligned.shape[0]):
                if abs(overall_mean[idx] - aligned[idx]) < multi*overall_std_dev[idx]:
                    aligned[idx] = 0
                else:
                    aligned[idx] = min(abs(aligned[idx]-(multi*overall_std_dev[idx]+overall_mean[idx])), abs(aligned[idx]-(-multi*overall_std_dev[idx]+overall_mean[idx])))
            weight = np.square(aligned)
            weight = weight[weight!=0]
            if weight.size == 0: W[row-i-1,j] = 0
            else:
                W[row-i-1,j] = np.mean(weight)
    weighted_rf = rf*W

    return rf, weighted_rf

def find_positions(arr1, arr2):
    positions = []
    for elem in arr1:
        if elem in arr2:
            positions.append(arr2.index(elem)) # 找到元素在arr2中的位置
        else:
            positions.append(-1) # 如果找不到该元素，返回-1
    return positions


# parameters
paradigms = ['WN15', 'WN20']
trainSet = 10
randnum = 20
shuffle_num = 10
seed_ori = 2024
k_l = 100
multi = 3
chn19 = ['P7','C5','PO7','P3','Cb1','P1','C1','O1','Pz','POz','Oz','O2','PO4','P2','Cb2','P4','PO8','P6','P8']
chnlabel = ['chn19','chn66']
noisename = os.path.join(os.getcwd(), 'noise')
stimulus = {}
for noise in os.listdir(noisename):
    noisepath = os.path.join(noisename, noise)
    with open(noisepath, 'rb') as fp:
        stimulus[os.path.splitext(noise)[0]] = pickle.load(fp)

filename = os.path.join(os.getcwd(), 'highdensity')

frames = []
for sub in os.listdir(filename):
    # read data
    subpath = os.path.join(filename, sub)
    date = os.listdir(subpath)[0]
    getpath = os.path.join(subpath, date)
    with open(getpath, 'rb') as fp:
        data = pickle.load(fp)

    print('Subject '+sub+':')

    for p in tqdm(paradigms, desc='Paradigms'):
        for cl in chnlabel:
            shuffle_data = []
            data_p = data[p]
            stim_p = stimulus[p]
            print(cl)

            strf = STRF(n_components=1)
            S = strf.Upresample(stim_p)
            S = np.flip(S, axis=2)
            if cl == 'chn19':
                chn66 = data_p['channel']
                position = find_positions(chn19, chn66)
                X = data_p['X'][:,position,int(strf.T_futu*strf.srate):]
            else:
                X = data_p['X'][:,:,int(strf.T_futu*strf.srate):]
            X = strf.preprocess(X)
            y = data_p['y']
            R = strf.tdca_filter(X,y)
            print(len(strf.filters[0]))
            unique_y = np.unique(y)
            for rnd in range(randnum):
                index = np.arange(0,strf.montage)
                np.random.seed(rnd)
                np.random.shuffle(index)
                rndR = R[:,index,:]
                rndy = unique_y[index]
                rndS = S[index]
                trainR = rndR[:,:trainSet,:]
                trainy = rndy[:trainSet]
                testR = rndR[:,trainSet:,:]
                testy_uniq = rndy[trainSet:]
                testX = []
                testy = []
                testS = rndS[trainSet:]
                for xid, x in enumerate(X):
                    if y[xid] in testy_uniq:
                        testX.append(x)
                        testy.append(y[xid])
                testX = np.stack(testX)
                testy = np.array(testy)
                shuffle_data = []
                # aligned
                compR = np.squeeze(trainR[0,:,:])
                Kz = 0
                K = 0
                for epochINX, epoch in enumerate(compR):

                    stim = S[trainy[epochINX]-1]
                    K_raw, K_norm = strf.reverse_correlation(epoch, stim)
                    Kz += K_norm
                Kz = Kz/compR.shape[0]
                shuffle_data.append(Kz)
                
                for seed in range(shuffle_num):
                    np.random.seed(seed+seed_ori)
                    the_trainy = trainy
                    np.random.shuffle(the_trainy)
                    Kz = 0
                    K = 0
                    for epochINX, epoch in enumerate(compR):

                        stim = S[the_trainy[epochINX]-1]
                        K_raw, K_norm = strf.reverse_correlation(epoch, stim)
                        Kz += K_norm
                    Kz = Kz/compR.shape[0]
                    shuffle_data.append(Kz)

                rf, w_rf = stablizeSTRF(shuffle_data, k_l, multi)
                corr_w = strf.predict_series(testS, testR, w_rf)
                corr_w = corr_w.mean()
                time_series = np.arange(0,3,0.3)+0.3
                for t in time_series:
                    tlen = int(t*strf.srate)
                    result_w = strf.predict_epoch(testX[:,:,:tlen], testS[:,:tlen,:,:],testy_uniq,w_rf)
                    accuracy_w = accuracy_score(np.array(result_w),testy)
                    frame = pd.DataFrame({
                        'subject':[sub],
                        'time':[t],
                        'accuracy':[accuracy_w],
                        'corr':[corr_w],
                        'rnd':[rnd],
                        'chnlabel':[cl],
                        'paradigm':[p],
                    })
                    frames.append(frame)

df = pd.concat(frames, axis=0, ignore_index=False)
df.to_csv('results/Recon_highdensity.csv')



    
