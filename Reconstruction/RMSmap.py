import pickle
import numpy as np
import os
from tqdm import tqdm
import math
from scipy.optimize import curve_fit
import pandas as pd

def RMSmap(kernal, l_min, l_max, srate):

    kernal = kernal[int(l_min*srate):int(l_max*srate)]      
    kernal_square = np.square(kernal)
    RMS = kernal_square.mean(axis=0)
    RMS = np.sqrt(RMS)

    return RMS

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
    return offset + amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))
    )

def gaussian_2d_wrapper(coords, x0, y0, sigma_x, sigma_y, amplitude, offset):
    x, y = coords
    return gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset).ravel()

def fit_gaussian(data):

    row, column = data.shape
    x = np.linspace(0,row-1,row)
    y = np.linspace(0,column-1,column)
    x, y = np.meshgrid(x, y)

    initial_guess = ((row-1)/2, (column-1)/2, 2, 2, np.max(data), np.min(data))
    popt, pcov = curve_fit(gaussian_2d_wrapper, (x, y), data.ravel(), p0=initial_guess, bounds=(0,[row,column,row/2,column/2,20,20]))
    (x0, y0, sigma_x, sigma_y, amplitude, offset) = popt
    fwhm_x = 2 * np.sqrt(2 * np.log(2)) * abs(sigma_x)
    fwhm_y = 2 * np.sqrt(2 * np.log(2)) * abs(sigma_y)
    size = (fwhm_x+fwhm_y)/2
    loc_x = x0-(row-1)/2
    loc_y = y0-(column-1)/2

    return loc_x, loc_y, x0, y0, size

with open('results/STRF_shuffle.pickle','rb') as fp:
    data = pickle.load(fp)

k_l = 100
multi = 3
srate = 250
l_min = 0.04
l_max = 0.2

pix = {'WN10':1,
       'WN15':1.5,
       'WN20':2}
reliableRMS = {}
reliableRF = {}
reliableGauss = {}
frames = []

for sub in tqdm(data):
    subdata = data[sub]
    s_RF = {}
    s_rms = {}
    s_gauss = {}
    for p in subdata:
        p_data = subdata[p]
        p_RF = {}
        p_rms = {}
        p_gauss = {}
        # for chn in channels:
        for chn in p_data[0].keys():
            _, row, column = p_data[0][chn].shape
            # rf = np.zeros((k_l, row, column))
            weighted_rf = np.zeros((k_l, row, column))
            W = np.zeros((row, column))
            for i in range(row):
                for j in range(column):
                    shuffline = []
                    for shf in range(len(p_data)):
                        if shf == 0: 
                            aligned = np.flip(p_data[shf][chn][:,row-i-1,j], axis=0)
                            aligned = aligned[:k_l]
                            weighted_rf[:,row-i-1,j] = aligned
                        else:
                            kernal = np.flip(p_data[shf][chn][:,row-i-1,j], axis=0)
                            kernal = kernal[:k_l]
                            shuffline.append(kernal)
                    shuffline = np.stack(shuffline)
                    overall_mean = np.mean(shuffline, axis=0)
                    overall_std_dev = np.std(shuffline, axis=0, ddof=1)
                    for idx in range(aligned.shape[0]):
                        # aligned[idx] = (aligned[idx] - overall_mean[idx])/overall_std_dev[idx]*aligned[idx]
                        if abs(overall_mean[idx] - aligned[idx]) < multi*overall_std_dev[idx]:
                            aligned[idx] = 0
                        else:
                            aligned[idx] = min(abs(aligned[idx]-(multi*overall_std_dev[idx]+overall_mean[idx])), abs(aligned[idx]-(-multi*overall_std_dev[idx]+overall_mean[idx])))
                    # rf[:,row-i-1,j] = aligned
                    weight = np.square(aligned)
                    weight = weight[weight!=0]
                    if weight.size == 0: W[row-i-1,j] = 0
                    else:
                        W[row-i-1,j] = np.mean(weight)
            weighted_rf = weighted_rf*W
            rms = RMSmap(weighted_rf, l_min, l_max, srate)
            # guass
            max_index = np.unravel_index(np.argmax(rms), rms.shape)
            xdx, ydx = max_index
            if math.ceil(row/4)-1 < xdx < math.floor(row/4*3) and math.ceil(column/4)-1 < ydx < math.floor(column/4*3):
                try:
                    p_gauss[chn] = fit_gaussian(rms)
                    loc_x, loc_y, _, _, size = p_gauss[chn]
                    if size*pix[p] > 8 or loc_x*pix[p]< -4 or loc_x*pix[p]>4 or loc_y*pix[p] < -4 or loc_y*pix[p] > 4: 
                        p_gauss[chn] = []
                    else:
                        frame = pd.DataFrame({
                            'subject':[sub],
                            'channel':[chn],
                            'size':[size*pix[p]],
                            'x':[loc_x*pix[p]],
                            'y':[loc_y*pix[p]],
                            'paradigm':[p]})
                        frames.append(frame)
                except:
                    print('Failed')
                    print(sub,p,chn)
                    p_gauss[chn] = []
            else: p_gauss[chn] = []
            p_RF[chn] = weighted_rf
            p_rms[chn] = rms
        s_RF[p] = p_RF
        s_rms[p] = p_rms
        s_gauss[p] = p_gauss
    reliableRF[sub] = s_RF
    reliableRMS[sub] = s_rms
    reliableGauss[sub] = s_gauss

with open(os.path.join('results/reliableRF.pickle'), "wb+") as fp:
    pickle.dump(reliableRF, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join('results/reliableRMS.pickle'), "wb+") as fp:
    pickle.dump(reliableRMS, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join('results/reliableGauss.pickle'), "wb+") as fp:
    pickle.dump(reliableGauss, fp, protocol=pickle.HIGHEST_PROTOCOL)
df = pd.concat(frames, axis=0, ignore_index=True)
df.to_csv('results/RFsize_reliable.csv')


