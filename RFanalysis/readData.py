
import numpy as np
import mne
import os
import pickle
from scipy import signal
from tqdm import tqdm

class readData():
    
    def __init__(self, filename, tstart=0, tend=3, lag=0.14, srate=250) -> None:
        # file address
        self.filename = filename
        self.subjects = []
        
        # parameters
        self.srate = srate
        self.tstart = tstart
        self.tend = tend
        self.lag = lag
        self.downRatio = int(self.srate*(self.tend-self.tstart+self.lag))
        
        self.filters = self._initFilter()
        
        self.picklename = self.filename.strip(self.filename.split(os.sep)[-1]) + 'datasets'
        if os.path.exists(self.picklename) is False:
            os.makedirs(self.picklename)
        
        mne.set_log_level(verbose='ERROR')
    
        ahList = os.listdir(self.picklename)
        self.alreadyHave = {ah:len(os.listdir(os.path.join(self.picklename, ah))) for ah in ahList}
    
    def _initFilter(self):
    
        # notch
        fs = 250.0  # Sample frequency (Hz)
        f0 = 50.0  # Frequency to be removed from signal (Hz)
        Q = 30.0  # Quality factor

        b, a = signal.iirnotch(f0, Q, fs)
        notch = [b, a]

        # band pass
        b, a = signal.butter(N=5, Wn=80, fs=fs, btype='lowpass')
        bp = [b, a]

        return notch, bp
    
    def preprocess(self,x):
    
        from scipy.signal import resample
        x = resample(x,self.downRatio,axis=-1)

        notchFilter, bpFilter = self.filters

        b_notch, a_notch = notchFilter
        b_bp, a_bp = bpFilter

        x_notched = signal.filtfilt(b_notch, a_notch, x, axis=-1)
        x_filtered = signal.filtfilt(b_bp, a_bp, x_notched, axis=-1)

        processed = x_filtered - np.mean(x_filtered,axis=-1,keepdims=True)
        
        return processed
        
    def readRaw(self):
        # get subject paths
        self.getSubject()
        
        for subpath in tqdm(self.subListpath):
            # for subject level
            subName = subpath.split(os.sep)[-1]
            subpicklepath = os.path.join(self.picklename,subName)
            if os.path.exists(subpicklepath) == False:
                os.makedirs(subpicklepath)
            
            dateList = os.listdir(subpath)
            for dateName in dateList:
                datepath = subpath+os.sep+dateName
                getList = os.listdir(datepath)
                
                epoches = {}
                for getName in getList:
                    
                    getpath = os.path.join(datepath, getName)
                    # split raw into epoch
                    epoch = self.getEpoch(getpath)
                    if epoch != []:
                        epoches[os.path.splitext(getName)[0]] = epoch
                    else: print('Failed to read data!')
                        
 
                with open('%s\%s.pickle' % (subpicklepath, dateName), "wb+") as fp:
                    pickle.dump(epoches, fp, protocol=pickle.HIGHEST_PROTOCOL)
        return
    
    def getEpoch(self, getpath):
        
        # get raw data
        raw = mne.io.read_raw_cnt(
            input_fname = getpath,
            data_format = 'auto',
            preload = True,
            date_format = 'mm/dd/yy')
        # get epoch
        task_event, task_dict = self.getEvent(raw)
        taskEpoch = mne.Epochs(raw, task_event, event_id=task_dict, 
                            tmin=self.tstart, tmax=self.tend+self.lag, preload=True, baseline=None)
        # get data
        X = taskEpoch.get_data()[:,:-1]
        X = self.preprocess(X)
        
        y = []
        newdict = {v: k for k, v in task_dict.items()}
        for event in task_event[:,-1]:
            y.append(int(newdict.get(event)))
        y = np.array(y)
        Channels = taskEpoch.ch_names[:-1]
        # data dict
        data = dict(
            X = X,
            y = y,
            channel = Channels)
    
        return data
        
    def getEvent(self, raw):
        
        events, event_dict = mne.events_from_annotations(raw)
        # event mistake must be settled
        valid_dict = {k: v for k, v in event_dict.items() if int(k) < 255}
        valid_event =  np.stack([e for e in events if e[-1] in [*valid_dict.values()]])
        x = np.squeeze(raw['Trigger'][0])
        # correct index
        onset = np.squeeze(np.argwhere(np.diff(x) > 0))
        valid_event[:, 0] = onset[:len(valid_event)]
        
        return valid_event, valid_dict             
            
    def getSubject(self):
        
        subList = os.listdir(self.filename)
        for sub, times in self.alreadyHave.items():
            if len(os.listdir(os.path.join(self.filename,sub))) == times:
                subList.remove(sub)
        self.subListpath = [os.path.join(self.filename, subName) for subName in subList]
        
if __name__ == '__main__':
        
    filename = os.path.join(os.getcwd(),'data')
    # filename = os.path.join(os.getcwd(),'highdensity')
        
    pickleMaker = readData(filename=filename)
    pickleMaker.readRaw()
        