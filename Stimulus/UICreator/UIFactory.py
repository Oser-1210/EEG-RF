import numpy as np
import matplotlib.patches as patches
from UICreator.StimTargetRect import StimTargetRect
import matplotlib.pyplot as plt
import os
import matplotlib
from tqdm import tqdm
import pickle
matplotlib.use('agg')
    
class frameLoader():
    def __init__(self) -> None:

        self.frameSet = None
        self.displayFrame = None

        pass


def fig2data(f):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    f.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = f.canvas.get_width_height()
    buf = np.fromstring(f.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


class UIFactory:

    def __init__(self,config):

        self.rowNUM, self.columnNUM = config.layout
        self.x_w, self.y_w = config.window_size
        self.refreshRate = config.srate
        self.classNUM = config.classNUM
        self.stiLEN = config.stiLEN
        self.path = config.path
        kind = config.kind
        filtered = config.filtered
        tau = config.tau
        
        self.maxFrames = int(self.refreshRate*self.stiLEN)

        self.cubeSize = config.cubeSize

        self.saveFolder = os.path.join(os.getcwd(), self.path)
        if os.path.exists(self.saveFolder) is False:
            os.makedirs(self.saveFolder)
        # generate 2-D stimulus    
        self.getStimulus(kind, filtered, tau)
            
    def getStimulus(self, kind, filtered, tau):
        
        N_t = self.maxFrames
        N_x = self.rowNUM
        N_y = self.columnNUM
        
        tau_fs = tau*self.refreshRate
        alpha = (tau_fs/(tau_fs+1))
        scaling_f = np.sqrt((1-alpha)/(1+alpha))
        
        noise_shape = (N_t, N_x, N_y)

        if kind == 'wn':
            # WN stimulus
            # random numbers
            noise = np.zeros((self.classNUM,)+noise_shape)
            for i in range(self.classNUM):
                np.random.seed(i)
                wn_noise = np.random.uniform(0, 1, noise_shape)
                noise[i] = wn_noise
            if filtered == True:
                # first order low-pass, wc = 1/(tau+1/srate)
                np.random.seed(0)
                # generate initial state for low_pass
                init_state = np.random.uniform(0, scaling_f, noise_shape[1:])
                wn_noise = np.concatenate((init_state[np.newaxis,:,:], wn_noise), 0)
                # low_pass filtered
                wn_noise = self.lowpass1_order(wn_noise, tau_fs)/scaling_f
                wn_noise = wn_noise[1:,:,:]
                # normalization?
                
            self.noise = noise
        
        elif kind == 'gauss':
            # Gauss stimulus
            np.random.seed(0)
            gauss_noise = np.random.normal(0, 1, noise_shape)
            if filtered == True:
                # initial state
                np.random.seed(0)
                init_state = np.random.normal(0, scaling_f, noise_shape[1:])
                gauss_noise = np.concatenate((init_state[np.newaxis,:,:], gauss_noise), 0)
                # low_pass filtered
                gauss_noise = self.lowpass1_order(gauss_noise, tau_fs)/scaling_f
                gauss_noise = gauss_noise[1:,:,:]
                # normalization?
            
            self.noise = gauss_noise
        
        elif kind == 'binary':
            # Binary stimulus
            noise = np.zeros((self.classNUM,)+noise_shape)
            for i in range(self.classNUM):
                np.random.seed(i)
                binary_noise = np.random.randint(0, 2, noise_shape)
                noise[i] = binary_noise
        
        self.noise = noise
        noiseFolder = os.path.join('noise', self.path+'.pickle')
        with open(noiseFolder, 'wb+') as fp:
            pickle.dump(noise, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    def lowpass1_order(signal, tau):
        ''' first order low-pass'''
        N = signal.shape[0]
        tau = float(tau)
        out = np.zeros(signal.shape)
        alpha = (tau/(tau+1))
        # initial condition
        out[0,:] = signal[0,:]
        for i in np.arange(1, N):
            out[i,:] = signal[i,:]*(1.-alpha) + out[i-1,:]*alpha
        return out
        
    def getFrames(self, class_i):

        rectSet = []
        # 
        for colINX in range(self.columnNUM):
            for rowINX in range(self.rowNUM):
                # left-bottom
                target_site_point = [colINX*self.cubeSize, rowINX*self.cubeSize]
                rectINX = (rowINX, colINX)
                rectSet.append(StimTargetRect(rectINX, target_site_point, self.cubeSize, np.squeeze(self.noise[class_i])))

        frameSet = []
        for N in tqdm(range(self.maxFrames+1)):
            # loop 
            f = plt.figure(figsize=(self.x_w/100, self.y_w/100), facecolor='none', dpi=100)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0,
                                right=1, hspace=0, wspace=0)

            plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
            current_axis = plt.gca()

            for rect in rectSet:
                # generate cubes
                if N == 0:
                    brightness = 0.5
                else:
                    brightness = rect.cal_brightness(N-1)
                x_loc = rect.site_point[0] / self.x_w
                y_loc = rect.site_point[1] / self.y_w
                x_size = rect.rect_size / self.x_w
                y_size = rect.rect_size / self.y_w

                # a cube
                cube = patches.Rectangle((x_loc,y_loc),
                                         x_size,y_size,
                                         linewidth=1, facecolor=[brightness, brightness, brightness], alpha=1)

                current_axis.add_patch(cube)

            plt.axis('off')
            frameSet.append(fig2data(f))
            plt.close(f)

        
        frames = frameLoader()
        frames.displayFrame = frameSet.pop(0)
        frames.frameSet = frameSet

        return frames


    def saveFrames(self,frames, class_i):

        frameSet = frames.frameSet
        path = os.path.join(self.saveFolder, 'noise'+str(class_i))
        if os.path.exists(path) is False:
            os.makedirs(path)
        for i, frame in enumerate(frameSet):
            plt.imsave(path+os.sep+'%i.png' % i, frame)
            
        displayFrame = frames.displayFrame
        plt.imsave(path+os.sep+'display_frame.png', displayFrame)



        



