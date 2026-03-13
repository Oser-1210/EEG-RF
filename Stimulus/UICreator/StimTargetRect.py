import numpy as np

class StimTargetRect():
    def __init__(self, rectINX, site_point, cubesize, noise):
        
        x, y = rectINX
        self.site_point = site_point
        self.rect_size = cubesize
        N_x = noise.shape[1]
        self.noise = np.squeeze(noise[:, N_x-1-x, y])
        
    def cal_brightness(self, frame_no):
          
        brightness = self.noise[frame_no]
        return brightness



