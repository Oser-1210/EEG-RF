import math
class Config():
    
    def __init__(self, srate=60, stiLEN=3, trialNUM = 6, blocksize = 1, classNUM = 20, layout=(10, 10), kind='wn', filtered=False, tau=0.05, path=None) -> None:
        
        self.srate = srate
        self.stiLEN = stiLEN
        self.trialNUM = trialNUM
        self.classNUM = classNUM
        # screen parameters 24.5 inch, 1920 × 1080 
        Diagonal = 62.23 # cm
        distance = 65 # cm eye to screen
        ratio = Diagonal/math.sqrt(1920**2+1080**2)
        self.cubeSize = int((blocksize/180)*math.pi*distance/ratio)

        x, y = layout
        self.window_size = (x*self.cubeSize, y*self.cubeSize)
        self.screen_size = (1920, 1080)
        self.layout = layout           

        self.target_size = 40
        self.kind = kind
        self.filtered = filtered
        self.tau = tau
        
        self.path = path
        self.addSTI = None
        self.addTG = None
        
        pass