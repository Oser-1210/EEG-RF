import sys
sys.path.append('.')

from Config import Config
from UICreator.UIFactory import UIFactory

params = [[2, 8], [1.5, 10], [1, 16]]
paths = ['WN20', 'WN15', 'WN10']

for i in range(len(params)):
    
    config = Config(blocksize=params[i][0], layout=(params[i][1], params[i][1]), path=paths[i])

    factory = UIFactory(config)
    for j in range(config.classNUM):
        frames = factory.getFrames(j)
        factory.saveFrames(frames, j)

