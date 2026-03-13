import sys
sys.path.append('.')
from Config import Config
from EventController import EventController
from tqdm import tqdm
from psychopy import event, visual, core
import os
import time
import random
import numpy as np
import pickle

config = Config()

config.addSTI = 'WN15' # WN10 WN15 WN20
config.addTG = 'target/plus.png'
config.COM = '3100'
# eventController = EventController(config.COM)
# eventController.clearEvent()

# number of trials
trialNUM = config.trialNUM
# target position
targetpos = (0, 0) 
# letter detection task
letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# detection list
detection = np.zeros((trialNUM, config.classNUM, 2))

# set the window
win = visual.Window([1920, 1080], monitor="testMonitor", units="pix", fullscr=False, waitBlanking=True, color=(0, 0, 0), colorSpace='rgb255', screen=1, allowGUI=False)
# loading pictures
text = 'Loading...'
text = visual.TextStim(win, pos=[0, 0], text=text, color=(255, 255, 255), colorSpace='rgb255')
text.draw()
win.flip()

picAdd = os.listdir(config.addSTI)
frameSets = []
for noiseINX in tqdm(range(len(picAdd))):
    noisepath = config.addSTI + os.sep + "noise%i" % noiseINX
    noiseAdd = os.listdir(noisepath)
    # display frame
    add = noisepath + os.sep + 'display_frame.png'
    displayFrame = visual.ImageStim(win, image=add, pos=[0, 0], units='pix', flipVert=False)
    
    frameSet = []
    # stimulation frames
    for picINX in range(len(noiseAdd)-1):
        add = noisepath + os.sep + '%i.png' % picINX
        frame = visual.ImageStim(win, image=add, pos=[0, 0], units='pix', flipVert=False)
        frameSet.append(frame)
    
    frameSets.append(frameSet)
# target frame
add = config.addTG
targetFrame = visual.ImageStim(win, image=add, pos=[0,0], size=[config.target_size, config.target_size], units='pix', flipVert=False)
# question
Q = 'Any X?'
Q = visual.TextStim(win, pos=[0, 0], text=Q, color=(255, 255, 255), colorSpace='rgb255')
Q.size = config.target_size

# stimulus start
text = 'press space to begin.'
text = visual.TextStim(win, pos=[0, 0], text=text, color=(255, 255, 255), colorSpace='rgb255')
text.size = config.target_size
text.draw()
win.flip()
event.waitKeys(keyList=['space'])

for i in range(trialNUM):
    
    for class_i in range(config.classNUM):
        
        frameSet = frameSets[class_i]
        # letter list
        letter_list = random.sample(letters, 6)
        if 'X' in letter_list: detection[i,class_i,0] = 1
        # cue
        displayFrame.draw()
        targetFrame.pos = targetpos
        targetFrame.draw()
        win.flip()
        time.sleep(1)
        
        frameINX = 0
        letterINX = 0
        startTime = core.getTime()
        # one stim loop
        while frameINX < len(frameSet):
            if frameINX == 0:
                # eventController.sendEvent(class_i+1)
                pass
            frameSet[frameINX].draw()
            if frameINX % 30 == 0:
                letter = letter_list[letterINX]
                letter_text = visual.TextStim(win, pos=[0, 0], text=letter, color=(255, 0, 0), colorSpace='rgb255')
                letter_text.size = config.target_size
                letterINX += 1
            letter_text.draw()
            win.flip()
            frameINX += 1   
                    
        endTime = core.getTime()
        print("STI ended{}".format(endTime-startTime))
        # clear event
        # eventController.clearEvent()
        Q.draw()
        win.flip() 
        response = event.waitKeys(keyList=['space', 'right'])  
        if 'space' in response: pass
        elif 'right' in response: detection[i,class_i,1] = 1   

text = 'Experiment over.'
text = visual.TextStim(win, pos=[0, 0], text=text, color=(255, 255, 255), colorSpace='rgb255')
text.size = config.target_size
text.draw()
win.flip()
event.waitKeys(keyList=['space'])

infoFolder = os.path.join('info', config.addSTI + '.pickle')
with open(infoFolder, 'wb+') as fp:
    pickle.dump(detection, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
win.close()
core.quit()
