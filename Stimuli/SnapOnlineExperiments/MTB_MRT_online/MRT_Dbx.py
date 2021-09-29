#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:48:52 2021

@author: ravinderjit
"""

import json
import dropbox
from scipy.io import loadmat 

def dlURL(url):
    ## convert db url to one that can be used to direct download
    dl_url = url[0:url.find('?')]
    dl_url = url[0:url.find('dropbox')] + 'dl.dropboxusercontent' + url[url.find('.com'):url.find('?')]
    return dl_url

def getFiles(folder_path):
    '''
    naming convention of files should have trial_#.wav at end
    '''
    files = dbx.files_list_folder(folder_path)
    fentries = files.entries
    trial_num = []
    wavFiles = []
    for x in range(len(fentries)):
        fname = fentries[x].name
        if fname[len(fname)-4:len(fname)].lower() == '.wav' and fname!='volstim.wav':
            trial_num.append(int(fname[8:-4]))
            wavFiles.append(fname)
    sortIndex = sorted(range(len(trial_num)), key = lambda k: trial_num[k])
    wavFiles = [wavFiles[i] for i in sortIndex]
    return wavFiles

with open('../dbAPIkey.json') as f:
    APIkey = json.load(f)

dbxAPIkey = APIkey['dbxAPIkey'] #importing API key from json file. Doing this to keep key hidden from public repo
dbx = dropbox.Dropbox(dbxAPIkey)

with open('mrtinfo.txt') as f:
    mrtJfile = f.read()
    
mrtJson = json.loads(mrtJfile)

json_fname = 'MRT_MTB' + '.json'
instructions = ["Welcome to the actual experiment! "]
feedback = True
holdfeedback = False
feedbackdur = 500 #duration of feedback in ms
serveraudio = False
#estimatedduration: 
randomize = False #randomize trial order
isi = 750 # interstimulus interval in ms

folder_path = '/OnlineStimWavs/MRT_mtb'
 

mrtJson['instructions'] = instructions
mrtJson['feedback'] = feedback
mrtJson['holdfeedback'] = holdfeedback
mrtJson['feedbackdur'] = feedbackdur
mrtJson['serveraudio'] = serveraudio
mrtJson['randomize'] = randomize
mrtJson['isi'] = isi

flink = dbx.sharing_create_shared_link(folder_path+'/volstim.wav') 
dd_link = dlURL(flink.url) 

mrtJson['volume'] = []
mrtJson['volume'].append({
    'plugin': "html-button-response",
    'prompt': ('Welcome! This task involves listening to sounds and providing '
    'responses. As a first step, we need to set an appropriate volume level for '
    'our task. To begin, make sure you are in a quiet room and wearing your '
    'headphones or earphones. Please do not use desktop speakers or laptop '
    'speakers. Also, wireless/bluetooth headphones/earphones can sometimes '
    'cause problems with our task. So please make sure you are wearing wired ' 
    'headphones/earphones.<p> Please bring down your computer volume to 10-20% ' 
    'of maximum. Then, click <strong>Continue</strong> to proceed.</p>'), 
    'choices': ['Continue']
    })    
mrtJson['volume'].append({
    'plugin': "html-button-response",
    'prompt': ('When you are ready, hit the <strong>Play</strong> button to '
    'play a sample sound.<p> While that sound is playing, adjust your computer '
    'volume up to a comfortable (but not too loud) level.</p>'),
    'choices': ['Play'] ,
    })
mrtJson['volume'].append({
    'plugin': 'hari-audio-button-response',
    'prompt': 'Now adjust your computer volume up to a comfortable (but not too loud) level.',
    'stimulus': dd_link,
    'choices': ["I have adjusted the volume, let's continue"]
    })
mrtJson['volume'].append({
    'plugin':"html-button-response",
    'prompt':('Thanks for adjusting your computer volume. We will use this '
    'volume setting for the remainder of the task. Please do not adjust the '
    'volume anymore throughout the task, as that could lead to sounds being '
    'too loud or too soft.<p>If this sounds OK, click <strong>Continue</strong> '
    'to begin the task.</p>'),
    "choices": ["Continue"], 
    })

wavFiles = getFiles(folder_path)

for tt in range(len(mrtJson['trials'])):
    print('Making DB link for trial: ' + str(tt+1))
    if wavFiles[tt] != mrtJson['trials'][tt]['stimulus']:
        print('ISSUEEE!!!!!!!!!!!!!')
    
    flink = dbx.sharing_create_shared_link(folder_path+'/'+wavFiles[tt])
    dd_link = dlURL(flink.url) 
    mrtJson['trials'][tt]['stimulus'] = dd_link
    

with open(json_fname, 'w') as outfile:
    json.dump(mrtJson,outfile, indent = 4)
    



