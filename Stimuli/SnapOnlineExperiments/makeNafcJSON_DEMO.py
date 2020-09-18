#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 20:34:09 2020

@author: ravinderjit
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:48:44 2020
Author: Ravinderjit Singh
"""

import json
import dropbox
import numpy as np
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
        if fname[len(fname)-4:len(fname)].lower() == '.wav' and fname!='DEMO_volstim.wav':
            trial_num.append(int(fname[fname.find('trial_')+6:fname.find('.wav')]))
            wavFiles.append(fname)
    sortIndex = sorted(range(len(trial_num)), key = lambda k: trial_num[k])
    wavFiles = [wavFiles[i] for i in sortIndex]
    return wavFiles

with open('dbAPIkey.json') as f:
    APIkey = json.load(f)

dbxAPIkey = APIkey['dbxAPIkey'] #importing API key from json file. Doing this to keep key hidden from public repo
dbx = dropbox.Dropbox(dbxAPIkey)

# Find detailed documentation here https://snaplabonline.com/task/howto/

trial_cond = 4
test_cond = 1 #use this for testing accuracy in flow of experiment

json_fname = 'Demo_AMphi_AM' + str(trial_cond) + '.json'
instructions = ["Welcome to the Demo! "]
feedback = True
holdfeedback = False
feedbackdur = 750 #duration of feedback in ms
serveraudio = False
#estimatedduration: 
randomize = False #randomize trial order
isi = 750 # interstimulus interval in ms


folder_path_demo = '/OnlineStimWavs/AMphi/DEMO_AMphi_' + str(trial_cond) # Path to folder in dropbox
folder_path_prac = '/OnlineStimWavs/AMphi/DEMOprac_AMphi_' + str(trial_cond) 
trial_plugin = 'hari-audio-button-response'
trial_prompt = 'Select the stimuli with an <strong>AM phase shift </strong>'
trial_choices = ['1', '2', '3']
demo_info_file = loadmat('DEMO_StimData_' + str(trial_cond) + '.mat')
demo_prac_file = loadmat('DEMOprac_StimData_' + str(trial_cond) + '.mat')
demo_correct_answers = demo_info_file['correct'].squeeze()
prac_correct_answers = demo_prac_file['correct_k'].squeeze()
phi_annotation = demo_info_file['phis'].squeeze()



data = {}
data['instructions'] = instructions
data['feedback'] = feedback
data['holdfeedback'] = holdfeedback
data['feedbackdur'] = feedbackdur
data['serveraudio'] = serveraudio
data['randomize'] = randomize
data['isi'] = isi

flink = dbx.sharing_create_shared_link(folder_path_demo+'/DEMO_volstim.wav')
dd_link = dlURL(flink.url) 

data['volume'] = []
data['volume'].append({
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
data['volume'].append({
    'plugin': "html-button-response",
    'prompt': ('When you are ready, hit the <strong>Play</strong> button to '
    'play a sample sound.<p> While that sound is playing, adjust your computer '
    'volume up to a comfortable (but not too loud) level.</p>'),
    'choices': ['Play'] 
    })
data['volume'].append({
    'plugin': 'hari-audio-button-response',
    'prompt': 'Now adjust your computer volume up to a comfortable (but not too loud) level.',
    'stimulus': dd_link,
    'choices': ["I have adjusted the volume, let's continue"]
    })
data['volume'].append({
    'plugin':"html-button-response",
    'prompt':('Thanks for adjusting your computer volume. We will use this '
    'volume setting for the remainder of the task. Please do not adjust the '
    'volume anymore throughout the task, as that could lead to sounds being '
    'too loud or too soft.<p>If this sounds OK, click <strong>Continue</strong> '
    'to begin the task.</p>'),
    "choices": ["Continue"] 
    })

data['trials'] = []
wavFiles_prac = getFiles(folder_path_prac)

data['trials'].append({
        'plugin': 'html-button-response',
        'prompt': 'INSTRUCTIONS: You will hear three intervals all with two amplitude modulated (AM) tones . '
        'Your goal is to detect the interval where the two AMs have a phase difference. '
        'To DETECT this difference, FOCUS ON finding the stimuli that is different in the  '
        'MIDDLE. IGNORE DIFFERNCES in the BEGINNING and END of the three stimuli, '
        'they are distractions.',
        'choices': ['Continue']
        })

data['trials'].append({
        'plugin': 'html-button-response',
        'prompt': 'INSTRUCTIONS: You will hear three intervals all with two amplitude modulated (AM) tones . '
        'Your goal is to detect the interval where the two AMs have a phase difference. '
        'We will start with some examples. The stimulus you need to detect is in the last interval in these examples ',
        'choices': ['Play']
    })


for k in range(len(wavFiles_prac)):
    flink = dbx.sharing_create_shared_link(folder_path_prac +'/'+wavFiles_prac[k])
    dd_link = dlURL(flink.url)
    
    if np.mod(k+1,5) ==1 and not k == 0:
        data['trials'].append({
            'plugin' : 'html-button-response',
            'prompt' : 'Now we will practice with smaller phase differences (i.e. it will be harder). '
            'We will start with 3 examples and then have 2 practices again.',
            'choices': ['Play']
            })

    if np.mod(k+1,5) <= 3 and not np.mod(k+1,5)==0 :
        data['trials'].append({
                'plugin': trial_plugin,
                'prompt': 'Example ' + str(np.mod(k,5)+1) + '/3 ... listen for the difference in interval 3',
                'choices': ['Continue'],
                'stimulus':  dd_link,
                'answer': 1,
                'cond': trial_cond,
                'trialfeedback': False
            })
    if np.mod(k+1,5) == 4:
        data['trials'].append({
            'plugin': 'html-button-response',
            'prompt': 'Now for a couple practice trials. This time the target will be randomly played. '
            'Your goal is to select which interval the target was played in.',
            'choices': ['Play']
            })
    if np.mod(k+1,5) == 4 or np.mod(k+1,5) == 0:
        data['trials'].append({
            'plugin': trial_plugin,
            'prompt': trial_prompt,
            'choices': trial_choices,
            'answer': int(prac_correct_answers[k]),
            'cond': trial_cond,
            'stimulus': dd_link
            })


data['trials'].append({
    'plugin': 'html-button-response',
    'prompt': 'Now for a full practice run mixed with easy and hard trials. Try your best! ',
    'choices': ['Play']
    })

wavFiles = getFiles(folder_path_demo)
for i in range(len(wavFiles)):
    flink = dbx.sharing_create_shared_link(folder_path_demo+'/'+wavFiles[i])
    dd_link = dlURL(flink.url) 
    data['trials'].append({
        'plugin': trial_plugin,
        'prompt': trial_prompt,
        'choices': trial_choices,
        'answer': int(demo_correct_answers[i]),
        'stimulus': dd_link,
        'cond': test_cond,
        'annot': {'phi': int(phi_annotation[i])}
        })
    

with open(json_fname, 'w') as outfile:
    json.dump(data,outfile, indent = 4)
    
    

        
            
            
            
    


\