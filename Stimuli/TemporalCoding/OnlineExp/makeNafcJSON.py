# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:48:44 2020
Author: Ravinderjit Singh
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
            trial_num.append(int(fname[fname.find('trial_')+6:fname.find('.wav')]))
            wavFiles.append(fname)
    sortIndex = sorted(range(len(trial_num)), key = lambda k: trial_num[k])
    wavFiles = [wavFiles[i] for i in sortIndex]
    return wavFiles

with open('../../SnapOnlineExperiments/dbAPIkey.json') as f:
    APIkey = json.load(f)

dbxAPIkey = APIkey['dbxAPIkey'] #importing API key from json file. Doing this to keep key hidden from public repo
dbx = dropbox.Dropbox(dbxAPIkey)

# Find detailed documentation here https://snaplabonline.com/task/howto/

AM =4
trial_cond = 1

json_fname = 'AMphi_AM' + str(AM) + '.json'
instructions = ["Welcome to the actual experiment! "]
feedback = True
holdfeedback = False
feedbackdur = 750 #duration of feedback in ms
serveraudio = False
#estimatedduration: 
randomize = False #randomize trial order
isi = 0 # interstimulus interval in ms


folder_path = '/OnlineStimWavs/AMphi/AMphi_' + str(AM) # Path to folder in dropbox
trial_plugin = 'hari-audio-button-response'
trial_prompt = 'Select the interval <strong> most different from the reference. </strong> <br> Stimuli Order: Reference, Stim A, Stim B, Stim C'
trial_choices = ['A', 'B', 'C']
stim_info_file = loadmat('StimData_' + str(AM) + '.mat')
correct_answers = stim_info_file['correct'].squeeze()
phi_annotation = stim_info_file['phis'].squeeze()



data = {}
data['instructions'] = instructions
data['feedback'] = feedback
data['holdfeedback'] = holdfeedback
data['feedbackdur'] = feedbackdur
data['serveraudio'] = serveraudio
data['randomize'] = randomize
data['isi'] = isi

flink = dbx.sharing_create_shared_link(folder_path+'/volstim.wav')
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
    'choices': ['Play'] ,
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
    "choices": ["Continue"], 
    })


wavFiles = getFiles(folder_path)

data['trials'] = []
for i in range(len(wavFiles)):
    flink = dbx.sharing_create_shared_link(folder_path+'/'+wavFiles[i])
    dd_link = dlURL(flink.url) 
    data['trials'].append({
        'plugin': trial_plugin,
        'prompt': trial_prompt,
        'choices': trial_choices,
        'answer': int(correct_answers[i])-1,
        'stimulus': dd_link,
        'cond': trial_cond,
        'annot': {'phi': int(phi_annotation[i])}
        })
    

with open(json_fname, 'w') as outfile:
    json.dump(data,outfile, indent = 4)
    
    

        
            
            
            
    


