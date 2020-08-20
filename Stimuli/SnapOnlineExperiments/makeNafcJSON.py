# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 11:48:44 2020

@author: StuffDeveloping
"""

import json

# Find detailed documentation here https://snaplabonline.com/task/howto/

instructions = ["Welcome to the actual experiment! "]
feedback = True
holdfeedback = False
feedbackdur = 600 #duration of feedback in ms
serveraudio = True
#estimatedduration: 
randomize = False #randomize trial order
isi = 600 # interstimulus interval in ms




data = {}
data['instructions'] = instructions
data['feedback'] = feedback
data['holdfeedback'] = holdfeedback
data['feedbackdur'] = feedbackdur
data['serveraudio'] = serveraudio
data['randomize'] = randomize
data['isi'] = isi
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
    'stimulus': 'volumeSetStim.wav',
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

data['trials'] = []
data['trials'].append({
    'plugin':"hari-audio-button-response",
    'prompt': "Select the <strong>word</strong> prompted by the voice",
    'choices': ['yo','no','ok'],
    'answer': 3,
    'stimulus': 'mrttrial1.wav',
    'cond': 1,
    'annot': {'SNR': '5'}
    })
    

with open('test.JSON', 'w') as outfile:
    json.dump(data,outfile, indent = 4)


