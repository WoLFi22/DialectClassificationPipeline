#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pydub import AudioSegment
from random import sample
from pathlib import Path
import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
import math
import os


# ## Create df with all classes, speaker and files
# ### Parameters:
# 
# - **path**:  
#   The path of the original audio files. Each folder within this path represents a dialect, and subfolders within each dialect folder should contain audio files from different speakers.  
#   
# - **path_aug**:  
#   The path of the augmented audio files. Each folder within this path represents a dialect, and subfolders within each dialect folder should contain audio files from different speakers.  
# 
# - **name_aug**:  
#   Used for storing the resulting DataFrame as './All_Files_' + name_aug + '.pkl'.
#   
# - **test_only**:  
#   True if the run is only for making predictions.
# 
# ### Returns:
# 
# Saves a DataFrame with the following columns:
# 
# - _'dialect'_: Represents the class (dialect) of the audio.
# - _'speaker'_: Represents the speaker or place associated with the audio.
# - _'file\_name'_: Represents the name of the audio file.
# - _'length'_: Represents the number of samples in the audio file.
# - _'file\_path'_: Represents the path of the audio file.
# - _'augmented'_: True if its an augmented file.
# 
# The DataFrame is saved to './All_Files_' + name_aug + '.pkl'.
# 

# In[2]:


def create_speaker_DF(path, path_aug, name_aug, test_only):
    df = pd.DataFrame(columns=['dialect', 'speaker', 'file_name', 'length', 'file_path', 'augmented'])

    df = sub(df, path, 'False', '')
    if (name_aug != ''):
        df = sub(df, path_aug, 'True', name_aug)
    
    if (test_only):
        df.to_pickle('./All_Files_test.pkl')
    else:
        df.to_pickle('./All_Files_.pkl')
    
    return df


# In[ ]:


def sub(df, path, aug, name_aug):
    all_speaker = []
    all_speaker_name = []
    
    all_dialects = glob(path + '\\*', recursive = True)
    for dialect in all_dialects:
        all_speaker = np.concatenate((all_speaker, glob(dialect + '\\*', recursive = True)), axis=None)
        all_speaker_name.append([f.name for f in os.scandir(dialect) if f.is_dir()])

    audios = []
    for path in all_speaker:
        audios.extend(tf.io.gfile.glob(path + '\\*.wav'))
    
    for audio in audios:
        split = audio.split('\\')
    
        audio_segment = AudioSegment.from_file(audio, "wav") 
        duration = len(audio_segment)
    
        speaker = split[-2]
        list_row = [split[-3], speaker, Path(audio).name, duration, audio, aug]
        df.loc[len(df)] = list_row
    return df


# ## select random speaker for test and for val for all classes
# ### Parameters:
# 
# - **df**:  
#   DataFrame created by the 'create_speaker_DF' function.
# 
# ### Returns:
# 
# - **speaker_test_val**:  
#   A list of dialects, where each dialect is represented by a list. Each sublist contains approximately ⌈(#speakers_in_class / 10)⌉ * 2 random speakers. The first half of the sublist is designated for testing, while the second half is for validation.
# 

# In[3]:


def choose(df):
    
    dialects = df['dialect'].unique().tolist()
    speaker_all = [[] for _ in range(0, len(dialects))]
    speaker_test_val = [[] for _ in range(0, len(dialects))]

    for i in range(0, len(dialects)):
        speaker_all[i] = df[(df['dialect'] == dialects[i]) & (df['augmented'] == 'False')]['speaker'].unique().tolist()
        print('Number of Speaker for', dialects[i], ':', len(speaker_all[i]))
    
    for i in range(0, len(speaker_all)):
        num_val_test = math.ceil(len(speaker_all[i])/10)
        speaker_test_val[i] = sample(speaker_all[i], k=num_val_test*2)
        print(dialects[i], 'Test', speaker_test_val[i][0:num_val_test], 'Val', speaker_test_val[i][num_val_test:num_val_test*2])
        
    return speaker_test_val


# ## select random speaker for val for all classes
# ### Parameters:
# 
# - **df**:  
#   DataFrame created by the 'create_speaker_DF' function.
# 
# ### Returns:
# 
# - **speaker_test_val**:  
#   A list contains approximately ⌈(#speakers_in_class / 10)⌉ random speakers.
# 

# In[ ]:


def choose_train_only(df):
    
    dialects = df['dialect'].unique().tolist()
    speaker_all = [[] for _ in range(0, len(dialects))]
    speaker_val = []

    for i in range(0, len(dialects)):
        speaker_all[i] = df[(df['dialect'] == dialects[i]) & (df['augmented'] == 'False')]['speaker'].unique().tolist()
        print('Number of Speaker for', dialects[i], ':', len(speaker_all[i]))
    
    for i in range(0, len(speaker_all)):
        num_val = math.ceil(len(speaker_all[i])/10)
        speaker_val.extend(sample(speaker_all[i], k=num_val))
        
    return speaker_val


# In[ ]:




