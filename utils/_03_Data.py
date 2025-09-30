#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import librosa
import random
import time


# In[ ]:


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ## calculates embeddings and saves the DF
# ### Parameters:
# 
# - **model_path**:  
#   The path to the model used for extracting embeddings.
# 
# - **length**:  
#   The desired length of each audio segment in seconds.
# 
# - **batch_size**:  
#   The batch size used for processing audio segments.
#   
# - **name_aug**:  
#   A string identifier used for distinguishing augmented data.
# 
# - **max_length_speaker**:  
#   The maximum length (in seconds) for each place (speaker) in the dataset.
# 
# - **max_length_dialect**:  
#   The maximum length (in seconds) for each dialect in the dataset.
#   
# - **test_only**:  
#   True if the run is only for making predictions.
# 
# ### Returns:
# 
# - **df_learn**:  
#   DataFrame containing the following columns:
#   - _'dialect'_: Represents the dialect class of the audio.
#   - _'file_name'_: Represents the name of the audio file.
#   - _'trillsson'_: Contains the embeddings calculated by the pre-trained model.
#   - _'file_path'_: Represents the path of the audio file.
#   - _'speaker'_: Represents the speaker associated with the audio.
#   - _'samples_begin'_: Indicates the starting sample index of each segment.
#   - _'samples_end'_: Indicates the ending sample index of each segment.
#   
#   The DataFrame is saved as './Data_.pkl' and './Data_.csv'.
# 

# In[3]:


def create_data(model_path, length, batch_size, name_aug, max_length_speaker, max_length_dialect, test_only):
    timeCountTotal = 0.0
    startTotal = time.time()
    if (test_only):
        df = pd.read_pickle('./All_Files_test.pkl')
    else:
        df = pd.read_pickle('./All_Files_.pkl')
    
    model = hub.load(model_path)

    model.trainable = False

    df_learn = pd.DataFrame(columns=['dialect', 'file_name', 'trillsson', 'file_path', 'speaker', 'samples_begin', 'samples_end'])
    audio_samples = []
    
    if (max_length_speaker is not None):
        if (max_length_speaker % length != 0):
            raise ValueError("max_length_speaker is not an multiple of audio_length.")
        else:
            max_times_speaker = int(max_length_speaker/length)
    if (max_length_dialect is not None):
        if (max_length_dialect % length != 0):
            raise ValueError("max_length_dialect is not an multiple of audio_length.")
        else:
            max_times_dialect = int(max_length_dialect/length)
        
    length = int(length*16000)
    
    # cut Audios in length long Segments and save it in df_learn
    for index, row in df.iterrows():
        file_path = None
        if (name_aug != ''):
            if row['length'] >= 0 and row['augmented']=='True':
                file_path = row['file_path']
                name = row['file_name']
                speaker = row['speaker']
                class_label = row['dialect']
        else:
            if row['length'] >= 0 and row['augmented']=='False':
                file_path = row['file_path']
                name = row['file_name']
                speaker = row['speaker']
                class_label = row['dialect']
        
        if (file_path is not None):
            audio, sr = librosa.load(file_path, sr=16000, dtype=np.float32)
    
            times = len(audio)//(length)
            for i in range(0, times):
                ad = audio[i*length:((i+1)*length)]
                audio_samples.append(ad)
                list_row = [class_label, name, [], file_path, speaker, i*length, ((i+1)*length)-1]
                df_learn.loc[len(df_learn)] = list_row 
                
    if max_length_speaker is not None:
        df_learn, audio_samples = filter_excess_names(df_learn, audio_samples, max_times_speaker, 'speaker')
    if max_length_dialect is not None:
        df_learn, audio_samples = filter_excess_names(df_learn, audio_samples, max_times_dialect, 'dialect')  
    
    # shuffle indices of df_learn, so that embedding calculation is not sequential
    df_size = df_learn.shape[0]
    indices = list(range(df_size))
    random.shuffle(indices)
    embeddings_list = [None] * len(df_learn)

    i = 0
    timeCount = 0.0
    print('total to calcuate: ' + str(df_size))
    while i < df_size:
        indices_batch = indices[i:min(i + batch_size, df_size)]
        audios = [audio_samples[ind] for ind in indices_batch]
            
        # calculate actual embeddings
        start = time.time()
        embeddings = model(audios)['embedding']
        end = time.time()
        timeCount += end - start
        embeddings_list_tmp = embeddings.numpy().tolist()
        
        for ind, emb in zip(indices_batch, embeddings_list_tmp):
            embeddings_list[ind] = emb
        i += batch_size
        print('actual calculated: ' + str(i))

    df_learn['trillsson'] = embeddings_list
    print('Time for extracting Features:', timeCount)
    
    # save embeddings as pkl and csv
    if (test_only):
        df_learn.to_pickle('./Data_test.pkl')
        df_learn.to_csv('./Data_test.csv',  sep=';')
    elif (name_aug != ''):
        df_learn.to_pickle('./Data_' + name_aug + '_aug.pkl')
        df_learn.to_csv('./Data_' + name_aug + '_aug.csv',  sep=';')
    else:
        df_learn.to_pickle('./Data_.pkl')
        df_learn.to_csv('./Data_.csv',  sep=';')
        
    endTotal = time.time()
    timeCountTotal += endTotal - startTotal
    print('Time for extracting Features in total:', timeCountTotal)
    
    return df_learn


# In[ ]:


def filter_excess_names(df_learn, array, max_occurrences, row):
    
    original_state = random.getstate()
    random.seed(42)
    df = pd.read_pickle('./Data_.pkl')

    name_counts = df_learn[row].value_counts()
    name_counts_orig = df[row].value_counts()
    excess_names = name_counts[(name_counts+name_counts_orig) > max_occurrences].index.tolist()
    
    for name in excess_names:
        excess_occurrences = min(name_counts[name], (name_counts[name] + name_counts_orig[name]) - max_occurrences)
        if excess_occurrences > 0:
            excess_indices = df_learn[(df_learn[row] == name) & (df_learn['file_name'].str.startswith('aug'))].sample(n=excess_occurrences).index  
            df_learn = df_learn.drop(excess_indices).reset_index(drop=True)
            mask = np.ones(len(array), dtype=bool)
            mask[excess_indices] = False
            array = [array[i] for i in range(len(array)) if mask[i]]
        
    random.setstate(original_state)       
    return df_learn, array


# ## generates boxplots for speaker counts and bar plots for class counts
# ### Parameters:
# 
# - **name_aug**:  
#   A name identifier used for loading the augmented DataFrame.
# 

# In[4]:


def getAbsoluteCounts(name_aug):
    
    df_learn = pd.read_pickle('./Data_.pkl')
    dialect_counts = df_learn['dialect'].value_counts()
    name_counts = df_learn['file_name'].value_counts()
    
    name_counts_df = name_counts.reset_index()
    name_counts_df.columns = ['file_name', 'count']
    name_counts_df.to_csv('speaker_counts.csv', index=False)
    
    plotAbsoluteCounts('', name_counts, dialect_counts)
    
    if (name_aug is not None):
        df_learn_aug = pd.read_pickle('./Data_' + name_aug + '_aug.pkl')
        dialect_counts_aug = df_learn_aug['dialect'].value_counts()
        name_counts_aug = df_learn_aug['file_name'].value_counts()
        
        name_counts_df_aug = name_counts_aug.reset_index()
        name_counts_df_aug.columns = ['file_name', 'count']
        name_counts_df_aug.to_csv('speaker_counts_aug.csv', index=False)
        
        combined_dialect_counts = dialect_counts.add(dialect_counts_aug, fill_value=0)
        
        plotAbsoluteCounts('_' + name_aug + '_aug', name_counts_aug, dialect_counts_aug)
        plotAbsoluteCounts('_' + name_aug + '_aug_combined', None, combined_dialect_counts)
    


# In[ ]:


def plotAbsoluteCounts(imageName, name_counts, dialect_counts):    
    if name_counts is not None:
        plt.boxplot(name_counts.values)
        #whisker_high = name_counts.median() + 1.5 * (name_counts.quantile(0.75) - name_counts.quantile(0.25))
        #whisker_low = name_counts.median() - 1.5 * (name_counts.quantile(0.75) - name_counts.quantile(0.25))
        whisker_high = name_counts.quantile(0.75) + 1.5 * (name_counts.quantile(0.75) - name_counts.quantile(0.25))
        whisker_low = name_counts.quantile(0.25) - 1.5 * (name_counts.quantile(0.75) - name_counts.quantile(0.25))
        print('lower whisker: ' + str(whisker_low) + "; middle: " + str(name_counts.median()) + '; upper whisker: ' + str(whisker_high))
        outlier_indices = (name_counts > whisker_high) | (name_counts < whisker_low)
        cnt = 1
        
        for i in range(len(name_counts)):
            if outlier_indices[i]:
                count = name_counts[i]
                if cnt%2 == 1:
                    plt.annotate(
                        name_counts.index[i],
                        (1, count),
                        textcoords="data",
                        xytext=(1+0.02, count),
                        ha='left',
                        va='center',
                        fontsize=5,
                    )
                else:
                    plt.annotate(
                        name_counts.index[i],
                        (1, count),
                        textcoords="data",
                        xytext=(1-0.02, count),
                        ha='right',
                        va='center',
                        fontsize=5,
                    )
                cnt += 1
        plt.ylabel("Counts")
        plt.title("Speaker #Segments")
        ax = plt.gca()
        min_y, max_y = ax.get_ylim()
        for y in range(0, int(max_y), 25):
            ax.axhline(y=y, linestyle='--', color='gray', linewidth=0.5)
        plt.savefig('speaker_counts' + imageName + '.png', bbox_inches='tight', dpi=600)
        plt.close()
    
    ax = dialect_counts.plot(kind='bar')
    plt.title("Classes #Segments")
    plt.grid(axis='y')
    plt.ylabel("Count")
    plt.savefig('classes_counts' + imageName + '.png', bbox_inches='tight', dpi=600)
    plt.close()


# In[ ]:




