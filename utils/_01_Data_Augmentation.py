#!/usr/bin/env python
# coding: utf-8

# In[1]:


from parselmouth.praat import run_file, call
from scipy.fft import fft, ifft
from itertools import repeat
import tensorflow as tf
import soundfile as sf
import multiprocessing
from glob import glob
import pandas as pd
import numpy as np
import parselmouth
import pathlib
import librosa
import random
import math
import os
import re


# In[ ]:


#path of own praat scripts
praat_changeformants = "...\\changeformants_own.praat"


# ## create augmented files
# ### Parameters:
# 
# - **name_aug**:  
#   Name of the augmentation. Should be one of the keywords.
# 
# - **data_path**:  
#   The data path of the original audios.
# 
# - **data_path_aug**:  
#   The data path for the augmented audios.
# 
# - **aug_perc**:  
#   Percentage for augmentation per file.
# 
# - **aug_num**:  
#   Number of augmented files per original file.
# 
# - **aug_len**:  
#   Segment length for each augmentation.
# 
# - **audio_length**:  
#   Length of one segment from the whole audio.
# 
# - **fixed_speaker**:  
#   Speaker for Test and Vaidation. So that None of the speaker is used for Training.
#   Only important for 'speaker_insertion' and 'frequency_insertion'.
# 

# In[2]:


def augment(name_aug, data_path, data_path_aug, aug_perc, aug_num, aug_len, audio_length, fixed_speaker):
    if aug_perc > 1.0:
        raise Warning("Augmentation percentage should be maximal 1.0!")
    if aug_len > audio_length:
        raise Warning("Augmentation length shoud be shorter than audio length!")
    
    
    if not os.path.exists(data_path_aug):
        os.makedirs(data_path_aug)
    
    all_dialects = glob(data_path + '\\*', recursive = True)
                
    if ('speaker_insertion' in name_aug):
        speaker_all = [[] for _ in range(0, len(all_dialects))]
        
        for i in range(0, len(all_dialects)):
            dialect = all_dialects[i]
            speaker_all[i] = np.concatenate((speaker_all[i], glob(dialect + '\\*', recursive = True)), axis=None)
            
        audios = [[] for _ in range(0, len(all_dialects))]
        for i in range (0, len(all_dialects)):
            for path in speaker_all[i]:
                audios[i].extend(tf.io.gfile.glob(path + '\\*.wav'))
                new_path = data_path_aug + '\\' + path.split('\\')[-2] + '\\' + path.split('\\')[-1]
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
        
        for i in range (0, len(all_dialects)):
            for audio in audios[i]:
                y, sr = librosa.load(audio, sr=16000, res_type='soxr_vhq')
                for j in range (0, aug_num):
                    y_new = y.copy()
                    times_total = len(y)//int(audio_length*16000)
                    for num in range(0, times_total):
                        times = int((aug_perc*audio_length)/aug_len)
                        intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                        for interval_start in intervals_start:
                            interval_start_index = int(interval_start + (num * audio_length * 16000))
                            end_index = int(interval_start_index + (aug_len * 16000))
                            audio_other = random.choice([audio for audio in audios[i] if not any(term in audio for sublist in fixed_speaker for term in sublist)])
                            y_other, sr_other = librosa.load(audio_other, sr=16000, res_type='soxr_vhq')
                            interval_start_index_other = random.randint(0, len(y_other)-(aug_len * 16000))
                            end_index_other = int(interval_start_index_other + (aug_len * 16000))
                            ad = y_other[interval_start_index_other:end_index_other]
                            y_new[interval_start_index:end_index] = ad
                    new_path = data_path_aug + '\\' + audio.split('\\')[-3] + '\\' + audio.split('\\')[-2] + '\\' + 'aug' + str(j) + '_' + audio.split('\\')[-1]
                    sf.write(new_path, y_new, 16000)
                    
    elif ('frequency_insertion' in name_aug):
        speaker_all = [[] for _ in range(0, len(all_dialects))]
        
        for i in range(0, len(all_dialects)):
            dialect = all_dialects[i]
            speaker_all[i] = np.concatenate((speaker_all[i], glob(dialect + '\\*', recursive = True)), axis=None)
            
        audios = [[] for _ in range(0, len(all_dialects))]
        for i in range (0, len(all_dialects)):
            for path in speaker_all[i]:
                audios[i].extend(tf.io.gfile.glob(path + '\\*.wav'))
                new_path = data_path_aug + '\\' + path.split('\\')[-2] + '\\' + path.split('\\')[-1]
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
        
        for i in range (0, len(all_dialects)):
            for audio in audios[i]:
                y, sr = librosa.load(audio, sr=16000, res_type='soxr_vhq')
                for j in range (0, aug_num):
                    y_new = y.copy()
                    times_total = len(y)//int(audio_length*16000)
                    for num in range(0, times_total):
                        interval_start_index = int(num * audio_length * 16000)
                        ad = y[interval_start_index:int(interval_start_index+(16000*audio_length))]
                        audio_other = random.choice([audio for audio in audios[i] if not any(term in audio for sublist in fixed_speaker for term in sublist)])
                        y_other, sr_other = librosa.load(audio_other, sr=16000, res_type='soxr_vhq')
                        interval_start_index_other = random.randint(0, len(y_other)-(audio_length * 16000))
                        end_index_other = int(interval_start_index_other + (audio_length * 16000))
                        ad_other = y_other[interval_start_index_other:end_index_other]
                        bandwidth = random.randint(100, 2500)
                        times = random.randint(1, 3)
                        intervals_start = generate_intervals(bandwidth, times, 8000)
                        fft_result = fft(ad)
                        freqs = np.fft.fftfreq(len(ad), 1/16000)
                        fft_result_other = fft(ad_other)
                        freqs_other = np.fft.fftfreq(len(ad_other), 1/16000)
                        for interval_start in intervals_start:
                            band_mask = (freqs >= interval_start) & (freqs < interval_start+bandwidth)
                            band_mask_other = (freqs_other >= interval_start) & (freqs_other < interval_start+bandwidth)
                            fft_result[band_mask] = fft_result_other[band_mask_other]
                        swapped_signal = np.real(np.fft.ifft(fft_result))
                        y_new[interval_start_index:int(interval_start_index+(16000*audio_length))] = swapped_signal 
                    new_path = data_path_aug + '\\' + audio.split('\\')[-3] + '\\' + audio.split('\\')[-2] + '\\' + 'aug' + str(j) + '_' + audio.split('\\')[-1]
                    sf.write(new_path, y_new, 16000)    
                
    else:
        all_speaker = []
    
        for dialect in all_dialects:
            all_speaker = np.concatenate((all_speaker, glob(dialect + '\\*', recursive = True)), axis=None)
        
        audios = []       
            
        for path in all_speaker:
            audios.extend(tf.io.gfile.glob(path + '\\*.wav'))
            new_path = data_path_aug + '\\' + path.split('\\')[-2] + '\\' + path.split('\\')[-1]
            if not os.path.exists(new_path):
                os.makedirs(new_path)
        
        #for audio in audios:
        #    augmentation(audio, name_aug, data_path_aug, aug_perc, aug_num, aug_len, audio_length)
            
        pool = multiprocessing.Pool(5)
        pool.starmap(augmentation, zip(np.array(audios), repeat(name_aug), repeat(data_path_aug), repeat(aug_perc),
                                       repeat(aug_num), repeat(aug_len), repeat(audio_length)))
                     


# In[1]:


def augmentation(audio, name_aug, data_path_aug, aug_perc, aug_num, aug_len, audio_length):
    y, sr = librosa.load(audio, sr=16000, res_type='soxr_vhq')
    
    if ('background_noise' in name_aug or 'best' in name_aug):
        noise_dir = "D:\\musan\\noise"
        noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))

    for aug_num_cnt in range (0, aug_num):
        y_new = y.copy()
        
        if('best' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            offset = 0
            for i in range(0, times_total):
                times = int((0.5*audio_length)/0.3)
                intervals_start = generate_intervals(0.3*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    end_index = int(interval_start_index + (0.3 * 16000))
                    interval_start_index -= offset
                    end_index -= offset
                    y_new = np.concatenate((y_new[:interval_start_index], y_new[end_index:]), axis=None)
                    offset += (end_index - interval_start_index)               
            new_path = data_path_aug + '\\' + audio.split('\\')[-3] + '\\' + audio.split('\\')[-2] + '\\' + audio.split('\\')[-1]
            sf.write(new_path, y_new, 16000)
            times_total = len(y_new)//int(audio_length*16000)
            sound = parselmouth.Sound(new_path)
            for i in range(0, times_total):
                interval_start_index = int(i * audio_length * 16000)
                bandwidth = random.randint(100, 2500)
                times = random.randint(1, 3)
                intervals_start = generate_intervals(bandwidth, times, 8000)
                interval_start_sec = int(i*audio_length)
                part = call(sound, "Extract part", interval_start_sec, interval_start_sec+audio_length, 'rectangular', 1.0, 'no')
                for interval_start in intervals_start:
                    part = call(part, "Filter (stop Hann band)", interval_start, interval_start+bandwidth, 100)
                y_new[interval_start_index:int(interval_start_index+(16000*audio_length))] = part.values.flatten()
            os.remove(new_path) 
        

        #Main Methods

        elif ('change_formants' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            sound = parselmouth.Sound(audio)
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    f1 = random.randint(220, 780)
                    f2 = random.randint(1200, 2000)
                    f3 = random.randint(2200, 3000)
                    interval_start_sec = (interval_start/16000)+(i*audio_length)
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    part = call(sound, "Extract part", interval_start_sec, interval_start_sec+aug_len, 'rectangular', 1.0, 'no')
                    new_sound = run_file(part, praat_changeformants, f1, f2, f3, 0, 0, 5000, 'yes', 'yes', 'no')
                    start_index = int(interval_start_index)
                    end_index = int(interval_start_index + (aug_len * 16000))
                    y_new[start_index:end_index] = new_sound[0].values.flatten()
        
        elif ('shifting_pitch' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            sound = parselmouth.Sound(audio)
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    interval_start_sec = (interval_start/16000)+(i*audio_length)
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    new_pitch = random.randint(90, 160)
                    part = call(sound, "Extract part", interval_start_sec, interval_start_sec+aug_len, 'rectangular', 1.0, 'no')
                    new_sound = call(part, "Change gender", 80, 170, 1.0, new_pitch, 1.0, 1.0)
                    start_index = int(interval_start_index)
                    end_index = int(interval_start_index + (aug_len * 16000))
                    y_new[start_index:end_index] = new_sound.values.flatten()
                    
        elif ('segment_removal' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            offset = 0
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    end_index = int(interval_start_index + (aug_len * 16000))
                    interval_start_index -= offset
                    end_index -= offset
                    y_new = np.concatenate((y_new[:interval_start_index], y_new[end_index:]), axis=None)
                    offset += (end_index - interval_start_index)
                    
        #https://jonathanbgn.com/2021/08/30/audio-augmentation.html
        #http://www.openslr.org/17/
        elif ('background_noise' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    end_index = int(interval_start_index + (aug_len * 16000))
                    ad = y[interval_start_index:end_index]
                    
                    while True:
                        random_noise_file = str(random.choice(noise_files_list))
                        noise, sr = librosa.load(random_noise_file, sr=16000, res_type='soxr_vhq')
                        librosa.to_mono(noise)
                        if len(noise) >= aug_len*16000:
                            break
                    noise_start_index = random.randint(0, len(noise)-(aug_len*16000))
                    noise = noise[noise_start_index:noise_start_index + int(aug_len * 16000)]
                    snr_db = random.randint(0, 30)            
    
                    # Calculate power of the signals
                    signal_power = np.sum(ad**2) / len(ad)
                    noise_power = np.sum(noise**2) / len(noise)
                    # Calculate the scale factor for the noise to achieve the target SNR
                    target_snr_linear = 10**(snr_db / 10)
                    scale_factor = np.sqrt(signal_power / (target_snr_linear * noise_power))   
                    # Scale the noise to achieve the target SNR
                    scaled_noise = noise * scale_factor
                    new_sound = ad + scaled_noise
                    
                    y_new[interval_start_index:end_index] = new_sound
                    
        elif ('segment_swap' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                times = times-(times%2)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                random.shuffle(intervals_start)
                midpoint = len(intervals_start) // 2
                first_half = intervals_start[:midpoint]
                second_half = intervals_start[midpoint:]
                for interval_start_first, interval_start_second in zip(first_half, second_half):
                    interval_start_first_index = int(interval_start_first + (i * audio_length * 16000))
                    end_first_index = int(interval_start_first_index + (aug_len * 16000))
                    ad_first = y[interval_start_first_index:end_first_index]
                    interval_start_second_index = int(interval_start_second + (i * audio_length * 16000))
                    end_second_index = int(interval_start_second_index + (aug_len * 16000))
                    ad_second = y[interval_start_second_index:end_second_index]
                    y_new[interval_start_second_index:end_second_index], y_new[interval_start_first_index:end_first_index] = ad_first, ad_second
                    
          
        #Sub Methods
        elif ('volume_confusion' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    end_index = int(interval_start_index + (aug_len * 16000))
                    ad = y[interval_start_index:end_index]
                    rate = random.uniform(0.2, 0.8)
                    new_sound = ad / (max(ad)+rate)
                    y_new[interval_start_index:end_index] = new_sound
                    
        elif ('time_reversing' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    end_index = int(interval_start_index + (aug_len * 16000))
                    ad = y[interval_start_index:end_index]
                    new_sound = np.flipud(ad)
                    y_new[interval_start_index:end_index] = new_sound
                    
        elif ('time_stretching' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            sound = parselmouth.Sound(audio)
            offset = 0
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    interval_start_sec = (interval_start/16000)+(i*audio_length)
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    end_index = int(interval_start_index + (aug_len * 16000))
                    rate = random.uniform(0.8, 1.2)
                    part = call(sound, "Extract part", interval_start_sec, interval_start_sec+aug_len, 'rectangular', 1.0, 'no')
                    new_sound = call(part, "Lengthen (overlap-add)", 75, 300, rate)
                    interval_start_index += offset
                    end_index += offset
                    y_new = np.concatenate((y_new[:interval_start_index], new_sound.values.flatten(), y_new[end_index:]), axis=None)
                    offset += len(new_sound.values.flatten())-int(aug_len*16000)
                    
        elif ('speed_confusion' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            sound = parselmouth.Sound(audio)
            offset = 0
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    interval_start_sec = (interval_start/16000)+(i*audio_length)
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    end_index = int(interval_start_index + (aug_len * 16000))
                    rate = random.uniform(0.8, 1.2)
                    new_sampling_rate = 16000*rate
                    part = call(sound, "Extract part", interval_start_sec, interval_start_sec+aug_len, 'rectangular', 1.0, 'no')
                    new_sound = call(part, "Resample", new_sampling_rate, 50)
                    interval_start_index += offset
                    end_index += offset
                    y_new = np.concatenate((y_new[:interval_start_index], new_sound.values.flatten(), y_new[end_index:]), axis=None)
                    offset += len(new_sound.values.flatten())-int(aug_len*16000)
                    
        elif ('time_masking' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            for i in range(0, times_total):
                times = int((aug_perc*audio_length)/aug_len)
                intervals_start = generate_intervals(aug_len*16000, times, int(audio_length*16000))
                for interval_start in intervals_start:
                    interval_start_index = int(interval_start + (i * audio_length * 16000))
                    end_index = int(interval_start_index + (aug_len * 16000))
                    interval_length = end_index - interval_start_index
                    new_sound = [0] * interval_length
                    y_new[interval_start_index:end_index] = new_sound
                    
        elif ('frequency_masking' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            sound = parselmouth.Sound(audio)
            for i in range(0, times_total):
                interval_start_index = int(i * audio_length * 16000)
                bandwidth = random.randint(100, 2500)
                times = random.randint(1, 3)
                intervals_start = generate_intervals(bandwidth, times, 8000)
                interval_start_sec = int(i*audio_length)
                part = call(sound, "Extract part", interval_start_sec, interval_start_sec+audio_length, 'rectangular', 1.0, 'no')
                for interval_start in intervals_start:
                    part = call(part, "Filter (stop Hann band)", interval_start, interval_start+bandwidth, 100)
                y_new[interval_start_index:int(interval_start_index+(16000*audio_length))] = part.values.flatten() #masked_audio_signal
                
                  
        elif ('frequency_swap' in name_aug):
            times_total = len(y)//int(audio_length*16000)
            for i in range(0, times_total):
                interval_start_index = int(i * audio_length * 16000)
                ad = y[interval_start_index:int(interval_start_index+(16000*audio_length))]
                times = 2
                bandwidth = random.randint(100, 2500)
                intervals_start = generate_intervals(bandwidth, times, 8000)
                random.shuffle(intervals_start)
                midpoint = int(len(intervals_start) // 2)
                first_half = intervals_start[:midpoint]
                second_half = intervals_start[midpoint:]
                fft_result = fft(ad)
                freqs = np.fft.fftfreq(len(ad), 1/16000)
                for interval_start_first, interval_start_second in zip(first_half, second_half):
                    band1_mask = (freqs >= interval_start_first) & (freqs < interval_start_first+bandwidth)
                    band2_mask = (freqs >= interval_start_second) & (freqs < interval_start_second+bandwidth)
                    fft_signal_band1 = fft_result[band1_mask].copy()
                    fft_signal_band2 = fft_result[band2_mask].copy()
                    fft_result[band2_mask] = fft_signal_band1
                    fft_result[band1_mask] = fft_signal_band2
                swapped_signal = np.real(np.fft.ifft(fft_result))
                y_new[interval_start_index:int(interval_start_index+(16000*audio_length))] = swapped_signal    


        new_path = data_path_aug + '\\' + audio.split('\\')[-3] + '\\' + audio.split('\\')[-2] + '\\' + 'aug' + str(aug_num_cnt) + '_' + audio.split('\\')[-1]
        sf.write(new_path, y_new, 16000)


# In[5]:


def generate_intervals(length, times, total_len):
    result = []
    # Ensure there's enough space for intervals
    if times * length > total_len:
        raise ValueError("Not enough space for intervals in the given range.")
    
    # Generate 'times' random interval starting points
    end = 0
    for i in range(times):
        old_end = end
        start_tmp = random.randint(0, total_len - ((times-i) * length))
        start = start_tmp + old_end
        end = start + length
        result.append(start)
        # Adjust starting point for the next interval to avoid overlap
        total_len -= start_tmp + length
    return result


# In[ ]:




