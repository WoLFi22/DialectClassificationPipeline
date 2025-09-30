#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.layers import Dense, Dropout, Activation, LeakyReLU, Conv1D, GlobalAveragePooling1D, Flatten, MaxPooling1D,  BatchNormalization 
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight, shuffle
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.dummy import DummyClassifier
from keras.utils import to_categorical
from keras.regularizers import l1, l2
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import backend as K
from random import sample
import tensorflow as tf
import keras.metrics
import pandas as pd
import numpy as np

import tensorflow_addons as tfa


# ## GPU

# In[2]:


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# ## divide intro Train, Validation and Test
# ### Parameters:
# 
# - **speaker_test_val**:  
#   A list of dialects with each sublist containing a list of speakers. The first half of each sublist is designated for testing, and the second half is for validation.
# 
# - **df**:  
#   The original DataFrame containing audio data.
# 
# - **df_aug**:  
#   The augmented DataFrame, if available.
# 
# - **name_aug**:  
#   A name identifier used for loading augmented data or specifying augmentation settings.
# 
# ### Returns:
# 
# - **x_train**:  
#   Numpy array of features for the training set.
# 
# - **y_train**:  
#   List of dialect labels for the training set.
# 
# - **x_test**:  
#   Numpy array of features for the testing set.
# 
# - **y_test**:  
#   List of dialect labels for the testing set.
# 
# - **x_val**:  
#   Numpy array of features for the validation set.
# 
# - **y_val**:  
#   List of dialect labels for the validation set.
# 
# - **y_test_names**:  
#   List of names for the testing set.
# 
# - **y_test_speaker**:  
#   List of speakers for the testing set.
# 
# - **y_test_segment_begin**:  
#   List of starting sample indices for segments in the testing set.
# 
# - **y_test_segment_end**:  
#   List of ending sample indices for segments in the testing set.
# 

# In[3]:


def train_test(speaker_test_val, df, df_aug, name_aug):
    
    train_speaker = []
    test_speaker = []
    val_speaker = []
    
    # get data seperated by train, val and test
    for i in range(0, len(speaker_test_val)):
        num_val_test = len(speaker_test_val[i])//2
        test_speaker = np.concatenate((test_speaker, speaker_test_val[i][0:num_val_test]), axis=None)
        val_speaker = np.concatenate((val_speaker, speaker_test_val[i][num_val_test:num_val_test*2]), axis=None)
    test = df[df['speaker'].isin(test_speaker)]
    val = df[df['speaker'].isin(val_speaker)]
    train_speaker = np.concatenate((test_speaker, val_speaker))
    train = df[~df['speaker'].isin(train_speaker)]
    
    # choose augmented speaker Audios for train with speaker from speaker only already in train
    if (df_aug is not None):
        if ('Speaker_random' in name_aug):
            elements = train['speaker'].unique().tolist()
            dialects = train['dialect'].unique().tolist()
            train_aug = df_aug[df_aug['speaker'].apply(lambda x: set(x).issubset(set(elements)))]
            to_augment_number = int(len(df.index)/100*rate/len(dialects))

            train_aug_res = []
            for dialect in dialects:
                train_aug_tmp = train_aug[train_aug['dialect'] == dialect].sample(n=to_augment_number)
                train_aug_res = pd.concat([train_aug_res, train_aug_tmp], ignore_index=True)
            train_aug = train_aug_res
            train = pd.concat([train, train_aug], ignore_index=True)
        elif (name_aug == ''):
            train_aug = df_aug[~df_aug['speaker'].isin(train_speaker)]
            train = train_aug
        else:
            train_aug = df_aug[~df_aug['speaker'].isin(train_speaker)]
            train = pd.concat([train, train_aug], ignore_index=True)
    
    x_train = np.asarray(train['trillsson'].tolist())
    y_train = train.dialect.tolist()
    x_val = np.asarray(val['trillsson'].tolist())
    y_val = val.dialect.tolist()
    x_test = np.asarray(test['trillsson'].tolist())
    y_test = test.dialect.tolist()
    y_test_names = test.file_name.tolist()
    y_test_speaker = test.speaker.tolist()
    y_test_segment_begin = test.samples_begin.tolist()
    y_test_segment_end = test.samples_end.tolist()
   
    return x_train, y_train, x_test, y_test, x_val, y_val, y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end


# ## Divide into Train and Validation
# ### Parameters:
# 
# - **speaker_val**:  
#   A list containing a list of speakers.
# 
# - **df**:  
#   The original DataFrame containing audio data.
# 
# - **df_aug**:  
#   The augmented DataFrame, if available.
# 
# - **name_aug**:  
#   A name identifier used for loading augmented data or specifying augmentation settings.
# 
# ### Returns:
# 
# - **x_train**:  
#   Numpy array of features for the training set.
# 
# - **y_train**:  
#   List of dialect labels for the training set.
# 
# - **x_val**:  
#   Numpy array of features for the validation set.
# 
# - **y_val**:  
#   List of dialect labels for the validation set.
# 

# In[ ]:


def train_val(speaker_val, df, df_aug, name_aug):

    val = df[df['speaker'].isin(speaker_val)]
    train = df[~df['speaker'].isin(speaker_val)]
    
    # choose augmented speaker Audios for train with speaker from speaker only already in train
    if (df_aug is not None):
        train_aug = df_aug[~df_aug['speaker'].isin(speaker_val)]
        train = pd.concat([train, train_aug], ignore_index=True)
    
    x_train = np.asarray(train['trillsson'].tolist())
    y_train = train.dialect.tolist()
    x_val = np.asarray(val['trillsson'].tolist())
    y_val = val.dialect.tolist()
   
    return x_train, y_train, x_val, y_val


# ## get Features
# ### Parameters:
# 
# - **x_train**:  
#   Numpy array of features for the training set.
# 
# - **y_train**:  
#   List of dialect labels for the training set.
# 
# - **x_test**:  
#   Numpy array of features for the testing set.
# 
# - **y_test**:  
#   List of dialect labels for the testing set.
# 
# - **x_val**:  
#   Numpy array of features for the validation set.
# 
# - **y_val**:  
#   List of dialect labels for the validation set.
# 
# - **y_test_names**:  
#   List of names for the testing set.
# 
# - **y_test_speaker**:  
#   List of speakers for the testing set.
# 
# - **y_test_segment_begin**:  
#   List of starting sample indices for segments in the testing set.
# 
# - **y_test_segment_end**:  
#   List of ending sample indices for segments in the testing set.
# 
# - **df_learn**:  
#   DataFrame containing audio data used for encoding labels.
# 
# ### Returns:
# 
# - **y_train**:  
#   List of shuffled dialect labels for the training set.
# 
# - **x_train**:  
#   Numpy array of shuffled features for the training set.
# 
# - **y_val**:  
#   List of shuffled dialect labels for the validation set.
# 
# - **x_val**:  
#   Numpy array of shuffled features for the validation set.
# 
# - **y_test**:  
#   List of shuffled dialect labels for the testing set.
# 
# - **x_test**:  
#   Numpy array of shuffled features for the testing set.
# 
# - **yy_train**:  
#   Categorical labels for the training set.
# 
# - **yy_test**:  
#   Categorical labels for the testing set.
# 
# - **yy_val**:  
#   Categorical labels for the validation set.
# 
# - **y_test_names**:  
#   List of names for the testing set.
# 
# - **y_test_speaker**:  
#   List of speakers for the testing set.
# 
# - **y_test_segment_begin**:  
#   List of starting sample indices for segments in the testing set.
# 
# - **y_test_segment_end**:  
#   List of ending sample indices for segments in the testing set.
# 
# - **label_mapping**:  
#   Mapping of original dialect labels to encoded categorical labels.
# 

# In[4]:


def features(x_train, y_train, x_test, y_test, x_val, y_val, y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end, df_learn):
    y_train, x_train = shuffle(y_train, x_train)
    y_val, x_val = shuffle(y_val, x_val)
    y_test, x_test, y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end = shuffle(y_test, x_test, y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end)

    # Encode the classification labels
    le = LabelEncoder()
    le.fit(sorted(df_learn['dialect'].unique().tolist()))
    yy_train = to_categorical(le.transform(y_train))    
    yy_test = to_categorical(le.transform(y_test))
    yy_val = to_categorical(le.transform(y_val))
    
    label_mapping = dict(zip(y_train, yy_train))
    
    return y_train, x_train, y_val, x_val, y_test, x_test, yy_train, yy_test, yy_val, y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end, label_mapping


# ## get Features for Trainng and Validation
# ### Parameters:
# 
# - **x_train**:  
#   Numpy array of features for the training set.
# 
# - **y_train**:  
#   List of dialect labels for the training set.
# 
# - **x_val**:  
#   Numpy array of features for the validation set.
# 
# - **y_val**:  
#   List of dialect labels for the validation set.
# 
# - **df_learn**:  
#   DataFrame containing audio data used for encoding labels.
# 
# ### Returns:
# 
# - **y_train**:  
#   List of shuffled dialect labels for the training set.
# 
# - **x_train**:  
#   Numpy array of shuffled features for the training set.
# 
# - **y_val**:  
#   List of shuffled dialect labels for the validation set.
# 
# - **x_val**:  
#   Numpy array of shuffled features for the validation set.
# 
# - **yy_train**:  
#   Categorical labels for the training set.
# 
# - **yy_val**:  
#   Categorical labels for the validation set.
# 
# - **label_mapping**:  
#   Mapping of original dialect labels to encoded categorical labels.
# 

# In[ ]:


def features_train_only(x_train, y_train, x_val, y_val, df_learn):
    y_train, x_train = shuffle(y_train, x_train)
    y_val, x_val = shuffle(y_val, x_val)

    # Encode the classification labels
    le = LabelEncoder()
    le.fit(sorted(df_learn['dialect'].unique().tolist()))
    yy_train = to_categorical(le.transform(y_train))
    yy_val = to_categorical(le.transform(y_val))
    
    label_mapping = dict(zip(y_train, yy_train))
    
    return y_train, x_train, y_val, x_val, yy_train, yy_val, label_mapping


# ## create model
# ### Parameters:
# 
# - **df_learn**:  
#   DataFrame containing audio data used for model training.
# 
# - **x_train**:  
#   Numpy array of features for the training set.
# 
# - **lr**:  
#   Learning rate for model optimization.
# 
# - **dr**:  
#   Dropout rate for regularization.
# 
# - **units**:  
#   Number of units/neurons in the dense layers of the model.
# 
# - **l1_val**:  
#   L1 regularization parameter for the dense layers.
# 
# - **l2_val**:  
#   L2 regularization parameter for the dense layers.
# 
# - **alpha_val**:  
#   Alpha parameter for LeakyReLU.
# 
# - **dummy**:  
#   Boolean indicating whether to use a DummyClassifier or a custom neural network model.
# 
# ### Returns:
# 
# - **model**:  
#   Compiled Keras model for classification.
# 
# - **callback**:  
#   EarlyStopping callback to monitor validation loss and restore the best weights during training.
# 

# In[ ]:


def create_model(df_learn, x_train, lr, dr, units, l1_val, l2_val, alpha_val, dummy):
    
    if dummy:
        model = DummyClassifier(strategy='prior') #stratified #prior #uniform #most_frequent
    else:
        dialects = df_learn['dialect'].unique().tolist()
        num_labels = len(dialects)
        
        f1_w = tfa.metrics.F1Score(num_classes=num_labels, average='weighted', name='f1_w')
        f1_m = tfa.metrics.F1Score(num_classes=num_labels, average='micro', name='f1_m')
        f1_macro = tfa.metrics.F1Score(num_classes=num_labels, average='macro', name='f1_macro')

        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'),
            f1_m,
            f1_w,
            f1_macro
        ]

        def build_model_graph(metrics=METRICS):
            model = Sequential()
            model.add(BatchNormalization(input_shape=(np.array(x_train).shape[-1],), name='BatchNorm'))
            model.add(Dense(units*2,
                            kernel_regularizer=l2(l2_val), activity_regularizer=l1(l1_val),
                            name='Dense1'))
            model.add(LeakyReLU(alpha=alpha_val))
            model.add(Dropout(dr))
            
            model.add(Dense(units,
                            kernel_regularizer=l2(l2_val), activity_regularizer=l1(l1_val),
                            name='Dense2'))
            model.add(LeakyReLU(alpha=alpha_val))  
            model.add(Dropout(dr))
           
            model.add(Dense(num_labels, name='Output'))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy', metrics=['accuracy', f1_m, f1_w, f1_macro], optimizer=Adam(learning_rate = lr))
            
            return model

        model = build_model_graph()

        callback = EarlyStopping(
            monitor="val_loss",
            mode='min',
            min_delta=0.005,
            patience=10,
            verbose=1,
            baseline=None,
            restore_best_weights=True,
        )

    return model, callback


# ## train Model
# ### Parameters:
# 
# - **x_train**:  
#   Numpy array of features for the training set.
# 
# - **x_val**:  
#   Numpy array of features for the validation set.
# 
# - **yy_train**:  
#   Categorical labels for the training set.
# 
# - **yy_val**:  
#   Categorical labels for the validation set.
# 
# - **class_weights**:  
#   Dictionary of class weights for handling class imbalance.
# 
# - **model**:  
#   Compiled Keras model for training.
# 
# - **batch_size**:  
#   Batch size used for training.
# 
# - **num_epochs**:  
#   Number of epochs for training.
# 
# - **callback**:  
#   EarlyStopping callback for monitoring validation loss.
# 
# - **i**:  
#   Index used for TensorBoard log directory.
# 
# - **lr**:  
#   Learning rate used in the model.
# 
# - **dr**:  
#   Dropout rate used in the model.
# 
# - **units**:  
#   Number of units/neurons in the dense layers of the model.
# 
# - **l1_val**:  
#   L1 regularization parameter used in the model.
# 
# - **l2_val**:  
#   L2 regularization parameter used in the model.
# 
# - **alpha_val**:  
#   Alpha parameter for LeakyReLU.
# 
# - **tb**:  
#   Boolean indicating whether to enable TensorBoard logging.
# 
# - **log_dir**:  
#   Directory path for TensorBoard logs.
# 
# - **dummy**:  
#   Boolean indicating whether to train a DummyClassifier.
#   
# - **train_only**:  
#   True if the run is only for training.
# 
# - **model_num**
#   The Number of the Model for saving the weights when train_only
# 
# ### Returns:
# 
# - **history**:  
#   History object containing training metrics and loss values.
# 

# In[7]:


def weights(y_train):
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))
    print('Class weights:', class_weights)
    return class_weights


# In[8]:


def train_model(x_train, x_val, yy_train, yy_val, class_weights, model, batch_size, num_epochs, callback, i, lr, dr, units, l1_val, l2_val, alpha_val, tb, log_dir, dummy, train_only, model_num=0):
      
    if tb:
        log_dir = log_dir + str(i) + "lr_" + str(lr) + "dr_" + str(dr) + "units_" + str(units) + "l1_" + str(l1_val) + "l2_" + str(l2_val) + "alpha_" + str(alpha_val) + "bs_" + str(batch_size)
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=(log_dir), histogram_freq=1,
        )
        callbacks = [callback, tensorboard_callback]
    else:
        callbacks = [callback]
    
    if dummy:
        integer_labels = np.argmax(yy_train, axis=1)
        history = model.fit(np.array(x_train), integer_labels)
    else:
        history = model.fit(np.array(x_train), yy_train, batch_size=batch_size, epochs=num_epochs,
                            validation_data=(np.array(x_val), yy_val), verbose=1,
                            shuffle=True, class_weight=class_weights, callbacks=callbacks)
    
    if train_only:
        model.save_weights('model_weights_' + str(model_num) + '.h5')
    
    return history


# ## test Model

# In[9]:


def pred(x_test, yy_test, model, dummy):
    pred_test = model.predict(np.array(x_test))

    if dummy:
        b = np.zeros((pred_test.size, pred_test.max() + 1))
        b[np.arange(pred_test.size), pred_test] = 1
        pred_test = b
        
    classes_x=np.argmax(pred_test,axis=1)
    classes_true=np.argmax(yy_test,axis=1)
    df_result = pd.DataFrame(list(zip(classes_x, classes_true)), columns=['Pred', 'True'])
    
    return df_result, classes_x, classes_true


# In[10]:


def get_false(y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end, df_result):
    indices = df_result.index[df_result['Pred'] != df_result['True']].tolist()
    false_names = [y_test_names[index] for index in indices]
    false_speaker = [y_test_speaker[index] for index in indices]
    false_segments_begin = [y_test_segment_begin[index] for index in indices]
    false_segments_end = [y_test_segment_end[index] for index in indices]
    false_simplified = [(y_test_names[index] + ' ' + str(y_test_segment_begin[index]/16000)) for index in indices]
    return false_names, false_speaker, false_segments_begin, false_segments_end, false_simplified


# In[11]:


def firstPictures(history, num_epochs):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 8))
    plt.plot(train_loss, label='Training loss', color='#185fad')
    plt.plot(val_loss, label='Validation loss', color='orange')
    plt.title('Training and Validation loss by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.xticks(range(0,num_epochs,5), range(0,num_epochs,5))
    plt.legend(fontsize = 18)
    plt.savefig('ex_loss_epoch_.png', bbox_inches='tight')
    plt.close()
    
    train_loss = history.history['f1_m']
    val_loss = history.history['val_f1_m']

    plt.figure(figsize=(12, 8))
    plt.plot(train_loss, label='Training F1', color='#185fad')
    plt.plot(val_loss, label='Validation F1', color='orange')
    plt.title('Training and Validation F1 by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.xticks(range(0,num_epochs,5), range(0,num_epochs,5))
    plt.legend(fontsize = 18)
    plt.savefig('ex_f1_epoch.png', bbox_inches='tight')
    plt.close()
    
    train_loss = history.history['f1_w']
    val_loss = history.history['val_f1_w']

    plt.figure(figsize=(12, 8))
    plt.plot(train_loss, label='Training weighted F1', color='#185fad')
    plt.plot(val_loss, label='Validation weighted F1', color='orange')
    plt.title('Training and Validation weighted F1 by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.xticks(range(0,num_epochs,5), range(0,num_epochs,5))
    plt.legend(fontsize = 18)
    plt.savefig('ex_weighted_f1_epoch.png', bbox_inches='tight')
    plt.close()
    
    train_loss = history.history['accuracy']
    val_loss = history.history['val_accuracy']

    plt.figure(figsize=(12, 8))
    plt.plot(train_loss, label='Training Accuracy', color='#185fad')
    plt.plot(val_loss, label='Validation Accuracy', color='orange')
    plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
    plt.xlabel('Epoch', fontsize = 18)
    plt.xticks(range(0,num_epochs,5), range(0,num_epochs,5))
    plt.legend(fontsize = 18)
    plt.savefig('ex_acc_epoch.png', bbox_inches='tight')
    plt.close()
    


# ## main function
# ### Parameters:
# 
# - **first_pictures**:  
#   Boolean indicating whether to generate plots for the runs during training.
# 
# - **df_learn**:  
#   DataFrame containing audio data used for model training.
# 
# - **df_learn_aug**:  
#   Augmented DataFrame containing audio data, if available.
# 
# - **speaker_test_val**:  
#   A list of dialects with each sublist containing a list of speakers. The first half of each sublist is designated for testing, and the second half is for validation.
# 
# - **name_aug**:  
#   A name identifier used for loading augmented data or specifying augmentation settings.
# 
# - **i**:  
#   Index used for TensorBoard log directory.
# 
# - **lr**:  
#   Learning rate used in the model.
# 
# - **dr**:  
#   Dropout rate used in the model.
# 
# - **units**:  
#   Number of units/neurons in the dense layers of the model.
# 
# - **l1_val**:  
#   L1 regularization parameter used in the dense layers.
# 
# - **l2_val**:  
#   L2 regularization parameter used in the dense layers.
# 
# - **batch_size**:  
#   Batch size used for training.
# 
# - **tb**:  
#   Boolean indicating whether to enable TensorBoard logging.
# 
# - **log_dir**:  
#   Directory path for TensorBoard logs.
# 
# - **dummy**:  
#   Boolean indicating whether to train a DummyClassifier.
# 
# - **max_epochs**:  
#   Maximum number of epochs for training.
# 
# ### Returns:
# 
# - **list_row**:  
#   A list containing various metrics and data for evaluation and analysis.
# 
# - **label_mapping**:  
#   Mapping of original dialect labels to encoded categorical labels.
# 
# ### Description:
# 
# This function orchestrates the entire workflow for training and evaluating the classification model for dialect classification. It performs data preprocessing, model creation, training, evaluation, and result extraction. Depending on the parameters, it either trains a custom neural network model or a DummyClassifier. It returns a list of evaluation metrics and data for analysis, along with a mapping of original dialect labels to encoded categorical labels.
# 

# In[12]:


def do_all(first_pictures, df_learn, df_learn_aug, speaker_test_val, name_aug, i, lr, dr, units, l1_val, l2_val, alpha_val, batch_size, tb, log_dir, dummy, max_epochs):

    x_train, y_train, x_test, y_test, x_val, y_val, y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end = train_test(speaker_test_val, df_learn, df_learn_aug, name_aug)
    y_train, x_train, y_val, x_val, y_test, x_test, yy_train, yy_test, yy_val, y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end, label_mapping = features(x_train, y_train, x_test, y_test, x_val, y_val, y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end, df_learn)
    class_weights = weights(y_train)
    model, callback = create_model(df_learn, x_train, lr, dr, units, l1_val, l2_val, alpha_val, dummy)
    history = train_model(x_train, x_val, yy_train, yy_val, class_weights, model, batch_size, max_epochs, callback, i, lr, dr, units, l1_val, l2_val, alpha_val, tb, log_dir, dummy, False)
    df_result, classes_x, classes_true = pred(x_test, yy_test, model, dummy)
    if dummy:
        accuracy = accuracy_score(classes_true, classes_x)
        f1_macro = f1_score(classes_true, classes_x, average='macro')
        f1_w = f1_score(classes_true, classes_x, average='weighted')
        loss = 0
    else:
        loss, accuracy, f1, f1_w, f1_macro = model.evaluate(np.array(x_test), yy_test, verbose=0)  
    false_names, false_speaker, false_segments_begin, false_segments_end, false_simplified = get_false(y_test_names, y_test_speaker, y_test_segment_begin, y_test_segment_end, df_result)
    if first_pictures and not dummy:
        firstPictures(history, max_epochs)
    
    speaker_val = []
    speaker_test = []
    speaker_val = [sublist[len(sublist)//2:] for sublist in speaker_test_val]
    speaker_test = [sublist[0:len(sublist)//2] for sublist in speaker_test_val]
    
    if dummy:
        list_row = [None, None, accuracy,
                    None, None, loss,
                    None, None, f1_macro,
                    None, None, f1_w,
                    speaker_val, speaker_test, false_names, false_speaker, false_segments_begin, false_segments_end,
                    false_simplified, classes_x, classes_true, y_test_names, y_test_speaker]
    else:
        list_row = [history.history['accuracy'][-1], history.history['val_accuracy'][-1], accuracy,
                    history.history['loss'][-1], history.history['val_loss'][-1], loss,
                    history.history['f1_m'][-1], history.history['val_f1_m'][-1], f1,
                    history.history['f1_macro'][-1], history.history['val_f1_macro'][-1], f1_macro,
                    history.history['f1_w'][-1], history.history['val_f1_w'][-1], f1_w,
                    speaker_val, speaker_test, false_names, false_speaker, false_segments_begin, false_segments_end,
                    false_simplified, classes_x, classes_true, y_test_names, y_test_speaker]
    
    return list_row, label_mapping


# ## main function for Training
# ### Parameters:
# 
# - **model_num**
#   The Number of the Model for saving the weights
# 
# - **first_pictures**:  
#   Boolean indicating whether to generate plots for the runs during training.
# 
# - **df_learn**:  
#   DataFrame containing audio data used for model training.
# 
# - **df_learn_aug**:  
#   Augmented DataFrame containing audio data, if available.
# 
# - **speaker_val**:  
#   A list of speakers for validation.
# 
# - **name_aug**:  
#   A name identifier used for loading augmented data or specifying augmentation settings.
# 
# - **lr**:  
#   Learning rate used in the model.
# 
# - **dr**:  
#   Dropout rate used in the model.
# 
# - **units**:  
#   Number of units/neurons in the dense layers of the model.
# 
# - **l1_val**:  
#   L1 regularization parameter used in the dense layers.
# 
# - **l2_val**:  
#   L2 regularization parameter used in the dense layers.
# 
# - **batch_size**:  
#   Batch size used for training.
# 
# - **max_epochs**:  
#   Maximum number of epochs for training.
# 
# ### Returns:
# 
# - **list_row**:  
#   A list containing various metrics and data for evaluation and analysis.
# 
# - **label_mapping**:  
#   Mapping of original dialect labels to encoded categorical labels.
# 
# ### Description:
# 
# This function orchestrates the entire workflow for training the classification model for dialect classification. It performs data preprocessing, model creation, training and saving the model weights.
# 

# In[ ]:


def do_all_train(model_num, first_pictures, df_learn, df_learn_aug, speaker_val, name_aug, lr, dr, units, l1_val, l2_val, alpha_val, batch_size, max_epochs):

    x_train, y_train, x_val, y_val = train_val(speaker_val, df_learn, df_learn_aug, name_aug)
    y_train, x_train, y_val, x_val, yy_train, yy_val, label_mapping = features_train_only(x_train, y_train, x_val, y_val, df_learn)
    class_weights = weights(y_train)
    model, callback = create_model(df_learn, x_train, lr, dr, units, l1_val, l2_val, alpha_val, False)
    history = train_model(x_train, x_val, yy_train, yy_val, class_weights, model, batch_size, max_epochs, callback, 0, lr, dr, units, l1_val, l2_val, alpha_val, False, '', False, True, model_num)
    if first_pictures:
        firstPictures(history, max_epochs)
    
    list_row = [history.history['accuracy'][-1], history.history['val_accuracy'][-1], '',
                history.history['loss'][-1], history.history['val_loss'][-1], '',
                history.history['f1_m'][-1], history.history['val_f1_m'][-1], '',
                history.history['f1_macro'][-1], history.history['val_f1_macro'][-1], '',
                history.history['f1_w'][-1], history.history['val_f1_w'][-1], '',
                speaker_val, '', '', '', '', '',
                '', '', '', '', '']
    
    return list_row, label_mapping


# ## main function for Testing
# ### Parameters:
# 
# - **model_num**
#   The Number of the Model for loading the right weights
# 
# - **df_learn**:  
#   DataFrame containing audio data used for model training.
# 
# - **df_learn_test**:  
#   DataFrame containing audio data used for model testing.
# 
# - **lr**:  
#   Learning rate used in the model.
# 
# - **dr**:  
#   Dropout rate used in the model.
# 
# - **units**:  
#   Number of units/neurons in the dense layers of the model.
# 
# - **l1_val**:  
#   L1 regularization parameter used in the dense layers.
# 
# - **l2_val**:  
#   L2 regularization parameter used in the dense layers.
# 
# ### Returns:
# 
# - **predictions**:  
#   A list containing all predictions.
# 
# ### Description:
# 
# This function orchestrates the entire workflow for testing the classification model for dialect classification.
# 

# In[ ]:


def do_all_test(model_num, df_learn, df_test, lr, dr, units, l1_val, l2_val, alpha_val):
    
    x_test = np.asarray(df_test['trillsson'].tolist())
    x_speaker = np.asarray(df_test['speaker'].tolist())
    x_dialect = np.asarray(df_test['dialect'].tolist())
    model, callback = create_model(df_learn, x_test, lr, dr, units, l1_val, l2_val, alpha_val, False)
    model.load_weights('model_weights_' + str(model_num) + '.h5')
    predictions = model.predict(x_test)
    
    return predictions, x_speaker, x_dialect


# In[ ]:


def do_all_embeddings(model_num, df_learn, df_test, lr, dr, units, l1_val, l2_val, alpha_val):

    x_test = np.asarray(df_test['trillsson'].tolist())
    x_speaker = np.asarray(df_test['speaker'].tolist())
    x_dialect = np.asarray(df_test['dialect'].tolist())
    model, callback = create_model(df_learn, x_test, lr, dr, units, l1_val, l2_val, alpha_val, dummy=False)
    model.load_weights(f'model_weights_{model_num}.h5')

    feature_layer = model.get_layer('Dense2').output
    feature_model = Model(inputs=model.input, outputs=feature_layer)
    features = feature_model.predict(x_test)

    return features, x_speaker, x_dialect


# In[ ]:




