#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.colors as mcolors
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib
import pickle
import sys

matplotlib.use("Agg")


# ## get Boxplot and raincloudplot of results

# In[2]:


def boxplot(column, name, name_aug):
    df = pd.read_pickle('./Results_' + name + '_' + name_aug + '.pkl')
    dat = df[column].tolist()
    print(column)
    print('std:', np.std(dat))
    print('median:', np.median(dat))
    print('mean:', np.mean(dat))
    data = [dat]
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.title("Boxplot of " + column)
    bp = ax.boxplot(data, showmeans=True)
    ax.set_ylim(0,0.5)
    plt.savefig('boxplot_' + column + '_' + name + '_' + name_aug + '.png', bbox_inches='tight')
    plt.close()
    
    raincloudplot(column, name, name_aug)
    


# In[3]:


def raincloudplot(column, name, name_aug):
    
    fig, ax = plt.subplots(figsize=(8, 2))

    boxplot_colors = ['yellowgreen']
    violin_colors = ['purple']
    scatter_colors = ['tomato']

    df = pd.read_pickle('./Results_' + name + '_' + name_aug + '.pkl')
    data = [df[column].tolist()]

    bp = ax.boxplot(data, patch_artist=True, vert=False)

    for patch, color in zip(bp['boxes'], boxplot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    vp = ax.violinplot(data, points=500, 
                       showmeans=False, showextrema=False, showmedians=False, vert=False)

    for idx, b in enumerate(vp['bodies']):
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx + 1, idx + 2)
        b.set_color(violin_colors[idx])

    for idx, features in enumerate(data):
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.1, high=.1, size=len(idxs))
        y = out
        plt.scatter(features, y, s=.3, c=scatter_colors[idx])

    plt.yticks(np.arange(1, 2, 1), [name])
    plt.xlabel('Scores')
    ax.set_xlim(0,0.5)
    plt.title("RainCloudPlot of " + column)
    plt.savefig('raincloudplot_' + column + '_' + name + '_' + name_aug + '.png', bbox_inches='tight')
    plt.close()


# ## confusion matrix

# In[4]:


def confusionMatrix(name, name_aug, mapping, absolute, column):
    df_results = pd.read_pickle('./Results_' + name + '_' + name_aug + '.pkl')
    
    y_true = np.concatenate(df_results['classes_true'].tolist()).ravel()
    y_pred = np.concatenate(df_results['classes_x'].tolist()).ravel()

    mapping = {key: np.argmax(value) for key, value in mapping.items()}
    reversed_mapping = {value: key for key, value in mapping.items()}
    true_mapped_list = [reversed_mapping[key] for key in y_true]
    pred_mapped_list = [reversed_mapping[key] for key in y_pred]
    cm_labels = np.unique(true_mapped_list)
                  
    matrix = confusion_matrix(true_mapped_list, pred_mapped_list)
        
    if absolute:
        df_cfm = pd.DataFrame(matrix, index=cm_labels, columns=cm_labels)
    elif column:
        column_sums = matrix.sum(axis=0, keepdims=True)
        column_sums = np.where(column_sums == 0, sys.float_info.epsilon, column_sums)
        normalized_matrix = matrix / column_sums
        df_cfm = pd.DataFrame(normalized_matrix, index=cm_labels, columns=cm_labels)
    else:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, sys.float_info.epsilon, row_sums)
        normalized_matrix = matrix / row_sums
        df_cfm = pd.DataFrame(normalized_matrix, index=cm_labels, columns=cm_labels)
    
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='.2f' if not absolute else 'g')
    
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    cfm_plot.set_xticklabels(cfm_plot.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    cfm_plot.set_yticklabels(cfm_plot.get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    
    if absolute:
        if '_combined' in name_aug:
            cfm_plot.figure.savefig("confusion_matrix_combined.png")
        else:
            cfm_plot.figure.savefig("confusion_matrix.png")
    elif column:
        if '_combined' in name_aug:
            cfm_plot.figure.savefig("column_normalized_confusion_matrix_combined.png")
        else:
            cfm_plot.figure.savefig("column_normalized_confusion_matrix.png")
    else:
        if '_combined' in name_aug:
            cfm_plot.figure.savefig("normalized_confusion_matrix_combined.png")
        else:
            cfm_plot.figure.savefig("normalized_confusion_matrix.png")
    plt.close()
        


# In[5]:


def speakerFalse(name, name_aug):
    df_results = pd.read_pickle('./Results_' + name + '_' + name_aug + '.pkl')
    df_data = pd.read_pickle('./All_Files_.pkl')
    dialects = df_data['dialect'].unique().tolist()
    speaker = [[] for _ in range(0, len(dialects))]
    
    all_speaker = np.concatenate(df_results['y_test_speaker'].tolist()).ravel()
    false_speaker = np.concatenate(df_results['false_speaker'].tolist()).ravel()
    
    all_speaker_test = []
    
    for index, value in df_results['speaker_test'].iteritems():
        flat_array = np.array([item for sublist in value for item in sublist])
        all_speaker_test.extend(flat_array)
    
    for i in range(0, len(dialects)):
        speaker[i] = df_data[(df_data['dialect'] == dialects[i]) & (df_data['augmented'] == 'False')]['speaker'].unique().tolist()
        perc = []
        total = []
        for j in range(0, len(speaker[i])):
            cnt_all = all_speaker.tolist().count(speaker[i][j])
            cnt_false = false_speaker.tolist().count(speaker[i][j])
            try:
                perc.append(round((100/cnt_all)*cnt_false, 1))
            except ZeroDivisionError:
                print('warning:', speaker[i][j], 'is 0 times in the train-set.')
                perc.append(0)
            cnt_speaker = all_speaker_test.count(speaker[i][j])
            total.append(cnt_speaker)
        
        fig, (cax, sax, lax, ax) = plt.subplots(nrows=4, figsize=(10,5),  gridspec_kw={"height_ratios":[0.3, 0.2, 0.3, 0.7]})
        cfm_plot = sn.heatmap([perc], annot=True, square=True, vmin=0, vmax=100, xticklabels=speaker[i],
                              yticklabels=False, ax=ax, cbar=False)
        cfm_plot.axis('tight')
        colors = ['black', 'white']
        cmap = mcolors.ListedColormap(colors)
        cfm_plot2 = sn.heatmap([total], annot=True, square=True, cmap=cmap, vmin=0, vmax=10, xticklabels=False,
                               yticklabels=False, ax=lax, cbar=False, linewidths = 2, linecolor = "black")
        cfm_plot3 = sn.heatmap([total], xticklabels=False, yticklabels=False, ax=sax, cbar=False, cmap=cmap, vmin=-5, vmax=-1)
        cfm_plot2.axis('tight')
        fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
        for _, spine in cfm_plot2.spines.items():
            spine.set_visible(True)
        
        fig.tight_layout()
        cfm_plot.figure.savefig('confusion_matrix_' + dialects[i] + '.png')
        plt.close(fig)
        


# ## calcualte weighted f1

# In[6]:


def weighted_f1(name, name_aug, mapping):
    df_results = pd.read_pickle('./Results_' + name + '_' + name_aug + '.pkl')
    df_data = pd.read_pickle('./Data_.pkl')
    
    mapping = {key: np.argmax(value) for key, value in mapping.items()}
    
    class_weights = df_data['dialect'].value_counts(normalize=True).to_dict()
    class_weights = {name: weight for name, weight in class_weights.items()}
    flattened_list = [item for sublist in df_results['classes_true'] for item in sublist]
    counts = Counter(flattened_list)
    
    y_true = np.concatenate(df_results['classes_true'].tolist()).ravel()
    y_pred = np.concatenate(df_results['classes_x'].tolist()).ravel()
    
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []
    F1_list = []
    weights_list_all = []
    weights_list_test = []
    
    for label, val in class_weights.items():
        mapped_label = mapping[label]

        cm = confusion_matrix(y_true == mapped_label, y_pred == mapped_label)
    
        TP = cm[1, 1]
        FP = cm[0, 1]
        TN = cm[0, 0]
        FN = cm[1, 0]
    
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        
        F1_list.append(TP/(TP+0.5*(FP+FN)))
        weights_list_all.append(val)
        
        weights_list_test.append(counts[mapped_label])

    weighted_F1_score_all = np.sum([w * f1 for w, f1 in zip(weights_list_all, F1_list)]) / np.sum(weights_list_all)
    print('weighted over All F1 score:', weighted_F1_score_all)
    
    weighted_F1_score_test = np.sum([w * f1 for w, f1 in zip(weights_list_test, F1_list)]) / np.sum(weights_list_test)
    print('weighted over Test F1 score:', weighted_F1_score_test)
    
    names = list(class_weights.keys())
    sorted_names = sorted(names)
    sorted_f1_scores = [F1_list[names.index(name)] for name in sorted_names]
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(sorted_names))
    plt.barh(y_pos, sorted_f1_scores, color='blue')
    plt.yticks(y_pos, sorted_names)
    plt.xlabel('F1-Score')
    plt.title('F1-Scores per class')
    plt.xlim(0, 1)
    plt.savefig('f1_per_class.png', bbox_inches='tight')
    plt.close()

    with open('f1_scores_per_class.txt', 'w') as f:
        for i in range(0, len(sorted_names)):
            f.write(f'{sorted_names[i]}, F1 Score: {sorted_f1_scores[i]:.4f}\n')
        for i in range(0, len(sorted_names)):
            f.write(f'{sorted_f1_scores[i]:.3f}\n')
    


# ## Scores of combined Segments

# In[1]:


def combined_scores(name, name_aug):
    data = pd.read_pickle('./Results_' + name + '_' + name_aug + '.pkl')

    classes_x, classes_true, y_test_speaker = data['classes_x'], data['classes_true'], data['y_test_speaker']

    f1_scores = []
    conf_matrices = []
    accuracies = []
    all_reduced_data = []

    df_new = pd.DataFrame(columns=['classes_x', 'classes_true', 'y_test_speaker'])

    for row_classes_x, row_classes_true, row_speaker in zip(classes_x, classes_true, y_test_speaker):
        speaker_predictions = {}
        speaker_true = {}
    
        unique_speakers = set(row_speaker)
        for speaker in unique_speakers:
            speaker_indices = [i for i, s in enumerate(row_speaker) if s == speaker]
            speaker_classes_x = row_classes_x[speaker_indices]
            speaker_classes_true = row_classes_true[speaker_indices]
            most_predicted_class = np.argmax(np.bincount(speaker_classes_x))
            speaker_predictions[speaker] = most_predicted_class
            speaker_true[speaker] = speaker_classes_true[0]

        true_labels = np.array(list(speaker_true.values()))
        predicted_labels = np.array(list(speaker_predictions.values()))

        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        f1_scores.append(f1)

        correct_count = np.sum(predicted_labels == true_labels)
        accuracy = correct_count / len(true_labels)
        accuracies.append(accuracy)
    
        df_new.loc[len(df_new)] = [predicted_labels, true_labels, list(unique_speakers)]

    mean_f1 = np.mean(f1_scores)
    median_f1 = np.median(f1_scores)
    std_f1 = np.std(f1_scores)

    print("Mean F1 Score:", mean_f1)
    print("Median F1 Score:", median_f1)
    print("Std F1 Score:", std_f1)

    mean_accuracy = np.mean(accuracies)
    median_accuracy = np.median(accuracies)
    std_accuracy = np.std(accuracies)

    print("Mean Accuracy:", mean_accuracy)
    print("Median Accuracy:", median_accuracy)
    print("Std Accuracy:", std_accuracy)

    df_new.to_pickle('./Results_' + name + '_' + name_aug + '_combined' + '.pkl')
    df_new.to_csv('./Results_' + name + '_' + name_aug + '_combined' + '.csv',  sep=';')


# ## Boxplots of Predictions per Folder

# In[ ]:


def plot_dialect_boxplots_test(predictions_df, df_test, label_mapping):
    
    label_to_index = {dialect: np.argmax(vec) for dialect, vec in label_mapping.items()}
    index_to_label = {v: k for k, v in label_to_index.items()}

    results = []

    def get_percentages_for_dialect(prediction_row):
        return {index_to_label[i]: prediction_row[i] * 100 for i in range(len(prediction_row))}

    for i, row in predictions_df.iterrows():
        percentages = get_percentages_for_dialect(row['predictions'])
        percentages['dialect'] = row['dialect']
        results.append(percentages)

    results_df = pd.DataFrame(results)
    unique_dialects = results_df['dialect'].unique()
    
    for dialect in unique_dialects:
        plt.figure(figsize=(12, 6))
        filtered_df = results_df[results_df['dialect'] == dialect]
        melted_df = filtered_df.melt(id_vars='dialect', var_name='feature', value_name='percentage')
        sn.boxplot(x='feature', y='percentage', data=melted_df)
        
        plt.title(f'Percentage Classification for {dialect}')
        plt.xticks(rotation=90)
        plt.ylim(0, 100)
        plt.ylabel('Percentage (%)')
        plt.xlabel('Dialect')
        plt.tight_layout()
        save_path = f'prediction_{dialect}.png'
        plt.savefig(save_path)
        plt.close()


# ## Stacked Barplots of Predictions per Folder and Speaker

# In[ ]:


def plot_dialect_predictions(predictions_df, label_mappings):
    predictions_df['predictions'] = predictions_df['predictions'].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
    )
    unique_dialects = predictions_df['dialect'].unique()
    cmap = plt.cm.get_cmap('tab20', len(label_mappings))
    
    color_mapping = {dialect: cmap(i) for i, dialect in enumerate(label_mappings.keys())}

    for test_dialect in unique_dialects:
        filtered_df = predictions_df[predictions_df['dialect'] == test_dialect]
        stacked_data = {}

        for dialect, label_vector in label_mappings.items():
            for _, row in filtered_df.iterrows():
                speaker = row['speaker']
                prediction = row['predictions']
                if speaker not in stacked_data:
                    stacked_data[speaker] = {d: 0 for d in label_mappings.keys()}
                stacked_data[speaker][dialect] += np.mean(prediction[label_vector > 0])

        stacked_df = pd.DataFrame(stacked_data).T.fillna(0)
        stacked_df = stacked_df.div(stacked_df.sum(axis=1).replace(0, np.finfo(float).eps), axis=0)

        sorted_columns = stacked_df.sum(axis=0).sort_values(ascending=False).index
        sorted_df = stacked_df[sorted_columns]

        color_list = [color_mapping[col] for col in sorted_columns]

        num_speakers = len(stacked_df)
        plot_width = num_speakers * 2
        plt.figure(figsize=(plot_width + 3, 6))
        
        ax = sorted_df.plot(kind='bar', stacked=True, color=color_list, ax=plt.gca())

        plt.title(f'Normalized Predictions by Speaker: {test_dialect}')
        plt.ylabel('Proportion of Predictions')
        plt.xticks(rotation=45, ha='right')
        ax.legend(sorted_df.columns, title='Dialects', loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        save_path = f'normalized_predictions_{test_dialect}.png'
        plt.savefig(save_path)
        plt.close()


# In[ ]:


def plot_dialect_predictions_samples(predictions_df, label_mappings, threshold=0):
    predictions_df['predictions'] = predictions_df['predictions'].apply(np.array)
    unique_dialects = predictions_df['dialect'].unique()
    cmap = plt.cm.get_cmap('tab20', len(label_mappings))
    
    color_mapping = {dialect: cmap(i) for i, dialect in enumerate(label_mappings.keys())}
    color_mapping['not properly classified'] = 'whitesmoke'
    
    for test_dialect in unique_dialects:
        filtered_df = predictions_df[predictions_df['dialect'] == test_dialect]
        stacked_data = {}

        for index, row in filtered_df.iterrows():
            speaker = row['speaker']
            prediction = row['predictions']
            max_pred_value = np.max(prediction)
            max_pred_label = np.argmax(prediction)

            if speaker not in stacked_data:
                stacked_data[speaker] = {d: 0 for d in label_mappings.keys()}
                stacked_data[speaker]['not properly classified'] = 0

            if max_pred_value >= threshold:
                for dialect, label_vector in label_mappings.items():
                    if label_vector[max_pred_label] > 0:
                        stacked_data[speaker][dialect] += 1
                        break
            else:
                stacked_data[speaker]['not properly classified'] += 1

        stacked_df = pd.DataFrame(stacked_data).T.fillna(0)
        stacked_df = stacked_df.div(stacked_df.sum(axis=1), axis=0)

        sorted_columns = stacked_df.sum(axis=0).sort_values(ascending=False).index
        sorted_df = stacked_df[sorted_columns]

        color_list = [color_mapping[col] for col in sorted_columns]

        num_speakers = len(stacked_df)
        plot_width = num_speakers * 1.0
        plt.figure(figsize=(plot_width + 3, 6))

        ax = sorted_df.plot(kind='bar', stacked=True, color=color_list, width=0.8, ax=plt.gca())

        plt.ylabel('Proportion of Predictions', fontsize=14)
        plt.xticks(rotation=30, ha='right', fontsize=14)
        
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['legend.title_fontsize'] = 12
        leg = ax.legend(sorted_df.columns, title='Dialects', loc='upper left', bbox_to_anchor=(1, 1))
        ax.legend(sorted_df.columns, title='Dialects', loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        save_path = f'normalized_predictions_{test_dialect}_{int(threshold*100)}percent.png'
        plt.savefig(save_path)
        plt.close()


# In[ ]:




