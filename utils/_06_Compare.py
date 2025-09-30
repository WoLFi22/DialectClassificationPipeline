#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.metrics import confusion_matrix
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


# ## calculates t-test and u-test

# In[ ]:


def format_apa_t(mean1, sd1, n1, mean2, sd2, n2, t_stat, pval, d, df):
    p_str = f"< .001" if pval < .001 else f"= {pval:.3f}".lstrip("0")
    return (f"M = {mean2:.3f}, SD = {sd2:.3f}, n = {n2}; "
            f"M = {mean1:.3f}, SD = {sd1:.3f}, n = {n1}; "
            f"t({df}) = {t_stat:.2f}, p {p_str}, d = {d:.2f}")

def format_apa_u(mdn1, mdn2, U_stat, pval, r):
    p_str = f"< .001" if pval < .001 else f"= {pval:.3f}".lstrip("0")
    return (f"Mdn = {mdn2:.3f} vs. {mdn1:.3f}; "
            f"U = {U_stat}, p {p_str}, r = {r:.2f}")


# In[ ]:


def significant(path_base, path_aug, column, alternative):
    df_base = pd.read_pickle(path_base)
    df_aug = pd.read_pickle(path_aug)
    
    f1_base = df_base[column].tolist()
    f1_aug = df_aug[column].tolist()
    
    U_base, p = mannwhitneyu(f1_base, f1_aug, alternative=alternative)
    n_base, n_aug = len(f1_base), len(f1_aug)
    U_aug = n_base*n_aug - U_base
    
    res = ttest_ind(f1_base, f1_aug, alternative=alternative)
    
    var_base, var_aug = np.var(f1_base), np.var(f1_aug)
    
    mean_base = np.mean(f1_base)
    sd_base   = np.std(f1_base, ddof=1)
    mean_aug  = np.mean(f1_aug)
    sd_aug    = np.std(f1_aug,   ddof=1)
    n_base, n_aug = len(f1_base), len(f1_aug)
    
    pooled_sd = np.sqrt(((n_base-1)*sd_base**2 + (n_aug-1)*sd_aug**2) / (n_base + n_aug - 2))
    cohens_d = (mean_aug - mean_base) / pooled_sd

    df = n_base + n_aug - 2

    mu_U    = n_base * n_aug / 2
    sigma_U = np.sqrt(n_base * n_aug * (n_base + n_aug + 1) / 12)
    z = (U_base - mu_U) / sigma_U
    r = z / np.sqrt(n_base + n_aug)

    return U_base, U_aug, p, res, var_base, var_aug, format_apa_t(mean_base, sd_base, n_base, mean_aug, sd_aug, n_aug, res.statistic, res.pvalue, cohens_d, df), format_apa_u(np.median(f1_base), np.median(f1_aug), U_base, p, r)



# ## boxplot and raincoudplot of difference from two different runs

# In[3]:


def boxplot_diff(column, base, name, name_aug):
    df = pd.read_pickle('./Results_' + base + '.pkl')
    df_aug = pd.read_pickle('./Results_' + name + '_' + name_aug + '.pkl')
    dat = df[column].tolist()
    dat2 = df_aug[column].tolist()
    data = [dat, dat2]
    fig = plt.figure(figsize =(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([1, 2])
    ax.set_xticklabels([base, name])
    plt.title("Boxplot of " + column)
    bp = ax.boxplot(data, showmeans=True)
    ax.set_ylim(0,0.5)
    plt.savefig('boxplot_diff_' + column + '_' + base + '_' + name + '_' + name_aug + '.png', bbox_inches='tight')
    plt.close()
    


# In[4]:


def raincloud_diff(column, base, name, name_aug):
    
    fig, ax = plt.subplots(figsize=(8, 4))

    boxplots_colors = ['yellowgreen', 'yellowgreen']
    
    df = pd.read_pickle('./Results_' + base + '.pkl')
    df_aug = pd.read_pickle('./Results_' + name + '_' + name_aug + '.pkl')
    dat = df[column].tolist()
    dat2 = df_aug[column].tolist()
    data = [dat, dat2]

    bp = ax.boxplot(data, patch_artist = True, vert = False)

    for patch, color in zip(bp['boxes'], boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    violin_colors = ['purple', 'purple']

    vp = ax.violinplot(data, points=500, 
               showmeans=False, showextrema=False, showmedians=False, vert=False)

    for idx, b in enumerate(vp['bodies']):
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], idx+1, idx+2)
        b.set_color(violin_colors[idx])

    scatter_colors = ['tomato', 'tomato']

    for idx, features in enumerate(data):
        y = np.full(len(features), idx + .8)
        idxs = np.arange(len(y))
        out = y.astype(float)
        out.flat[idxs] += np.random.uniform(low=-.1, high=.1, size=len(idxs))
        y = out
        plt.scatter(features, y, s=.3, c=scatter_colors[idx])

    plt.yticks(np.arange(1,3,1), [base, name])
    plt.xlabel('Scores')
    ax.set_xlim(0,0.5)
    plt.title("RainCloudPlot of " + column)
    plt.savefig('raincloudplot_diff_' + column + '_' + base + '_' + name + '_' + name_aug + '.png', bbox_inches='tight')
    plt.close()
    


# ## difference of confusion matrices

# In[1]:


def compareConfusionMatrix(name, name_aug, name_base, mapping, absolute):
    df_results1 = pd.read_pickle('./Results_' + name + '_' + name_aug + '.pkl')
    df_results2 = pd.read_pickle('./Results_' + name_base + '.pkl')
    
    y_true1 = np.concatenate(df_results1['classes_true'].tolist()).ravel()
    y_pred1 = np.concatenate(df_results1['classes_x'].tolist()).ravel()
    
    y_true2 = np.concatenate(df_results2['classes_true'].tolist()).ravel()
    y_pred2 = np.concatenate(df_results2['classes_x'].tolist()).ravel()

    mapping = {key: np.argmax(value) for key, value in mapping.items()}
    reversed_mapping = {value: key for key, value in mapping.items()}
    
    true_mapped_list1 = [reversed_mapping[key] for key in y_true1]
    pred_mapped_list1 = [reversed_mapping[key] for key in y_pred1]
    
    true_mapped_list2 = [reversed_mapping[key] for key in y_true2]
    pred_mapped_list2 = [reversed_mapping[key] for key in y_pred2]
    
    cm_labels = np.unique(true_mapped_list1 + true_mapped_list2)
                  
    matrix1 = confusion_matrix(true_mapped_list1, pred_mapped_list1, labels=cm_labels)
    matrix2 = confusion_matrix(true_mapped_list2, pred_mapped_list2, labels=cm_labels)

    if absolute:
        diff_matrix = matrix1 - matrix2
        df_cfm = pd.DataFrame(diff_matrix, index=cm_labels, columns=cm_labels)
    else:
        row_sums1 = matrix1.sum(axis=1, keepdims=True)
        row_sums2 = matrix2.sum(axis=1, keepdims=True)
        normalized_matrix1 = matrix1 / row_sums1
        normalized_matrix2 = matrix2 / row_sums2
        diff_matrix = normalized_matrix1 - normalized_matrix2
        df_cfm = pd.DataFrame(diff_matrix, index=cm_labels, columns=cm_labels)
    
    plt.figure(figsize = (10,7))
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='.2f' if not absolute else 'g', annot_kws={"fontsize": 8} if not absolute else {"fontsize": 10})
    
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    cfm_plot.set_xticklabels(cfm_plot.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    cfm_plot.set_yticklabels(cfm_plot.get_yticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    
    if absolute:
        cfm_plot.figure.savefig("diff_confusion_matrix.png")
    else:
        cfm_plot.figure.savefig("diff_normalized_confusion_matrix.png")
    
    plt.close()
    


# In[ ]:




