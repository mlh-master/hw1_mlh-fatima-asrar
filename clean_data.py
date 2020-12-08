# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """
    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    c_ctg={q: s.dropna() for q, s in (CTG_features.replace(to_replace=['NaN','--','Nan',"#"],value=np.nan).drop([extra_feature],                     axis =1)).iteritems()}
   
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    
    c_cdf={q:s for q,s in (CTG_features.replace(to_replace=['NaN','--','Nan',"#"],value=np.nan).drop([extra_feature], axis                          =1)).iteritems()}

    for key in c_cdf:
        data = c_cdf[key]
        nans = data.isnull()
        for i in range(1,(len(data))+1):
            if np.isnan(data[i]):
                data[i]=np.random.choice(data[~nans].values)
 

    return (pd.DataFrame(c_cdf))


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    
    """
    
    d_summary=c_feat.describe().drop(['std','count' ,'mean']).rename({'25%':"Q1" , '50%': "median" , '75%':                                         "Q3"}).to_dict()
   
    return d_summary



def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    for key in d_summary:
        IQR= d_summary[key]['Q3'] - d_summary[key]['Q1']                                
        c_no_outlier[key] = c_feat[key][(c_feat[key] >= (d_summary[key]['Q1'] - 1.5 * IQR)) & (c_feat[key] <= (d_summary[key]['Q3']           +1.5 *IQR))]          
       
    return (pd.DataFrame(c_no_outlier))


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    index=c_cdf[feature] < thresh
    filt_feature=c_cdf[feature][index]

    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    nsd_res=CTG_features.copy()
    for col in nsd_res.columns:
        
        if mode == 'standart' :
            nsd_res[col] =(CTG_features[col] -np.mean(CTG_features[col]))/np.std(CTG_features[col])
        
        elif mode == 'mean':
            nsd_res[col] =(CTG_features[col] -np.mean(CTG_features[col]))/(max(CTG_features[col])-min(CTG_features[col]))
        
        elif mode =='MinMax':
            nsd_res[col] =(CTG_features[col] -min(CTG_features[col]))/(max(CTG_features[col])-min(CTG_features[col]))
            
        elif mode == 'none':
            nsd_res[col]=CTG_features[col]
  

    if flag :
            d=nsd_res.hist(column=[x,y],bins=100,layout=(1,2))
            for i, ax in enumerate(d.flatten()):
                ax.set_xlabel([selected_feat[i],mode])
                ax.set_ylabel("Count") 
            plt.show()    
            
    return pd.DataFrame(nsd_res)
