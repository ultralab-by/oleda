import numpy as np
import matplotlib.pyplot as pls 
import pandas as pd

from IPython.display import display, HTML
import seaborn as sns


from .eda_core import *
from .eda_core import __cramer_v_corr
from .eda_pairwise import pairwise_report


import warnings

warnings.filterwarnings("ignore")

#=====================#=====================#=====================
# dataset comparision
#=====================#=====================#=====================  
#
#pairwise report
#
#def pairwise_report(df1,df2,target=None,ignore=[],nbrmax=20,full=True):
    # Compare two datasets:
    # df1         pandas dataframe     first dataset 
    # df2         pandas dataframe     second dataset
    # ignore   list   features to ignore
    # nbrmax  int     max number of features (with max shap values) to print
    # full            full mode : also prints nan correlations heatmaps
#pairwise_report(df1,df2,target,ignore,nbrmax,full)

#for compatibility
print_report = pairwise_report      

#=====================#=====================#=====================
# single dataset  eda
#=====================#=====================#=====================  

#single dataset report
def report(df,target,ignore=[],nbrmax=20,full=True):
     return do_eda(df, target,ignore,nbrmax,full)  

#shap values
def  plot_shaps(df, target,ignore=[],nbrmax=20):
    #
    # plot shaps values
    # target should be binary
    # returns features list sorted by importance
    #    
    return plot_shap(df, target,ignore,nbrmax)

#=====================#=====================#=====================
# time series plots
#=====================#=====================#=====================
    
# plots n top feature values counts per day   
def plot_ntop_categorical_values_counts(df,feature,target,nbr_max=4,figsize=(20,4),linewidth=2.0,period="1d"):
    values=df[feature].value_counts()[:nbr_max].index.to_list()
    if  len(values)==0:
        return
    ax=df[df[feature]==values[0]][target].resample(period).count().plot(x_compat=True,figsize=figsize, grid=True,linewidth=linewidth)
    legend=[values[0]]
    for i in range(1,len(values)):
        df[df[feature]==values[i]][target].resample(period).count().plot(x_compat=True,figsize=figsize,ax=ax, grid=True, linewidth=2.0,title='{} per day'.format(feature))
        legend.append(values[i])

    if len(values) > 1:
        ax.lines[1].set_linestyle(":")
    ax.lines[0].set_linestyle("--")
    pls.legend(legend, bbox_to_anchor=(1.2, 0))
    pls.show()
    
    
def plot_ntop_categorical_values_sums(df,feature,target,nbr_max=4,figsize=(20,4),linewidth=2.0,period="1d"):
    values=df[feature].value_counts()[:nbr_max].index.to_list()
    if  len(values)==0:
        return
    ax=df[df[feature]==values[0]][target].resample(period).sum().plot(x_compat=True,figsize=figsize, grid=True,linewidth=2.0)
    legend=[values[0]]
    for i in range(1,len(values)):
        df[df[feature]==values[i]][target].resample(period).sum().plot(x_compat=True,figsize=figsize,ax=ax, grid=True, linewidth=linewidth,title='{} sum per {} per day'.format(target,feature))
        legend.append(values[i])

    if len(values) > 1:
        ax.lines[1].set_linestyle(":")
    ax.lines[0].set_linestyle("--")
    pls.legend(legend, bbox_to_anchor=(1.2, 0))
    pls.show()
    
def plot_ntop_categorical_values_means(df,feature,target,nbr_max=4,figsize=(20,4),linewidth=2.0,period="1d"):
    
    values=df[feature].value_counts()[:nbr_max].index.to_list()
    if  len(values)==0:
        return
    ax=df[df[feature]==values[0]][target].resample(period).mean().plot(x_compat=True,figsize=figsize, grid=True,linewidth=linewidth )
    legend=list()
    legend.append(values[0])
    for i in range(1,len(values)):
        df[df[feature]==values[i]][target].resample(period).mean().plot(x_compat=True,figsize=figsize,ax=ax, grid=True, linewidth=linewidth,title='{} % mean per {} per day'.format(target,feature))
        legend.append(values[i])

    if len(values) > 1:
        ax.lines[1].set_linestyle(":")
    ax.lines[0].set_linestyle("--")
    pls.legend(legend,bbox_to_anchor=(1.2, 0))
    pls.show()
    
    
#=====================#=====================#=====================#=====================
# continues
#=====================#=====================#=====================#=====================

def plot_cuts(df,feature,target,bins=[-300,-50,-20-5,-1,0,1,5,10,15,20,30,50,300], figsize=(12,6)):
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=figsize)
    pls.title('Histogram of {}'.format(feature)); 
    ax1.set_xlabel(feature); 
    ax1.set_ylabel('count');
    ax2.set_xlabel(feature); 
    ax2.set_ylabel(target);
    df.groupby(pd.cut(df[feature], bins=bins))[target].count().plot(kind='bar',ax=ax1)
    df.groupby(pd.cut(df[feature], bins=bins))[target].mean().plot(kind='bar',ax=ax2)
    pls.show()  
    
#=====================#=====================#=====================#=====================
# categorical 
#=====================#=====================#=====================#=====================

import seaborn as sns

def plot_stats(df,feature,target,horizontal_layout=True, max_nbr=20):
    end=max_nbr
    cat_count = df[feature].value_counts()
    cat_count = pd.DataFrame({feature: cat_count.index,'Count ': cat_count.values})
    cat_count.sort_values(by='Count ', ascending=False, inplace=True)

    cat_perc = df[[feature, target]].groupby([feature],as_index=False).mean()
    cat_perc=pd.merge(cat_perc,cat_count,on=feature)
    cat_perc.sort_values(by='Count ', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = pls.subplots(nrows=2, figsize=(12,14))
        
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Count ",order=cat_count[feature][:max_nbr],data=cat_count[:max_nbr])
    s.set_xticklabels(s.get_xticklabels(),rotation=90)   
    
    s = sns.barplot(ax=ax2, x = feature, y=target, order=cat_perc[feature][:max_nbr], data=cat_perc[:max_nbr])  
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
        
    pls.ylabel('Percent ', fontsize=10)
    pls.tick_params(axis='both', which='major', labelsize=10)

    pls.show();

def plot_melt(df,feature,target1,target2,end=20):
    
    cat_count = df[feature].value_counts()
    cat_count = pd.DataFrame({feature: cat_count.index,'Count ': cat_count.values})
    cat_count.sort_values(by='Count ', ascending=False, inplace=True)

    cat_perc = df[[feature, target1]].groupby([feature],as_index=False).mean()
    cat_perc=pd.merge(cat_perc,cat_count,on=feature)
    #cat_perc.sort_values(by='Count ', ascending=False, inplace=True)

    cat_perc2 = df[[feature, target2]].groupby([feature],as_index=False).mean()
    cat_perc=pd.merge(cat_perc,cat_perc2,on=feature)   
    cat_perc.sort_values(by='Count ', ascending=False, inplace=True)
    cat_perc=cat_perc[:end]
    
    data_melted = pd.melt(cat_perc[[feature,target1,target2]], id_vars=feature,\
                           var_name="source", value_name="value_numbers")
    
    fig, (ax1, ax2) = pls.subplots(ncols=2, figsize=(12,6))
    sns.set_color_codes("pastel") 
    s = sns.barplot(ax=ax1, x = feature, y="Count ",order=cat_count[feature][:end],data=(cat_count[:end]))
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
  
    s = sns.barplot(ax=ax2, x = feature, y="value_numbers",hue="source", order=data_melted[feature][:min(end,cat_count.shape[0])],data=(data_melted))
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    pls.tick_params(axis='both', which='major', labelsize=10)
    pls.legend(bbox_to_anchor=(2, 0))
    pls.show();
    
#==========================================#=====================#=====================
# nan
#==========================================#=====================#=====================

def plot_na(df):
    
    pls.style.use('seaborn-talk')

    fig = pls.figure(figsize=(18,6))
    miss = pd.DataFrame((df.isnull().sum())*100/df.shape[0]).reset_index()

    ax = sns.pointplot("index",0,data=miss)
    pls.xticks(rotation =90,fontsize =7)
    pls.title("Missed data")
    pls.ylabel(" %")
    pls.xlabel("Features")
    
def print_na(df,max_row=20):
    mdf=missing_values_table(df)
    if mdf.shape[0]:
        print(missing_values_table(df).head(max_row))
    else:
        print('No missed values in dataframe ')
    
#=====================#=====================#=====================#=====================
# correlations
#=====================#=====================#=====================#=====================
    
def corr(df,maxnbr=20,figsize=(20,20)):
    #
    # plots correlation heatmap for all numerical features
    #
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    l=df.select_dtypes(include=numerics).columns.to_list()
    corr_(df,l,figsize=figsize)
        
def corr_(df,features,figsize=(20,20),maxnbr=60):
    #
    #plots correlation heatmap for features from the list
    #
    if len(features)>=2:
        fig, ax = pls.subplots(1,1,figsize=figsize)
        sns.heatmap(df[features[:maxnbr]].corr(), cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
        pls.title('Correlation Heatmap')
        pls.show()
        
def corr__(df,features_x,features_y,figsize=(20,20)):
    #
    #plots correlation heatmap between features from  two lists
    #
    c=features_x.copy()
    c.extend(features_y)
    fig, ax = pls.subplots(1,1,figsize=figsize)
    sns.heatmap(df[c].corr()[features_y].T[features_x], cmap = pls.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    pls.title('Correlation Heatmap')
    pls.show()    
    
    
#=====================#=====================#=====================#=====================
# cramers V
#=====================#=====================#=====================#=====================

import scipy.stats as ss
import itertools
import seaborn as sns

    #Theil’s U, conditional_entropy (no symetrical)
    #https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    #https://github.com/shakedzy/dython/blob/master/dython/nominal.py

def plot_cramer_v_corr(df,max_features=20,figsize=(10,10)):
    # plot features correlation (Theil’s U, conditional_entropy) heatmap
    #max_features max features to display
    #features are selected automaticly - categorical or binary  
    #features with too many different values are ignored
    fig, ax = pls.subplots(1,1,figsize=figsize)
    __cramer_v_corr(df.loc[:,df.apply(pd.Series.nunique) < df.shape[0]/2],ax,10)
    pls.show()    
    
    
def cramer_v_corr(df,categoricals,figsize=(10,10)):
    
    # plot categorical or binary features specifyed in categoricals list 
    # correlation (Theil’s U, conditional_entropy)  heatmap   
    
    fig, ax = pls.subplots(1,1,figsize=figsize)

    correlation_matrix = pd.DataFrame(
        np.zeros((len(categoricals), len(categoricals))),
        index=categoricals,
        columns=categoricals
    )

    for col1, col2 in itertools.combinations(categoricals, 2):
        idx1, idx2 = categoricals.index(col1), categoricals.index(col2)
        correlation_matrix.iloc[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
        correlation_matrix.iloc[idx2, idx1] = correlation_matrix.iloc[idx1, idx2]

    ax = sns.heatmap(correlation_matrix, annot=True, ax=ax); 
    ax.set_title("Cramer V Correlation between Variables");
    pls.show() 

    
#=====================#=====================#=====================#=====================
# report
#=====================#=====================#=====================#=====================

def print_features(df,target=None,sorted_features=[]):
    
    features = sorted_features if len(sorted_features)>0 else list(set(df.columns.to_list()))

    _,tg_cardinality,_ = get_feature_info(df,target)
    
    for feature in features:
                                                                   
        if feature==target:
            continue
                                                                   
        print('\n ')
        display(HTML("<hr>"))
        display(HTML("<h3 align=\"center\">{}</h3>".format(feature)))
        print('\n ') 
                                                                   
        feature_type,cardinality,missed = get_feature_info(df,feature)

        info = pd.DataFrame(
        index=['Type :' ,'Distinct count :', 'Missed %:'],
        columns=[' '])       
        info[' ']=[feature_type,cardinality,missed ]
        print(info.head())
        print('\n ') 
        
        if feature_type=='Categorical' or feature_type=='Boolean':
                                                                   
            if cardinality > df.shape[0]/2.0 :
                print("Too many values to plot ")
            elif df[feature].isnull().values.all() :
                print("All values are null")
            elif cardinality<2:
                print("Zero variance")
                                                                   
            else:
                plot_stats(df,feature,target,30)
                                                                   
                #count of records with feature=value per day
                if target != None and  df.index.dtype==np.dtype('datetime64[ns]'):
                    display(HTML("<h3 align=\"center\">Top {} count per day</h3>".format(feature)))
                    plot_ntop_categorical_values_counts(df,feature,target,4)
                
                    #mean of target for records with feature=value per day
                    display(HTML("<h3 align=\"center\">{} mean per day </h3>".format(target)))
                    plot_ntop_categorical_values_means(df,feature,target,4)
                
        elif feature_type=='Numeric':
            #pairwise_feature_sum_per_day(df1,df2,feature)
            #pairwise_feature_mean_per_day(df1,df2,feature)
            if cardinality<=25:
                plot_stats(df,feature,target,30)
            else:
                #df[feature].hist() 
                fig,ax = pls.subplots(1, 2,figsize=(16, 5))
                sns.distplot(df[feature],kde=True,ax=ax[0]) 
                ax[0].axvline(df[feature].mean(),color = "k",linestyle="dashed",label="MEAN")
                ax[0].legend(loc="upper right")
                ax[0].set_title('Skewness = {:.4f}'.format(df[feature].skew()))
                sns.boxplot(df[feature],color='blue',orient='h',ax=ax[1])
                pls.show()
                
                if  target !=None and tg_cardinality < 10:
                    fig,ax = pls.subplots(1, 2,figsize=(16, 5))
                    ax[0].scatter(df[feature], df[target], marker='.', alpha=0.7, s=30, lw=0,  edgecolor='k')
                    ax[0].xlabel("feature")
                    ax[0].ylabel("target")
                    sns.violinplot(x=target, y=feature, data=df, ax=ax[1], inner='quartile')
                    pls.show()    
        else:
                                                                   
            print("Time column skip plotting ")

def do_eda(df,target,ignore=[],nbrmax=20,full=True,figsize=(14,7),linewidth=2):
    
    #detect time columns
    df = df.apply(lambda col: pd.to_datetime(col, errors='ignore') 
              if col.dtypes == object else col, axis=0)
    
    header('Missed values' )
    print_na(df)
                      
    #print shap values for each frame predicting targed
    feature_type,cardinality,missed = get_feature_info(df,target)
    
    if feature_type=='Numeric' :#and cardinality==2:
        header('Shap values')

        sorted_features=plot_shap(df,target,ignore=ignore,nbrmax=nbrmax)[:nbrmax]
    else:
        sorted_features=[]

    # if dataframe has timedate index - plot time series
    if target !=None and  df.index.dtype==np.dtype('datetime64[ns]') :
        header('Time series' )
        ax=df[target].resample('1d').mean().plot( grid=True,x_compat=True,figsize=figsize,linewidth=linewidth)
        pls.legend(' {} mean per day '.format(target))
        pls.show() 
    
    header('Features' )
    print_features(df,target,sorted_features)
     
    if full:

        #for numeric variables only
        header('Pearson correlations' )     
        corr(df,nbrmax) 

        #correlations of categorical variables
        header('Cramers V staticstics' )    
        #third parameter max features to display
        plot_cramer_v_corr(df,nbrmax)      
    return sorted_features
