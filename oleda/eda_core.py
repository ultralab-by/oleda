import numpy as np
import matplotlib.pyplot as pls 
import pandas as pd

from IPython.display import display, HTML
import seaborn as sns

from lightgbm import LGBMClassifier,LGBMRegressor
import lightgbm as lgb
import shap
#=====================#=====================#=====================#=====================
# shap 
#=====================#=====================#=====================#=====================

def plot_shap(x, target,ignore=[],nbrmax=20):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','datetime64[ns]','m8[ns]','datetime64[ns, UTC]']

    features=x.columns.to_list()
    features.remove(target)
    #x = x.apply(lambda col: pd.to_datetime(col, errors='ignore') 
    #          if col.dtypes == object else col, axis=0)
        
    for f in x.columns.to_list():
        if isTime(x[f].dtype)or x[f].isnull().values.all() or (len(x[f].unique())>x.shape[0]/2.0 and x[f].dtype not in numerics)  and f in features:
            features.remove(f)
    features=list(set(features)-set(ignore))


    #list of categorical features
    categorical_features=x[features].select_dtypes(exclude=numerics).columns.to_list()

    #change type to categorical for lightgbm
    for c in categorical_features:
        x[c] = x[c].astype('category')
    target_type,target_cardinality,_=get_feature_info(x,target)
    binary_target=(target_type=='Numeric' and target_cardinality==2)
    if binary_target:
        clf = LGBMClassifier(
                             objective='binary'
                             ,n_estimators=100
                            , min_data_in_leaf = 10
                            , min_sum_hessian_in_leaf = 10
                            , feature_fraction = 0.9
                            , bagging_fraction = 1
                            , bagging_freq = 1                     
                            , metric='auc'
                            , learning_rate = 0.03
                            , num_leaves = 19
                            , num_threads = 2
                            , nrounds = 500 
                            )
    else:
        clf = LGBMRegressor(
                             #objective='regression'
                             n_estimators=100
                            , min_data_in_leaf = 10
                            , min_sum_hessian_in_leaf = 10
                            , feature_fraction = 0.9
                            , bagging_fraction = 1
                            , bagging_freq = 1                     
                            #, metric='auc'
                            , learning_rate = 0.03
                            , num_leaves = 19
                            , num_threads = 2
                            , nrounds = 500 
                            )
    clf.fit(x[features], x[target])#,categorical_feature=categorical_features)
    
    shap_values = shap.TreeExplainer(clf.booster_).shap_values(x[features])
    shap.summary_plot(shap_values, x[features],
                      max_display=30, auto_size_plot=True)
    
    if binary_target:
        vals= np.abs(shap_values).mean(0)
    else:
        vals= shap_values
        
    feature_importance = pd.DataFrame(list(zip(x[features].columns, sum(vals))), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    sorted_features=feature_importance['col_name'].to_list()

    X=x.copy()

    for f in categorical_features:
        X[f]=  X[f].astype(object)
        X[f]=pd.factorize(X[f])[0]     
       
    if binary_target:
        shap.summary_plot(shap_values[1], x[features])        
        
        for name in sorted_features[:nbrmax]:
            fig, ax = pls.subplots(1,1,figsize=(20,10))
            shap.dependence_plot(name, shap_values[1], X[features], display_features=x[features], interaction_index=None,ax=ax)
            pls.show()

    #restore type
    for c in categorical_features:
        x[c] = x[c].astype('object')
    return sorted_features

#=====================#=====================#=====================#=====================
# cramers V
#=====================#=====================#=====================#=====================

import scipy.stats as ss
import itertools


def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
def __cramer_v_corr(df,ax,max_features=20):

    categoricals=get_categorical(df)[:max_features]

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

    #Theil’s U, conditional_entropy (no symetrical)
    #https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    #https://github.com/shakedzy/dython/blob/master/dython/nominal.py
 #=====================#=====================#=====================#=====================
# nan
#=====================#=====================#=====================#=====================


def missing_values_table(df):
    
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    return mis_val_table   

#=====================#=====================#=====================#=====================
# html
#=====================#=====================#=====================#=====================

def header(title,sz='h2'):
    print('\n \n')
    display(HTML("<hr>"))
    display(HTML("<{} align=\"center\">{}</{}>".format(sz,title,sz)))
    print('\n  ') 
    
#=====================#=====================#=====================#=====================
# features 
#=====================#=====================#=====================#=====================

def get_feature_type(dtype):
    if dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
        return 'Numeric'
    elif isTime(dtype):
        return 'Time'
    elif dtype in [np.bool]:
        return 'Boolean'
    else:
        return 'Categorical'    
    
def get_feature_info(df,feature):
    if feature in df.columns:
        cardinality = len(df[feature].unique())
        missed= 100 * df[feature].isnull().sum() / df.shape[0]
        feature_type = get_feature_type(df[feature].dtype)
        return [feature_type,cardinality,missed]
    else:
        return "","",""
    
def isTime(dtype):
    if dtype in [np.dtype('datetime64[ns]'),np.dtype('m8[ns]')] or 'datetime' in str(dtype):   
        return True
    return False
    
def get_categorical(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','datetime64'] 
    #keep columns with % of missed less then 60
    categoricals = df.loc[:, df.isnull().mean() <= .6].select_dtypes(exclude=numerics).columns.to_list()
    #add binary columns
    bool_cols = [col for col in df.select_dtypes(include=numerics).columns.to_list() if 
               df[col].dropna().value_counts().index.isin([0,1]).all()]
    
    categoricals.extend(bool_cols)
    
    #drop columns with no variance
    categoricals=[col for col in categoricals if 
               df[col].dropna().value_counts().shape[0]>1]
    return categoricals