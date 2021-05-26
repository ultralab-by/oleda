from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import pandas as pd

def isNumeric(S):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        return (S.dtype in numerics)
    
def isTime(S):
    if '[ns]'  in str(S.dtype) or S.dtype in [np.dtype('datetime64[ns]'),np.dtype('m8[ns]')] or 'datetime' in str(S.dtype) :   
        return True
    return False

    

#chi2

#Normalization: MinMaxScaler (values should be bigger than 0)
#Impute missing values: yes
def chi2_selector(df,y,kmax=16):
    
    columns=df.select_dtypes(include=[np.number])
    positive=[]
    for c in columns:
        if (df[c] >= 0).all() & np.array_equal(df[c], df[c].astype(int)):
            positive.append(c)
    
    chi_selector = SelectKBest(chi2, k=kmax)
    chi_selector.fit(df[positive], y)
    chi_support = chi_selector.get_support()
    chi_support = [True if i in positive[chi_support] else False for i in df.columns.tolist()]
    chi_feature = df.loc[:,chi_support].columns.tolist()
    print( '\n chi2 test selected features \n\t',chi_feature)
    return chi_support

# Pearson Correlation
# Linear dependens quantative vers quantative

#Normalization: no
#Impute missing values: yes
    
def cor_selector(X, y,kmax=16):
    cor_list = []
    nc=[]
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        if isNumeric(X[i]):
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
            nc.append(i)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X[nc].iloc[:,np.argsort(np.abs(cor_list))[-kmax:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in  X.columns.tolist()]
    print('\n pearson correlation selected features \n\t', cor_feature)
    return cor_support


#L1

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression,Lasso

def l1_selector(X_norm,y,rate='1.75*mean',max_features_to_select=16):
    numeric=X_norm.select_dtypes(include=[np.number]).columns.tolist()
    
    if sorted(y.dropna().unique()) == [0, 1]:
        embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1",solver='liblinear'), rate)
    else:
        embeded_lr_selector = SelectFromModel(Lasso(alpha=1.0), rate,max_features =max_features_to_select)
    embeded_lr_selector.fit(X_norm[numeric],y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_support = [True if i in np.array(numeric)[embeded_lr_support] else False for i in X_norm.columns.tolist()]
    embeded_lr_feature = X_norm.loc[:,embeded_lr_support].columns.tolist()
    print( '\n l1 selected features \n\t',embeded_lr_feature,)

    return embeded_lr_support


#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression,LinearRegression

def rfe_selector_(X_norm,y,n_features_to_select=16):
    
    numeric=X_norm.select_dtypes(include=[np.number]).columns.tolist()
    
    if sorted(y.dropna().unique()) == [0, 1]:
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=n_features_to_select, step=10)
    else:
        rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select =n_features_to_select, step=10)        
    rfe_selector.fit(X_norm[numeric], y)
    rfe_support = rfe_selector.get_support()

    rfe_support = [True if i in np.array(numeric)[rfe_support] else False for i in X_norm.columns.tolist()]
    rfe_feature = X_norm.loc[:,rfe_support].columns.tolist()
    print( '\n rfe selected features \n\t', rfe_feature)
    return rfe_support


#  Extra Trees Classifier

from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

# feature extraction
def extra_tree_selector(df,y,minimpotance=0.1):
    numeric=df.select_dtypes(include=[np.number]).columns.tolist()
    model = ExtraTreesClassifier() if sorted(y.dropna().unique()) == [0, 1] else ExtraTreesRegressor()
    model.fit(df[numeric],y)
    extra_tree_support=list(model.feature_importances_>minimpotance)
    extra_tree_support = [True if i in np.array(numeric)[extra_tree_support] else False for i in df.columns.tolist()]
    extra_tree_features = df.loc[:,extra_tree_support].columns.tolist()
    print( '\n extra tree selected features \n\t',extra_tree_features)
    return extra_tree_support



#ANOVA
#categorical-quantative

from sklearn.feature_selection import f_classif
def anova_selector(df,y,n_features_to_select=15):
    numeric=df.select_dtypes(include=[np.number]).columns.tolist()
    f_selector = SelectKBest(f_classif, k=n_features_to_select)
    f_selector.fit(df[numeric], y)
    f_support = f_selector.get_support()
    f_support = [True if i in np.array(numeric)[f_support] else False for i in df.columns.tolist()]
    f_feature = df.loc[:,f_support].columns.tolist()
    print( '\n ANOVA selected features \n\t',f_feature)
    return f_support



#Rundom forest
#Normalization: No
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

def random_forest_selector(df,y,threshold='3.25*median',max_features_to_select=16):
    numeric=df.select_dtypes(include=[np.number]).columns.tolist()
    model=RandomForestClassifier(n_estimators=100) if sorted(y.dropna().unique()) == [0, 1] else RandomForestRegressor(n_estimators=100)
    embeded_rf_selector =  SelectFromModel(model, threshold=threshold,max_features=max_features_to_select)
    embeded_rf_selector.fit(df[numeric], y)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_support = [True if i in np.array(numeric)[embeded_rf_support] else False for i in df.columns.tolist()]
    embeded_rf_feature = df.loc[:,embeded_rf_support].columns.tolist()
    print( '\n random forest selected features \n\t',embeded_rf_feature)
    return embeded_rf_support



#Lightgbm
#Normalization: No
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier,LGBMRegressor

def lightgbm_selector(df,y,threshold='4.15*mean',max_features_to_select=16):
    if sorted(y.dropna().unique()) == [0, 1] :
        lgbc=LGBMClassifier(n_estimators=50, learning_rate=0.05, num_leaves=22, colsample_bytree=0.2,
                    reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    else:
        lgbc=LGBMRegressor(n_estimators=50, learning_rate=0.05, num_leaves=22, colsample_bytree=0.2,
                    reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    embeded_lgb_selector = SelectFromModel(lgbc, threshold=threshold,max_features=max_features_to_select)#median
    embeded_lgb_selector.fit(df,y)
    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = df.loc[:,embeded_lgb_support].columns.tolist()
    print( '\n lightgbm selected features \n\t',embeded_lgb_feature)
    return embeded_lgb_support


from skrebate import ReliefF
#not for big data
def relief_selector(df,y,threshold=0.005):
    numeric=df.select_dtypes(include=[np.number]).columns.tolist()
    X_norm = MinMaxScaler().fit_transform(df[numeric])
    fs = ReliefF()
    fs.fit(X_norm,y)

    reliefF_support=list(abs(fs.feature_importances_)>threshold)
    reliefF_features=list(np.asarray(X_norm.columns.to_list(), dtype=object)[abs(fs.feature_importances_>threshold)])
    reliefF_support = [True if i in np.array(numeric)[reliefF_support] else False for i in df.columns.tolist()]
    print( '\n relieF selected features \n\t',reliefF_features)
    return reliefF_support

import shap

def shap_selector(x,y,nbrmax):
    if (sorted(y.dropna().unique()) == [0, 1]):
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
        clf = LGBMClassifier(
                            # objective='binary'
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

    clf.fit(x, y)
    
    shap_values = shap.TreeExplainer(clf.booster_).shap_values(x)
    
    vals= np.abs(shap_values).mean(0)

    feature_importance = pd.DataFrame(list(zip(x.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    sorted_features=feature_importance['col_name'].to_list()
    shap_support = [True if i in sorted_features[:nbrmax] else False for i in x.columns.to_list()]
    print( '\n shap selected features \n\t',sorted_features[:nbrmax])
    
    return shap_support


#example only
#have no sense to run all tests together
#some tests for binary data other for continues etc

def run_all(df,y,nbrmax=16):
    
    pd.set_option('display.max_rows', None)
    
    #lightgbm fails on datetime features
    df = df.apply(lambda col: pd.to_datetime(col, errors='ignore') 
              if col.dtypes == object else col, axis=0)
    
    backup={}
    cat=[]
    numeric=[]
    for c in df.columns.tolist():
        if isNumeric(df[c]):#for algo that works only with numeric columns
            numeric.append(c)
        elif not isTime(df[c]):
            backup[c]=df[c].dtype
            df[c] = df[c].astype('category')#for lightgbm
            cat.append(c)
    #no time columns        
    nt=list(set(numeric)|set( cat))
    
    if len(cat)>0:
        print('Warning all tests but LightGBM and shap are performed on numeric features only. Please encode \n' ,cat,' \n')

    tests={'Feature':nt, 
                                         'Pearson':cor_selector(df[nt],y,nbrmax), 
                                         #'Chi-2':chi2_selector(df,y,nbrmax), 
                                         'RFE':rfe_selector_(df[nt],y,nbrmax),
                                         'l1':l1_selector(df[nt],y,'mean',nbrmax),
                                         'Random Forest': random_forest_selector(df[nt],y, '1.15*median',nbrmax), 
                                         'Extra Trees Classifier': extra_tree_selector(df[nt],y,minimpotance=0.03),
                                         'ANOVA':anova_selector(df[nt],y,nbrmax),   
                                         'LightGBM': lightgbm_selector(df[nt],y,'1.15*mean', nbrmax),
                                         'shap':shap_selector(df[nt],y,nbrmax),
                                         #'relief':relief_selector(df,y,threshold=0.005)
                                        }
    
    if (sorted(y.dropna().unique()) == [0, 1]):
        tests['Chi-2']=chi2_selector(df[nt],y,nbrmax)
  
    feature_selection_df = pd.DataFrame(tests)
    
    #restore column type
    for c in cat:
        df[c] = df[c].astype(backup[c])
    
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)

    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    #print(feature_selection_df.head(nbrmax*2))
    return feature_selection_df
 