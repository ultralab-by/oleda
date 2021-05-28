# oleda

 Exploratory Data Analysis with python
 Generates reports from a pandas DataFrame.
 Helps to investigate data, finds insights in it and select features
 Allows to compare datasets
 Automatic report generation from a pandas DataFrame.

Usage/installation
------------------
to install oleda
```bash
 sudo python3 setup.py install
```
Examples
--------

to explore single dataset and create report just type:
```python
from oleda import eda
eda.report(df)
```
report contains:
- missing values statistics
- information on each feature relevant for the feature type
- pearson correlation heatmap
- Cramers V staticstics (https://stackoverflow.com/a/46498792/5863503)
- pair plot for most correlated features
- if dataframe index is valid datetime index, time series plots are added to the report


dataset can be tested against an target variable (binary (0,1) or continues):  

```python
eda.report(df,target,ignore=[],nbrmax=20)
```
nbrmax- number of most important features selected by shap are explored
features need to be ignored can be added in ignore list 
featers are tested against target feature

 
to compare datasets please use:
```python   
eda.pairwise_report(df1,df2,ignore=[])
```


to run feature selection tests and print table to compare results:
```python 
feature_selection.run_all(df,target,max_nbr_features_to_select)
```

to create 2nd order interactions plots:
```python 
eda.interactions2x(df,maxnbr=6)
```
maxnbr - max feature values to test and plot (affects speed)

to create 3nd order interactions plots:
```python 
eda.interactions3x(df,maxnbr=6)
```

