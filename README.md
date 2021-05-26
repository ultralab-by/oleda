# oleda
EDA for 2020s

Install oleda
```bash
 sudo python3 setup.py install
```
to run your first example

to explore single dataset :
```python
from oleda import eda
eda.report(df)
```
or to test dataset against  target variable:  

```python
eda.report(df,target,ignore=[],nbrmax=20)
```

nbrmax most important features are explored
and featers are tested against target feature

features need to be ignored can be added in ignore list 

if dataframe index is valid datetime index , time series information is added to report
 
 
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

