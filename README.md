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
or 

```python
eda.report(df,target,ignore=[],nbrmax=20)
```

if target variable is set , nbrmax most important features are explored
and featers are tested against target feature

features need to be ignored can be added in ignore list 

if dataframe index is valid datetime index , time series information is added to report
 
 
to compare datasets please use:
```python   
eda.pairwise_report(df1,df2,ignore=[])
```

