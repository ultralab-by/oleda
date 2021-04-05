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
eda.report(df,target,ignore=[])
```

to compare datasets:
```python   
eda.pairwise_report(df1,df2,ignore=[])
```

