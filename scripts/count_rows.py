import pandas as pd
p='data/processed/sangli_labeled.csv'
for enc in ['utf-8','utf-8-sig','latin1','cp1252']:
    try:
        df=pd.read_csv(p, encoding=enc, engine='python')
        print(enc, 'rows=', len(df))
    except Exception as e:
        print(enc, 'failed:', e)
