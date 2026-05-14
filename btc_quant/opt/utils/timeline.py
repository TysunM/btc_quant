import polars as pl

INPUT = r'C:\btc_quant\data\processed\quant_data_clean.parquet'

def audit_timeline():
    print('\n[SYSTEM] Auditing Matrix Chronology...')
    df = pl.read_parquet(INPUT)
    
    print('\n[INFO] Detected Columns:', df.columns)
    
    print('\n--- FIRST 5 ROWS (HEAD) ---')
    print(df.head(5))
    
    print('\n--- LAST 5 ROWS (TAIL) ---')
    print(df.tail(5))

if __name__ == '__main__':
    audit_timeline()
