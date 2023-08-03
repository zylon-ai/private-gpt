from glob import glob
import pandas as pd

def main():
    for tsv_file in glob('./source_documents/*.tsv'):
        csv_table=pd.read_table(tsv_file, sep='\t')
        csv_table.to_csv(tsv_file[:-4]+'.csv', index=False)

if __name__ == '__main__':
    main()  