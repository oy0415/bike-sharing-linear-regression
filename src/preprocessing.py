#ライブラリの読み込み
import numpy as np
import pandas as pd


def load_and_clean_data(file_path):
    '''
    データを読み込み、データの表示、基本情報の表示、欠損値の個数
    欠損値があれば、欠損値の除去を行う
    ※時系列データは想定していません
    '''
    try: 
        # データの読み込み
        data = pd.read_csv(file_path)
        
        # 読み込んだデータの表示
        print('\n読み込んだデータ :' + '\n')
        print(data.head(3))
        
        print('\nデータの基本情報 :' + '\n')
        data.info()
        
        # 欠損値の除去を行う
        
        # 欠損値の個数
        nan_sum = data.isna().sum().sum()
        
        print(f'\n欠損値の個数 : {nan_sum}')
        
        # 欠損値がない場合
        if nan_sum == 0:
            print('欠損値はありません')
            return data
            
        else:
            print(f'欠損値が{nan_sum}個あります。')
            print('欠損値の除去を開始します。')
            
            before_rows = len(data)
            
            # 欠損値の除去
            data_clean = data.dropna()
            
            after_rows = len(data_clean)
            removed_rows = before_rows - after_rows
            print('欠損値の除去が完了しました。')
            print(f'{removed_rows}行を削除しました。')
            
            return data_clean
    
    except Exception as e:
        print('ERROR\n', e)