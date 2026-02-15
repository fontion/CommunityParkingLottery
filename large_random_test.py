import random
import pandas as pd
import numpy as np
from main import all_house, draw_lottery
from exchange import interchange_parking_space
from pandas.testing import assert_frame_equal
from tqdm import tqdm
import sys

def random_test():
    # 隨機產生去年的抽籤結果
    random.shuffle(all_house)
    df_pre = pd.Series(index=range(1,129), data=all_house)
    # 隨機產生預留車格清單
    random.shuffle(all_house)
    preserve = {
        76: all_house[0],
        77: all_house[1],
    }
    allno = list(range(1,76)) + list(range(78,129))
    random.shuffle(allno)
    preserve.update({
        no: house for no, house in zip(allno[2:9], all_house[2:9])
    })
    max_group_size = 5
    df_new = draw_lottery(df_pre, preserve, max_group_size)
    return df_new

if __name__=='__main__':
    test_exchange_times = 30 # 最高只能設64
    testN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    for i in tqdm(range(testN)):
        df_new = random_test()
        if test_exchange_times > 0:
            df_new.columns = ['去年','今年','組別']
            df_new.index.name = '車格'
            df_new = df_new.reset_index().set_index(['組別','車格'])
            idx = list(range(1,129))
            for _ in range(test_exchange_times):
                pair = random.sample(idx, 2)
                idx.remove(pair[0])
                idx.remove(pair[1])
                df_pre = df_new.copy()
                df_new = interchange_parking_space(df_new, pair)

                # sanity check
                df_pre = df_pre.reset_index().drop(columns='組別').set_index('車格')
                h1 = df_pre.at[pair[0], '今年']
                h2 = df_pre.at[pair[1], '今年']
                df_pre.at[pair[0], '今年'] = h2
                df_pre.at[pair[1], '今年'] = h1
                df_aft = df_new.reset_index().drop(columns='組別').set_index('車格')
                assert df_pre.sort_index().equals(df_aft.sort_index()), 'Sanity check failed!'

    print('Passed!')