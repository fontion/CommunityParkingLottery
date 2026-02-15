import os
import pandas as pd
import shutil
import numpy as np
from datetime import date
from main import all_house, all_number

def parse_txt(exchg_txt): # 讀取互換車格
    with open(exchg_txt,'rt',encoding='utf8') as f:
        contents = f.read().split('\n')
    pairs = []
    for content in contents:
        if bool(content) and content[0]!='#':
            c = content.split(',')
            m1 = int(c[0])
            m2 = int(c[1].strip())
            pairs.append((m1, m2))
    return pairs

def exchange_house_and_output1(xls_name, col, pairs, sheet_name=None): # 處理未包含組別的抽籤結果
    xls_path = os.path.join('抽籤結果',xls_name)
    if sheet_name is None:
        df = pd.read_excel(xls_path, index_col=0)
    else:
        df = pd.read_excel(xls_path, index_col=0, sheet_name=sheet_name)
    assert col in df.columns, f'{xls_name}未包含{col}紀錄'
    if '戶號' in xls_name:
        assert set(df[col])==set(all_house), f'{col}記錄未包含所有戶號，請檢查: {xls_name}'
    elif '門牌' in xls_name:
        assert set(df[col])==set(all_number), f'{col}記錄未包含所有門牌，請檢查: {xls_name}'
    # 交換車格
    for (m1, m2) in pairs:
        h1 = df.at[m1, col]
        h2 = df.at[m2, col]
        df.at[m1, col] = h2
        df.at[m2, col] = h1
    # 輸出結果
    xls_path_new = xls_path[:-5] + '&交換車格.xlsx'
    if sheet_name is None:
        df.to_excel(xls_path_new, index=True)
    else:
        if os.path.isfile(xls_path_new):
            with pd.ExcelWriter(xls_path_new, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        else:
            df.to_excel(xls_path_new, sheet_name=sheet_name, index=True)

def exchange_house_and_output2(xls_name, col, pairs, sheet_name): # 處理有分組的抽籤結果，需進一步考慮交換車格後的組別變化
    xls_path = os.path.join('抽籤結果',xls_name)
    df = pd.read_excel(xls_path, index_col=[0,1], sheet_name=sheet_name)
    assert col in df.columns, f'{xls_name}未包含{col}紀錄'
    if '戶號' in xls_name:
        assert set(df[col])==set(all_house), f'{col}記錄未包含所有戶號，請檢查: {xls_name}'
    elif '門牌' in xls_name:
        assert set(df[col])==set(all_number), f'{col}記錄未包含所有門牌，請檢查: {xls_name}'
    # 交換車格
    for pair in pairs:
        df = interchange_parking_space(df, pair)
    # 若最後一組是車格不變的組，重新排序
    if df.iat[-1,0]==df.iat[-1,1]:
        gp_loc, gp_no = df.index.get_loc_level(df.index[-1][0], level=0)
        arg = np.argsort(gp_no)
        if isinstance(gp_loc, np.ndarray) and gp_loc.dtype==bool:
            gp_ind = np.nonzero(gp_loc)[0]
            assert np.array_equal(gp_ind, np.arange(gp_ind[0], gp_ind[-1]+1)), 'Sanity check failed! Group {} is not contiguous.'.format(df.index[-1][0])
            gp_loc = slice(gp_ind[0], gp_ind[-1]+1)
        else:
            gp_ind = np.arange(gp_loc.start, gp_loc.stop, gp_loc.step)
        ind = np.r_[:gp_loc.start, gp_ind[arg]]
        df = df.iloc[ind].copy()

    # 輸出結果
    xls_path_new = xls_path[:-5] + '&交換車格.xlsx'
    if os.path.isfile(xls_path_new):
        with pd.ExcelWriter(xls_path_new, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=True)
    else:
        df.to_excel(xls_path_new, sheet_name=sheet_name, index=True)

def interchange_parking_space(df, pair):
    loc1, gp1 = df.index.get_loc_level(pair[0], level=1)
    loc2, gp2 = df.index.get_loc_level(pair[1], level=1)
    if np.nonzero(loc1)[0][0] > np.nonzero(loc2)[0][0]: # 讓index順序較前的做為交換的車格1
        loc1, loc2 = loc2, loc1
        gp1, gp2 = gp2, gp1
        pair = (pair[1], pair[0])
    # get relationships
    eq1 = df.loc[loc2,'去年'].iat[0]==df.loc[loc1,'今年'].iat[0] # 今年抽中第一個車格的住戶，交換回相同的車格
    eq2 = df.loc[loc1,'去年'].iat[0]==df.loc[loc2,'今年'].iat[0] # 今年抽中第二個車格的住戶，交換回相同的車格
    gp1_loc, gp1_no = df.index.get_loc_level(gp1[0],0) # gp1_loc應該為slice object, gp1_no為index object
    gp2_loc, gp2_no = df.index.get_loc_level(gp2[0],0)
    if isinstance(gp1_loc, np.ndarray) and gp1_loc.dtype==bool:
        loc = np.nonzero(gp1_loc)[0]
        assert np.array_equal(loc, np.arange(loc[0], loc[-1]+1)), 'Sanity check failed! Group {} is not contiguous.'.format(gp1[0])
        gp1_loc = slice(loc[0], loc[-1]+1)
    if isinstance(gp2_loc, np.ndarray) and gp2_loc.dtype==bool:
        loc = np.nonzero(gp2_loc)[0]
        assert np.array_equal(loc, np.arange(loc[0], loc[-1]+1)), 'Sanity check failed! Group {} is not contiguous.'.format(gp2[0])
        gp2_loc = slice(loc[0], loc[-1]+1)
    assert gp1_loc.step is None and gp2_loc.step is None, 'Unexpect grouping' # 同組的車格應該會集结在一起，若不是，代表分組方式不符合預期
    gp1_size = gp1_no.size # 取得第一個車格所在的組別的車格數量
    gp2_size = gp2_no.size # 取得第二個車格所在的組別的車格數量
    exist_unchanged_group = df.iat[-1,0]==df.iat[-1,1]
    # 交換車格
    h1 = df.at[(gp1[0], pair[0]), '今年']
    h2 = df.at[(gp2[0], pair[1]), '今年']
    df.at[(gp1[0], pair[0]), '今年'] = h2
    df.at[(gp2[0], pair[1]), '今年'] = h1
    if gp1!=gp2: # 為不同組別，交換後會合併成同一組別
        assert not eq1 and not eq2, '不同組別的車格交換後不應該有車格不變的情況'
        if not exist_unchanged_group or (gp1[0]!=df.index[-1][0] and gp2[0]!=df.index[-1][0]): # 確保二個車格皆不在最後一組，或最後一組不為機車格不動的組別
            # 車格1, 2平移到整組的最下面
            gp1_new_order = np.roll(np.arange(gp1_loc.start, gp1_loc.stop, gp1_loc.step), gp1_size-np.nonzero(gp1_no==pair[0])[0][0]-1) # 車格1平移到整組的最下面
            gp2_new_order = np.roll(np.arange(gp2_loc.start, gp2_loc.stop, gp2_loc.step), gp2_size-np.nonzero(gp2_no==pair[1])[0][0]-1) # 車格2平移到整組的最下面
            # 車格2的組別併入車格1
            codes = df.index.codes[0].copy()
            codes[gp2_loc] = codes[gp1_loc][0]
            df.index = df.index.set_codes(codes, level=0)
            arg = np.r_[:gp1_loc.start, gp1_new_order, gp2_new_order, gp1_loc.stop:gp2_loc.start, gp2_loc.stop:df.shape[0]]
            df_new = df.iloc[arg].copy()
        else: # 第二個車格在最後一組，且最後一組為機車格不動的組別
            assert gp2[0]==df.index[-1][0], '第二個車格的組別應為最後一組，但實際上不是，請檢查程式邏輯'
            codes = df.index.codes[0].copy()
            codes[loc2] = codes[loc1] # 車格2移到車格1的組別
            df.index = df.index.set_codes(codes, level=0)
            no2_loc = np.nonzero(loc2)[0][0]
            gp1_new_order = np.roll(np.arange(gp1_loc.start, gp1_loc.stop, gp1_loc.step), gp1_size-np.nonzero(gp1_no==pair[0])[0][0]-1) # 車格1平移到整組的最下面
            arg = np.r_[:gp1_loc.start, gp1_new_order, no2_loc, gp1_loc.stop:gp2_loc.start, gp2_loc.start:no2_loc, no2_loc+1:gp2_loc.stop]
            df_new = df.iloc[arg].copy()
    else:
        if eq1 and eq2:
            assert gp1==gp2 and set(gp1_no)==set(pair), '同一組別的車格交換後，兩個車格都不變，應該要在同一組別內，且該組別只有這兩個車格'
            if not exist_unchanged_group:
                # 最後一組非機車格不動的組別，將目前的組別移到最後一組
                arg = np.r_[:gp1_loc.start, gp1_loc.stop:df.shape[0], gp1_loc]
                df_new = df.iloc[arg].copy()
                level0 = df_new.index.levels[0].values
                level0[level0==gp1[0]] = round(level0.max())+1
                df_new.index = df_new.index.set_levels(level0, level=0)
            else:
                # 最後一組為機車格不動的組別，將目前的組別移到最後，與最後一組合併成同一組別
                codes0 = df.index.codes[0].copy()
                codes0[gp1_loc] = codes0[-1]
                df.index = df.index.set_codes(codes0, level=0)
                arg = np.r_[:gp1_loc.start, gp1_loc.stop:df.shape[0], gp1_loc]
                df_new = df.iloc[arg].copy()
        elif not eq1 and not eq2:
            if exist_unchanged_group and gp1[0]==df.index[-1][0]: # 照理應該不會發生，因為不動的都是卸任管委自選的車位，不太可能自己選了又跑去跟其他卸任管委選的車位交換
                # 將最後一組編號+1，交換的車格保留在原來的組別，其他移到新的最後一組
                level0 = np.full(df.index.levels[0].size+1, gp1[0]+1) # 新增一組
                level0 = level0.astype(df.index.levels[0].dtype) # level0 dtype is int64, 但有可能原本levels[0]的dtype是float64
                level0[:-1] = df.index.levels[0].values
                df.index = df.index.set_levels(level0, level=0)
                lg = np.logical_not(np.isin(gp1_no, pair))
                codes0 = df.index.codes[0].copy()
                codes0[gp1_loc][lg] = level0.size-1 # 未交換的車格移到新的最後一組
                df.index = df.index.set_codes(codes0, level=0)
                gp1_ind = np.arange(gp1_loc.start, gp1_loc.stop, gp1_loc.step)
                arg = np.r_[:gp1_loc.start, gp1_ind[~lg], gp1_ind[lg]]
                df_new = df.iloc[arg].copy()
            else: # 同組內互換車位，且新車位與去年皆不同
                ix = np.nonzero(np.isin(gp1_no, pair))[0]
                ix1 = np.r_[ix[0], ix[1]+1:gp1_size, 0:ix[0]]
                ix2 = np.r_[ix[0]+1:ix[1]+1]
                assert ix1.size + ix2.size == gp1_size, '交換車格索引計算錯誤'
                p = 0
                while round(gp1[0], -p)!=gp1[0]:
                    p -= 1
                p -= 1
                level0 = np.full(df.index.levels[0].size+1, round(gp1[0]+2*10**p, -p)) # 新增一組(如本來1，新增一組1.2)
                level0[:-1] = df.index.levels[0].values
                level0[level0==gp1[0]] = round(gp1[0] + 10**p, -p) # 本來的組別數字，如1，改為1.1
                df.index = df.index.set_levels(level0, level=0)
                codes0 = df.index.codes[0].copy()
                codes0[gp1_loc][ix2] = level0.size-1
                df.index = df.index.set_codes(codes0, level=0)
                gp1_ind = np.arange(gp1_loc.start, gp1_loc.stop, gp1_loc.step)
                arg = np.r_[:gp1_loc.start, gp1_ind[ix1], gp1_ind[ix2], gp1_loc.stop:df.shape[0]]
                df_new = df.iloc[arg].copy()
        else:
            if not exist_unchanged_group:
                level0 = np.full(df.index.levels[0].size+1, round(df.index.levels[0].max())+1) # 新增一組
                level0 = level0.astype(df.index.levels[0].dtype) # level0 dtype is int64, 但有可能原本levels[0]的dtype是float64
                level0[:-1] = df.index.levels[0].values
                df.index = df.index.set_levels(level0, level=0)
            if eq1:
                eq_no = pair[1]
            else: # eq2:
                eq_no = pair[0]
            codes0 = df.index.codes[0].copy()
            codes0[gp1_loc][gp1_no==eq_no] = np.argmax(df.index.levels[0])
            df.index = df.index.set_codes(codes0, level=0)
            gp1_ind = np.arange(gp1_loc.start, gp1_loc.stop, gp1_loc.step)
            arg = np.r_[:gp1_loc.start, gp1_ind[gp1_no!=eq_no], gp1_loc.stop:df.shape[0], gp1_ind[gp1_no==eq_no]]
            df_new = df.iloc[arg].copy()
    
    # sanity check
    for gp in df_new.index.get_level_values(0).unique():
        loc = np.nonzero(df_new.index.get_level_values(0)==gp)[0]
        assert np.array_equal(loc, np.arange(loc[0], loc[-1]+1)), 'Sanity check failed! Group {} is not contiguous.'.format(gp)
        if df_new.iat[loc[0],0]!=df_new.iat[loc[0],1]: # 這組不是機車格不動的組別
            arg = np.roll(loc, 1)
            assert np.array_equal(df_new.iloc[loc,0].to_numpy(), df_new.iloc[arg,1].to_numpy()), 'Sanity check failed!'
        else:
            assert df_new.iloc[loc,0].equals(df_new.iloc[loc,1]), 'Sanity check failed!'
    return df_new

if __name__=='__main__':
    txt_name = '互換車格.txt'
    txt_path = os.path.join('初始化設定',txt_name)
    pairs = parse_txt(txt_path)
    if pairs:
        print('讀取{}組互換車格'.format(len(pairs)))

        # 開始交換車格
        this_year = date.today().year
        for xls_name in ('歷年機車位(戶號).xlsx', '歷年機車位(門牌).xlsx'):
            exchange_house_and_output1(xls_name, this_year, pairs)
        
        col = '今年'
        for xls_name in (f'{this_year}抽籤結果(戶號).xlsx', f'{this_year}抽籤結果(門牌).xlsx'):
            exchange_house_and_output1(xls_name, col, pairs, sheet_name='依車格排序')
            exchange_house_and_output2(xls_name, col, pairs, sheet_name='依組別排序')
    else:
        print('未讀取到互換車格，請確認檔案:', txt_path)