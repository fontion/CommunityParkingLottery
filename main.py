# 注意︰
#   本程式假設車格為數字，從1開始，且沒有跳號
#   本程式假設車格編號不大於255(變數格式uint8)

import os
import pandas as pd
import numpy as np
from datetime import date
from itertools import chain

all_house = [f'{h}-1F' for h in range(16,34,2) if h!= 24] + \
            [f'{h}{n}-2F' for h in 'AB' for n in '3567'] + \
            [f'{h}{n}-{f}F' for h in 'AB' for n in '123567' for f in range(3,11)] + \
            [f'{h}{n}-11F' for h in 'AB' for n in '13567'] + \
            [f'{h}{n}-12F' for h in 'AB' for n in '157']

all_number = [f'{h}-1' for h in range(16,34,2) if h!= 24] + \
             [f'{h}-{f}' for h in ('18','28') for f in range(2,13)] + \
             [f'{h}-2-{v}' for h in ('18','28') for v in '123'] + \
             [f'{h}-{f}-{v}' for h in ('18','28') for f in range(3,11) for v in '12356'] + \
             [f'{h}-11-{v}' for h in ('18','28') for v in '1236'] + \
             [f'{h}-12-{v}' for h in ('18','28') for v in '26']

def house2number_mapper(): # 戶號轉門牌
    all_number = []
    d = {k:v for k,v in zip('65321','12356')}
    for house in all_house:
        code, floor = house[:-1].split('-')
        if code[0] in 'AB':
            h = '28' if code[0]=='A' else '18'
            if code[1]=='7':
                all_number.append('-'.join((h, floor)))
            else:
                all_number.append('-'.join((h, floor, d[code[1]])))
        else:
            all_number.append(house[:-1])
    mapper = {k:v for k,v in zip(all_house, all_number)}
    return mapper

def number2house_mapper(): # 門牌轉戶號
    all_house = []
    d = {k:v for k,v in zip('12356','65321')}
    for number in all_number:
        c = number.split('-')
        if len(c)==2:
            if c[1]=='1': # 1樓住戶
                all_house.append(number + 'F')
            else:
                if c[0]=='18':
                    all_house.append('B7-{}F'.format(c[1]))
                else: # c[0]=='28'
                    all_house.append('A7-{}F'.format(c[1]))
        elif len(c)==3:
            if c[0]=='18':
                all_house.append('B{}-{}F'.format(d[c[2]], c[1]))
            else: # c[0]=='28'
                all_house.append('A{}-{}F'.format(d[c[2]], c[1]))
    mapper = {k:v for k,v in zip(all_number, all_house)}
    return mapper

def parse_txt(preserve_txt): # 讀取預留車格
    preserve = []
    if os.path.isfile(preserve_txt):
        with open(preserve_txt,'rt',encoding='utf8') as f:
            contents = f.read().split('\n')
        all_houses = set(all_house)
        mapper = number2house_mapper() # 門牌轉戶號
        for content in contents:
            if bool(content) and content[0]!='#':
                c = content.split(':')
                no = int(c[0])
                house = c[1].strip().upper()
                if house not in all_house and house in all_number:
                    house = mapper[house]
                assert house in all_houses, f'非社區戶號: {content}'
                preserve.append((no, house))
    else:
        print('找不到預留車格的設定檔，抽籤將全部隨機')
    if preserve:
        No, houses = zip(*preserve)
        assert len(set(No))==len(No), '預留車格發現重覆車格號碼'
        assert len(set(houses))==len(houses), '預留車格發現重覆戶號'
    n = len(preserve)
    print('預留{}個車格, 共{}個車格參與抽籤'.format(n, 128-n))
    return dict(preserve)

def draw_lottery(df_pre, preserve, max_group_size):
    this_year = date.today().year
    original_map = {df_pre[no]:no for no in range(1, df_pre.shape[0]+1)} # 戶號: 去年停的車格號碼

    if bool(preserve):
        new_no = np.array(list(preserve), dtype=np.uint8)
        new_house = np.array(list(preserve.values()))
        pre_no = np.vectorize(original_map.get)(new_house) # 今年預留車格的住戶，去年停的車格號碼
        pre_house = df_pre.loc[new_no].to_numpy() # 今年預留的車格號碼，去年停的住戶

        # 排除沒換的車格
        lg = pre_house==new_house
        if lg.any():
            # 按車格號碼重新排序
            arg = np.argsort(new_no[lg])
            keepstay_ind = new_no[lg][arg] - 1
            keepstay_house = new_house[lg][arg]
            new_no = new_no[~lg]
            new_house = new_house[~lg]
            pre_no = pre_no[~lg]
            pre_house = pre_house[~lg]
        else:
            keepstay_ind = np.zeros(0, dtype=np.uint8)
        
        # 若卸任管委選的新車格，其舊住戶非卸任管委，而其舊車格為其他卸任管委的新車格，考量公平性需要先處理(這是最複雜的情況，需要一個一個串起來處理)
        predraw_inds = []
        predraw_houses = []
        lg1 = np.logical_not(np.isin(pre_house, new_house, assume_unique=True))
        lg2 = np.isin(pre_no, new_no, assume_unique=True)
        lg = lg1 & lg2
        if lg.any():
            pix = np.nonzero(lg)[0]
            lg = np.zeros(lg.size, dtype=bool)
            for ix in pix:
                lg[ix] = True
                predraw_ind = [new_no[ix]-1]
                predraw_house = [new_house[ix]]
                lgi = pre_no[ix]==new_no
                while lgi.any():
                    ix = np.nonzero(lgi)[0][0]
                    lg[ix] = True
                    predraw_ind.append(new_no[ix]-1)
                    predraw_house.append(new_house[ix])
                    lgi = pre_no[ix]==new_no
                predraw_ind.append(pre_no[ix]-1) # 最後一個是卸任管委的舊車格(但不在預留車格中)，後續會先抽一戶來遞補這個車格(才能決定predraw_house的最後一戶)
                # 輸出遞補車格的序列
                predraw_inds.append(predraw_ind)
                predraw_houses.append(predraw_house)
            new_no = new_no[~lg]
            new_house = new_house[~lg]
            pre_no = pre_no[~lg]
            pre_house = pre_house[~lg]

        # 排除預留但彼此互換的車格
        #   - condition1: 預留車格新舊住戶皆為卸任管委(卸任管委的新車位，是另一位卸任管委的舊車位)
        #   - condition2: 卸任管委去年也停在預留車格中(卸任管委的舊車位，是另一位卸任管委的新車位)
        lg1 = np.isin(pre_house, new_house, assume_unique=True)
        lg2 = np.isin(pre_no, new_no, assume_unique=True)
        lg = lg1 & lg2
        if lg.any():
            assert set(new_house[lg])==set(pre_house[lg]), '非預期錯誤，請檢查程式碼邏輯'
            # 按遞補順序重新排序
            pix = np.nonzero(lg)[0]
            arg = np.zeros(lg.sum(), dtype=np.uint8)
            j = 0
            for i in range(arg.size):
                arg[i] = j
                j = np.nonzero(pre_no[pix[j]]==new_no[pix])[0][0]
            interchg_ind = new_no[pix[arg]] - 1
            interchg_house = new_house[pix[arg]]
            new_no = new_no[~lg]
            new_house = new_house[~lg]
            pre_no = pre_no[~lg]
            pre_house = pre_house[~lg]
        else:
            interchg_ind = np.zeros(0, dtype=np.uint8)

        # 保留的車格不能同時是卸任管委的舊車格(final check)
        assert np.intersect1d(new_no, pre_no, assume_unique=True).size==0, '預留車格中發現卸任管委的舊車格'
    else:
        new_no = np.zeros(0, dtype=np.uint8) # 先定義好size為0的陣列(不影響後面的運算)
        pre_no = np.zeros(0, dtype=np.uint8)
        keepstay_ind = np.zeros(0, dtype=np.uint8)
        interchg_ind = np.zeros(0, dtype=np.uint8)
        predraw_inds = []


    # 取出要抽籤的車格
    preserve_ind = new_no - 1
    previous_ind = pre_no - 1
    lg = np.ones(df_pre.shape[0], dtype=bool)
    lg[keepstay_ind] = False # 車格維持不變的住戶，不參與抽籤
    lg[interchg_ind] = False # 卸任的管委互換預留的車格，不參與抽籤
    lg[preserve_ind] = False # 預留的車格不參與抽籤
    lg[list(chain.from_iterable(predraw_inds))] = False # 需先處理的預留車格不參與抽籤

    participate_ind = np.nonzero(lg)[0] # 參與抽籤的車格index
    if predraw_inds: # 先處理並預抽籤
        for i, predraw_ind in enumerate(predraw_inds):
            choiceN = np.random.choice(np.arange(1, max(1, max_group_size-len(predraw_ind))+1)) # 可能會超過max_group_size，公平起見至少仍需抽一戶來遞補卸任管委的舊車格
            inds = np.zeros(0, dtype=np.uint8)
            while inds.size < choiceN:
                inds = np.concatenate((np.random.choice(participate_ind).reshape(1), inds)) # 由後往前疊
                participate_ind = participate_ind[participate_ind!=inds[0]] # 抽出的車格不再參與抽籤
                lg = inds[0]==previous_ind
                if lg.any(): # 抽中卸任管委釋出的舊車格
                    inds = np.concatenate((preserve_ind[lg], inds)) # 需疊上卸任管委的新車格
                    preserve_ind = preserve_ind[~lg]
                    previous_ind = previous_ind[~lg]
            predraw_ind = np.concatenate((inds, predraw_ind))
            if np.unique(predraw_ind).size != predraw_ind.size:
                raise ValueError('預抽籤的車格發現重覆，請檢查程式碼邏輯')
            predraw_house = np.concatenate((df_pre.iloc[np.concatenate((inds[1:], predraw_inds[i][0:1]))].to_numpy(), np.array(predraw_houses[i]), np.array([df_pre.iat[inds[0]]]))) # 抽出的第一個車格inds[0]，其舊住戶遞補至卸任委員今年釋出的車格
            predraw_inds[i] = predraw_ind
            predraw_houses[i] = predraw_house
    
    np.random.shuffle(participate_ind) # 亂數排序車格(也就是抽籤的核心步驟啦！)

    lg = participate_ind.reshape(-1,1)==np.array(previous_ind).reshape(1,-1)
    idx = np.nonzero(np.any(lg, axis=1))[0] # 找出抽籤結果中預留戶本來使用的車格
    N = participate_ind.size + idx.size
    if N+sum([len(ind) for ind in predraw_inds]) + interchg_ind.size + keepstay_ind.size != df_pre.shape[0]:
        raise ValueError('參與抽籤的車格數量不正確，請檢查程式碼邏輯')
    result_ind = np.zeros(N, dtype=np.uint8)
    pointer = np.arange(participate_ind.size)
    be_last = np.ones(N, dtype=bool) # 是否可做為每組最後一個 (若為保留車格則否)
    for ix in idx:
        pointer[ix:] += 1 # 前面會多插入預留的車格
        result_ind[pointer[ix]-1] = preserve_ind[previous_ind==participate_ind[ix]][0]
        be_last[pointer[ix]-1] = False
    result_ind[pointer] = participate_ind

    # 決定各組互換的車格數量(2 ~ max_group_size)
    group = np.zeros(df_pre.shape[0], dtype=np.uint8)
    new_house = np.empty(df_pre.shape[0], dtype=object)
    k = 0
    gp = 1
    while k < N:
        max_group_size = min(max_group_size, N-k)
        possible_group_size = np.arange(2,max_group_size+1)[be_last[k+1:k+max_group_size]]
        group_size = np.random.choice(possible_group_size) # 隨機分配同組的車格數量
        if k+group_size==N-1: # 如果最後一組剩一個車格
            if group_size==max_group_size:
                group_size = possible_group_size[-2]
            else:
                group_size += 1
        group[k:k+group_size] = gp

        for q in range(k, k+group_size-1):
            if not be_last[q]: # 為保留車格
                new_house[q] = preserve[result_ind[q]+1]
            else:
                new_house[q] = df_pre.iat[result_ind[q+1]] # 下一順位車格的舊住戶就是目前車格的新住戶 (由下往上遞補)
        new_house[k+group_size-1] = df_pre.iat[result_ind[k]] # 最後一順位車格新住戶是第一順位車格的舊住戶

        k += group_size
        gp += 1

    # 加回卸任管委遞補與預抽籤的車格
    if predraw_inds:
        for predraw_ind, predraw_house in zip(predraw_inds, predraw_houses):
            result_ind = np.concatenate((result_ind, predraw_ind))
            group[k:k+predraw_ind.size] = gp
            new_house[k:k+predraw_ind.size] = predraw_house
            k += predraw_ind.size
            gp += 1

    # 加回預留但彼此互換的車格資料
    if interchg_ind.size > 0:
        result_ind = np.concatenate((result_ind, interchg_ind))
        group[k:k+interchg_ind.size] = gp
        new_house[k:k+interchg_ind.size] = interchg_house
        k += interchg_ind.size
        gp += 1

    # 加回沒換的車格
    if keepstay_ind.size > 0:
        result_ind = np.concatenate((result_ind, keepstay_ind))
        group[k:] = gp
        new_house[k:] = keepstay_house

    # sanity check
    assert np.unique(result_ind).size==result_ind.size==128, '抽籤結果發現重覆或遺漏的車格，請檢查程式碼邏輯'
    assert set(new_house)==set(all_house), '抽籤結果沒有包含所有戶號'
    for no, house in preserve.items():
        assert house==new_house[result_ind==no-1][0], '預留車格的住戶不正確'

    df_new = df_pre.iloc[result_ind].to_frame()
    df_new.insert(1, this_year, new_house)
    df_new.insert(2, 'group', group)
    return df_new

if __name__=='__main__':
    xls_name = '歷年機車位(戶號).xlsx'
    txt_name = '預留車格(戶號).txt'
    xls_path = os.path.join('初始化設定',xls_name)
    txt_path = os.path.join('初始化設定',txt_name)
    preserve = parse_txt(txt_path)
    this_year = date.today().year
    
    max_group_size = 5 # 每組閉環容許的機車格數量上限

    df_annual = pd.read_excel(xls_path, index_col=0)
    assert this_year-1 in df_annual.columns, '{}需包含{}年紀錄'.format(os.path.basename(xls_path), this_year-1)
    df_annual = df_annual[sorted(df_annual.columns)] # 年份由左至右遞增排序

    if this_year in df_annual.columns: # 移除本年度及之後的紀錄
        df_annual.drop(columns=df_annual.columns[df_annual.columns.get_loc(this_year):], inplace=True)
    df_annual.sort_index(inplace=True) # 按車格編號遞增排序
    df_pre = df_annual[this_year-1]
    assert set(df_pre)==set(all_house), '去年抽籤結果沒有包含所有戶號'

    # 抽籤
    df_new = draw_lottery(df_pre, preserve, max_group_size)
    df_annual[this_year] = df_new[this_year]

    # prepare output
    df_new.columns = ['去年','今年','組別']
    df_new = df_new.reset_index().set_index(['組別','車格'])
    mapper = house2number_mapper()

    output_dir = os.path.join(os.path.dirname(__file__), '抽籤結果')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    xls_result = os.path.join(output_dir, f'{this_year}抽籤結果(戶號).xlsx')
    with pd.ExcelWriter(xls_result) as writer:
        df = df_annual[[this_year-1, this_year]]
        df.columns = ['去年','今年']
        df.to_excel(writer, sheet_name='依車格排序')
        df_new.to_excel(writer, sheet_name='依組別排序')
    
    xls_result = os.path.join(output_dir, f'{this_year}抽籤結果(門牌).xlsx')
    with pd. ExcelWriter(xls_result) as writer:
        df = df_annual[[this_year-1, this_year]].map(mapper.get)
        df.columns = ['去年','今年']
        df.to_excel(writer, sheet_name='依車格排序')
        df_new.map(mapper.get).to_excel(writer, sheet_name='依組別排序')

    xls_path_new = os.path.join(output_dir, xls_name)
    df_annual.to_excel(xls_path_new)
    if xls_path_new[-9:]=='(戶號).xlsx':
        df_annual.map(mapper.get).to_excel(xls_path_new[:-9] + '(門牌).xlsx')