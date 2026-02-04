# 注意︰
#   本程式假設車格為數字，從1開始，且沒有跳號
#   本程式假設車格編號不大於255

import os
import pandas as pd
import numpy as np
from datetime import date

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
    with open(preserve_txt,'rt',encoding='utf8') as f:
        contents = f.read().split('\n')
    preserve = []
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
    No, houses = zip(*preserve)
    assert len(set(No))==len(No), '預留車格發現重覆車格號碼'
    assert len(set(houses))==len(houses), '預留車格發現重覆戶號'
    n = len(preserve)
    print('預留{}個車格, 共{}個車格參與抽籤'.format(n, 128-n))
    return dict(preserve)

if __name__=='__main__':
    xls_name = '歷年機車位(戶號).xlsx'
    txt_name = '預留車格(戶號).txt'
    xls_path = os.path.join('初始化設定',xls_name)
    txt_path = os.path.join('初始化設定',txt_name)
    preserve = parse_txt(txt_path)
    this_year = date.today().year
    
    update_xls = True # 更新"歷年機車位(戶號).xlsx" 並產生 "歷年機車位(門牌).xlsx"
    max_group_size = 5

    df_annual = pd.read_excel(xls_path, index_col=0)
    assert this_year-1 in df_annual.columns, '{}需包含{}年紀錄'.format(os.path.basename(xls_path), this_year-1)
    df_annual = df_annual[sorted(df_annual.columns)] # 年份由左至右遞增排序

    if this_year in df_annual.columns: # 移除本年度及之後的紀錄
        df_annual.drop(columns=df_annual.columns[df_annual.columns.get_loc(this_year):], inplace=True)
    df_annual.sort_index(inplace=True) # 按車格編號遞增排序
    df_pre = df_annual[this_year-1]
    assert set(df_pre)==set(all_house), '去年抽籤結果沒有包含所有戶號'
    original_map = {df_pre[no]:no for no in range(1, df_pre.shape[0]+1)} # 戶號: 車格

    if bool(preserve):
        new_no = np.array(list(preserve), dtype=np.uint8)
        new_house = np.array(list(preserve.values()))
        pre_no = np.vectorize(original_map.get)(new_house)
        pre_house = df_pre.loc[new_no].to_numpy()

        # 排除沒換的車格
        lg = pre_house==new_house
        if lg.any():
            keepstay_ind = new_no[lg] - 1
            keepstay_house = new_house[lg]
            new_no = new_no[~lg]
            new_house = new_house[~lg]
            pre_no = pre_no[~lg]
            pre_house = pre_house[~lg]
        else:
            keepstay_ind = np.zeros(0, dtype=np.uint8)
        
        # 排除預留但彼此互換的車格
        #   - condition1: 預留車格新舊住戶皆為卸任管委
        #   - condition2: 卸任管委去年也停在預留車格中
        lg1 = np.isin(pre_house, new_house, assume_unique=True)
        lg2 = np.isin(pre_no, new_no, assume_unique=True)
        lg = lg1 & lg2
        if lg.any():
            assert set(new_house[lg])==set(pre_house[lg]), 'Unexpect error'
            interchg_ind = new_no[lg] - 1
            interchg_house = new_house[lg]
            new_no = new_no[~lg]
            new_house = new_house[~lg]
            pre_no = pre_no[~lg]
            pre_house = pre_house[~lg]
        else:
            interchg_ind = np.zeros(0, dtype=np.uint8)
    else:
        new_no = np.zeros(0, dtype=np.uint8)
        pre_no = np.zeros(0, dtype=np.uint8)
        keepstay_ind = np.zeros(0, dtype=np.uint8)
        interchg_ind = np.zeros(0, dtype=np.uint8)


    # 取出要抽籤的車格
    preserve_ind = new_no - 1
    previous_ind = pre_no - 1
    lg = np.ones(df_pre.shape[0], dtype=bool)
    lg[keepstay_ind] = False # 車格維持不變的住戶，不參與抽籤
    lg[interchg_ind] = False # 卸任的管委互換預留的車格，不參與抽籤
    lg[preserve_ind] = False # 預留的車格不參與抽籤
    participate_ind = np.nonzero(lg)[0] # 參與抽籤的車格index
    np.random.shuffle(participate_ind) # 亂數排序車格(也就是抽籤啦！)

    lg = participate_ind.reshape(-1,1)==np.array(previous_ind).reshape(1,-1)
    idx = np.nonzero(np.any(lg, axis=1))[0] # 找出抽籤結果中預留戶本來使用的車格
    N = participate_ind.size + idx.size
    result_ind = np.zeros(N, dtype=np.uint8)
    pointer = np.arange(participate_ind.size)
    be_last = np.ones(N, dtype=bool) # 是否可做為每組最後一個 (若為保留車格則否)
    for ix in idx:
        pointer[ix:] += 1 # 前面會多插入預留的車格
        result_ind[pointer[ix]-1] = preserve_ind[previous_ind==participate_ind[ix]][0]
        be_last[pointer[ix]-1] = False
    result_ind[pointer] = participate_ind

    # 決定各組互換的車格數量(2 - max_group_size)
    group = np.zeros(df_pre.shape[0], dtype=np.uint8)
    new_house = np.empty(df_pre.shape[0], dtype=object)
    k = 0
    gp = 0
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

    df_new = df_pre.iloc[result_ind].to_frame()
    df_new.insert(1, this_year, new_house)
    df_new.insert(2, 'group', group)
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