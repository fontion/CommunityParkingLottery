# 社區機車位抽籤程式
## Issues
1. 區權會時間寶貴，而實體抽籤時間冗長，且後續仰賴人工核對登載後才能公告
2. 隨機抽籤方式，結果易形成前車卡後車(詳如下面說明)，最後很多機車無法同步更換車位，每年更換車位皆形成亂象

## Description


# Functionality
1. 亂數隨機更換車格
2. 預留卸任管委停車格
3. 排除區權會當日已抽中身障車格的住戶
4. 隨機設定斷點(目前設定至多5個車格構成一組)

## Concepts of Algorithm


## Limitations
1. 亂數結果會受初始條件(去年的抽籤結果)影響，無法達到每個車格的隨機性完全相同，但不會差很多
2. 目前只有演算法，尚無前端UI，待後續管委會或住戶開發，以利往後每年抽籤

## Build environment
via [pip](https://pip.pypa.io/en/latest/user_guide/#requirements-files) (directly install the necessary packages)
```
pip install -r requirements.txt
```
via [conda](https://www.anaconda.com/docs/getting-started/working-with-conda/environments) (create an environment named lijie)
```
conda env create -f environment.yml
conda activate lijie
```

## 抽籤流程
1. 確認「初始化設定」資料夾內存在「歷年機車位(戶號).xlsx」此檔，且包含去年的記錄
2. 編輯「預留車格(戶號).txt」，身障車格待區權會現場抽籤結果出爐再填入，每行格式為「車格編號: 戶號」或「車格編號: 門牌」
3. 執行程式
   ```
   python main.py
   ```
4. 程式會自動建立「抽籤結果」資料夾，並輸出抽籤結果，包含四個檔案(假設為2026年抽籤)：
   > 2026抽籤結果(戶號).xlsx <br>
   > 2026抽籤結果(門牌).xlsx <br>
   > 歷年機車位(戶號).xlsx <br>
   > 歷年機車位(門牌).xlsx <br>
5. 將抽籤結果公告於雲端資料夾

- (戶號)的結果會將每戶以戶型代號表示，如A1-2F, B7-8F, A5-10F <br>
- (門牌)的結果會將每戶以門牌號碼表示，如28號7樓之1，會表示為28-7-1 <br>
- 店面戶均會使用門牌表示，若門牌為22號，(戶號)中會顯示為22-1F，(門牌)中會顯示為22-1 <br>
- "歷年機車位(戶號).xlsx"檔案會新增今年的抽籤結果，明年抽籤前要先放入「初始化設定」資料夾
- "2026抽籤結果(戶號).xlsx"檔案內包含兩個sheets：依車格排序、依組別排序。需要同時更換的車格會被分在同一組中
