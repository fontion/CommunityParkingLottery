"""
Local replacement for the Google Colab notebook.
Run with: python app.py
Then open http://localhost:5000 in your browser.

Requirements: pip install flask pandas openpyxl
"""

import os
import subprocess
import pandas as pd
import json
import re
from datetime import date
from flask import Flask, jsonify, send_file, render_template_string

app = Flask(__name__)

BASE_PATH = os.getcwd()
RESULT_FOLDER = os.path.join(BASE_PATH, '抽籤結果')
FIXED_FILE_PATH = os.path.join(BASE_PATH, '初始化設定', '預留車格(戶號).txt')
this_year = date.today().year


# ---------------------------------------------------------------------------
# Backend logic (was run_main_and_load_results_v25 in the notebook)
# ---------------------------------------------------------------------------

def run_main_and_load_results():
    try:
        fixed_slots = []
        if os.path.exists(FIXED_FILE_PATH):
            with open(FIXED_FILE_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                # match = re.search(r'#\s*卸任管委\s*([\s\S]*)', content)
                # if match:
                #     lines = match.group(1).split('\n')
                #     for line in lines:
                #         slot_match = re.match(r'(\d+):', line.strip())
                #         if slot_match:
                #             fixed_slots.append(int(slot_match.group(1)))
            for line in content.splitlines():
                slot_match = re.match(r'(\d+):', line.strip())
                if slot_match:
                    fixed_slots.append(int(slot_match.group(1)))

        # Run the lottery script
        process = subprocess.run(
            ['python', 'main.py'],
            cwd=BASE_PATH,
            capture_output=True,
            text=True
        )
        door_name = f'{this_year}抽籤結果(門牌).xlsx'
        house_name = f'{this_year}抽籤結果(戶號).xlsx'
        door_path = os.path.join(RESULT_FOLDER, door_name)
        house_path = os.path.join(RESULT_FOLDER, house_name)

        df_door = pd.read_excel(door_path)
        df_house = pd.read_excel(house_path)

        return {
            "status": "success",
            "fixed_slots": fixed_slots,
            "door_data": df_door.fillna("").to_dict(orient='records'),
            "door_cols": df_door.columns.tolist(),
            "house_data": df_house.fillna("").to_dict(orient='records'),
            "house_cols": df_house.columns.tolist(),
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Flask routes — replace Colab kernel callbacks with plain HTTP endpoints
# ---------------------------------------------------------------------------

@app.route('/api/run', methods=['POST'])
def api_run():
    """Replaces: google.colab.kernel.invokeFunction('notebook.run_main_script', ...)"""
    result = run_main_and_load_results()
    return jsonify(result)


@app.route('/api/download/<filename>', methods=['GET'])
def api_download(filename):
    """Replaces: google.colab.kernel.invokeFunction('notebook.download_file', ...)
                 and files.download(path)"""
    # Whitelist only the two expected output files for safety
    allowed = {f'{this_year}抽籤結果(門牌).xlsx', f'{this_year}抽籤結果(戶號).xlsx', '歷年機車位(戶號).xlsx', '歷年機車位(門牌).xlsx'}
    if filename not in allowed:
        return jsonify({"error": "File not allowed"}), 403

    path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    return send_file(path, as_attachment=True, download_name=filename)


# ---------------------------------------------------------------------------
# Main page — same UI, but JS now calls /api/* instead of google.colab.*
# ---------------------------------------------------------------------------

HTML_PAGE = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>麗捷花園廣場-機車位抽籤系統 ({this_year})</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');

    :root {{
        --main-bg: #f8fafc;
        --white: #ffffff;
        --primary: #2563eb;
        --success: #10b981;
        --text-main: #1e293b;
        --border: #cbd5e1;
    }}

    body {{ margin: 0; padding: 20px; background: var(--main-bg); }}

    .container {{ background: var(--main-bg); padding: 25px; border-radius: 12px; font-family: 'Noto Sans TC', sans-serif; color: var(--text-main); position: relative; }}

    /* Loading overlay */
    #loadingOverlay {{
        display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255,255,255,0.8); z-index: 1000;
        flex-direction: column; align-items: center; justify-content: center;
    }}
    .spinner {{
        width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid var(--primary);
        border-radius: 50%; animation: spin 1s linear infinite; margin-bottom: 15px;
    }}
    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}

    .header-card {{
        background: var(--white); padding: 20px 30px; border-radius: 12px;
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 20px; border: 2px solid var(--border);
    }}
    .header-card h2 {{ color: #000000 !important; margin: 0; }}

    .search-bar {{
        background: var(--white) !important;
        padding: 12px 20px; border-radius: 8px;
        border: 2px solid var(--primary) !important;
        margin-bottom: 15px; display: flex; align-items: center;
    }}
    .search-input {{
        flex: 1; border: none !important; outline: none !important;
        font-size: 16px !important; margin-left: 10px;
        background: var(--white) !important; color: #000000 !important;
    }}
    .search-input::placeholder {{ color: #94a3b8; }}

    .btn {{ padding: 10px 20px; border-radius: 6px; font-weight: 700; cursor: pointer; border: none; font-size: 14px; }}
    .btn-run {{ background: var(--primary); color: white; }}
    .btn-dl  {{ background: var(--success); color: white; }}

    .dl-dropdown {{ position: relative; }}
    .dl-menu {{
        display: none; position: absolute; right: 0; top: 45px;
        background: white; border: 1px solid var(--border); border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2); z-index: 100; width: 220px;
    }}
    .dl-item {{ padding: 12px; cursor: pointer; border-bottom: 1px solid #eee; color: #333; }}
    .dl-item:hover {{ background: #f1f5f9; color: var(--primary); }}

    .tabs {{ display: flex; gap: 5px; }}
    .tab-btn {{ background: #e2e8f0; border: none; padding: 10px 20px; cursor: pointer; border-radius: 8px 8px 0 0; color: #64748b; }}
    .tab-btn.active {{ background: white; color: var(--primary); font-weight: 700; border: 2px solid var(--border); border-bottom: none; }}

    .panel {{ background: white; padding: 20px; border-radius: 0 8px 8px 8px; display: none; border: 2px solid var(--border); }}
    .panel.active {{ display: block; }}

    .table-box {{ max-height: 480px; overflow-y: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th {{ background: #f1f5f9; color: #475569; padding: 12px; position: sticky; top: 0; border-bottom: 2px solid var(--border); }}
    td {{ padding: 10px; border-bottom: 1px solid #f1f5f9; text-align: center; color: #333; }}

    .highlight-win {{ color: var(--primary); font-weight: 700; }}
    .fixed-row {{ background: #fffbeb !important; }}
    .fixed-tag {{ background: #fef08a; color: #854d0e; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 5px; font-weight: 700; }}

    #errorBox {{ display:none; background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; border-radius:8px; padding:12px 16px; margin-bottom:12px; }}
</style>
</head>
<body>

<div id="loadingOverlay">
    <div class="spinner"></div>
    <div id="loadingText" style="font-weight:700; color:var(--primary);">正在執行抽籤...</div>
    <div style="font-size:12px; color:#64748b; margin-top:5px;">請勿關閉視窗或重複點擊</div>
</div>

<div class="container">
    <div class="header-card">
        <div><h2>麗捷花園廣場-機車位抽籤系統 ({this_year})</h2></div>
        <div style="display:flex; gap:10px;">
            <button onclick="runLottery()" id="runBtn" class="btn btn-run">🚀 執行抽籤</button>
            <div class="dl-dropdown" id="dlGroup" style="display:none;">
                <button onclick="toggleDlMenu()" class="btn btn-dl">📥 下載檔案</button>
                <div id="dlMenu" class="dl-menu">
                    <div class="dl-item" onclick="initiateDownload('{this_year}抽籤結果(門牌).xlsx')">門牌公告版 (.xlsx)</div>
                    <div class="dl-item" onclick="initiateDownload('{this_year}抽籤結果(戶號).xlsx')">戶號查詢版 (.xlsx)</div>
                    <div class="dl-item" onclick="initiateDownload('歷年機車位(門牌).xlsx')">歷年門牌版 (.xlsx)</div>
                    <div class="dl-item" onclick="initiateDownload('歷年機車位(戶號).xlsx')">歷年戶號版 (.xlsx)</div>
                </div>
            </div>
        </div>
    </div>

    <div id="errorBox"></div>

    <div id="displaySection" style="display:none;">
        <div class="search-bar">
            <span style="font-size:20px;">🔍</span>
            <input type="text" id="searchInput" class="search-input"
                   placeholder="請在此輸入搜尋內容（如：A1 或 18-5）" onkeyup="doSearch()">
        </div>
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab(event,'p_d')">門牌公告版</button>
            <button class="tab-btn"        onclick="switchTab(event,'p_h')">戶號查詢版</button>
        </div>
        <div id="p_d" class="panel active"><div class="table-box"><table><thead><tr id="th_d"></tr></thead><tbody id="tb_d"></tbody></table></div></div>
        <div id="p_h" class="panel">       <div class="table-box"><table><thead><tr id="th_h"></tr></thead><tbody id="tb_h"></tbody></table></div></div>
    </div>
</div>

<script>
    let masterData = {{ door: [], house: [], doorCols: [], houseCols: [], fixedSlots: [] }};

    // -----------------------------------------------------------------------
    // Replaces: google.colab.kernel.invokeFunction('notebook.run_main_script')
    // Now uses a plain fetch POST to /api/run
    // -----------------------------------------------------------------------
    async function runLottery() {{
        const overlay = document.getElementById('loadingOverlay');
        const btn     = document.getElementById('runBtn');
        const errBox  = document.getElementById('errorBox');

        overlay.style.display = 'flex';
        btn.disabled = true;
        btn.innerText = '⏳ 處理中...';
        errBox.style.display = 'none';

        try {{
            const res  = await fetch('/api/run', {{ method: 'POST' }});
            const data = await res.json();

            if (data.status === 'success') {{
                masterData.door      = data.door_data;
                masterData.house     = data.house_data;
                masterData.doorCols  = data.door_cols;
                masterData.houseCols = data.house_cols;
                masterData.fixedSlots = data.fixed_slots;

                document.getElementById('displaySection').style.display = 'block';
                document.getElementById('dlGroup').style.display = 'block';
                doSearch();
            }} else {{
                errBox.textContent   = '執行失敗：' + data.message;
                errBox.style.display = 'block';
            }}
        }} catch (e) {{
            errBox.textContent   = '系統錯誤：' + e;
            errBox.style.display = 'block';
        }} finally {{
            overlay.style.display = 'none';
            btn.disabled  = false;
            btn.innerText = '重新執行';
        }}
    }}

    // -----------------------------------------------------------------------
    // Replaces: google.colab.kernel.invokeFunction('notebook.download_file')
    //           and files.download(path)
    // Now uses a plain <a> tag pointing to /api/download/<filename>
    // -----------------------------------------------------------------------
    function initiateDownload(filename) {{
        const a = document.createElement('a');
        a.href = '/api/download/' + encodeURIComponent(filename);
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        document.getElementById('dlMenu').style.display = 'none';
    }}

    function doSearch() {{
        const query = document.getElementById('searchInput').value.toLowerCase();
        const filterRows = rows => rows.filter(row =>
            Object.values(row).some(val => String(val).toLowerCase().includes(query))
        );
        renderTable('th_d', 'tb_d', masterData.doorCols,  filterRows(masterData.door),  masterData.fixedSlots);
        renderTable('th_h', 'tb_h', masterData.houseCols, filterRows(masterData.house), masterData.fixedSlots);
    }}

    function renderTable(hId, bId, cols, data, fixedSlots) {{
        document.getElementById(hId).innerHTML = cols.map(c => `<th>${{c}}</th>`).join('');
        document.getElementById(bId).innerHTML = data.map(row => {{
            const slot    = parseInt(row['車格']);
            const isFixed = fixedSlots.includes(slot);
            return `<tr class="${{isFixed ? 'fixed-row' : ''}}">
                ${{cols.map(c => {{
                    let val = row[c];
                    const cls = (c === '今年') ? 'class="highlight-win"' : '';
                    if (c === '車格' && isFixed) val += '<span class="fixed-tag">固定位</span>';
                    return `<td ${{cls}}>${{val}}</td>`;
                }}).join('')}}
            </tr>`;
        }}).join('');
    }}

    function switchTab(evt, id) {{
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
        document.getElementById(id).classList.add('active');
        evt.currentTarget.classList.add('active');
    }}

    function toggleDlMenu() {{
        const m = document.getElementById('dlMenu');
        m.style.display = (m.style.display === 'none' || !m.style.display) ? 'block' : 'none';
    }}

    window.onclick = function(event) {{
        if (!event.target.matches('.btn-dl')) {{
            const m = document.getElementById('dlMenu');
            if (m) m.style.display = 'none';
        }}
    }};
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)


if __name__ == '__main__':
    print("Starting lottery app at http://localhost:5000")
    app.run(debug=True, port=5000)
