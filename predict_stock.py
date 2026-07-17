import numpy as np
import pandas as pd
import requests
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from datetime import datetime

# 追蹤的港股列表 (可自由增減)
STOCKS = {
    "2800.HK": "盈富基金 (Tracker Fund)",
    "0700.HK": "騰訊控股 (Tencent)",
    "9988.HK": "阿里巴巴 (Alibaba)",
    "3690.HK": "美團 (Meituan)"
}

results = []

print("🚀 啟動【免 yfinance・底層 API 直連版】港股實時量化推演引擎...")

def fetch_stock_data_raw(ticker):
    """
    直接調用 Yahoo Finance v8 底層官方 Chart API (繞過 yfinance 的 Cookie/Crumb 阻擋)
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    
    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        raise Exception(f"Yahoo API 響應錯誤，HTTP Code: {response.status_code}")
    
    data = response.json()
    if 'chart' not in data or 'result' not in data['chart'] or data['chart']['result'] is None:
        raise Exception("Yahoo 返回了無效的 JSON 數據格式")
        
    result = data['chart']['result'][0]
    timestamps = result.get('timestamp') or []
    if not timestamps:
        raise Exception("未獲取到任何時間戳數據")
        
    indicators = result.get('indicators', {}).get('quote', [{}])[0]
    
    # 獲取各價格列表，若為 None 則回傳空列表，防止 None 導致 map 失敗
    open_prices = indicators.get('open') or []
    high_prices = indicators.get('high') or []
    low_prices = indicators.get('low') or []
    close_prices = indicators.get('close') or []
    volume_list = indicators.get('volume') or []
    
    # 安全校驗長度是否一致
    if not (len(timestamps) == len(open_prices) == len(high_prices) == len(low_prices) == len(close_prices) == len(volume_list)):
        raise Exception("Yahoo 數據長度不一致 (Data mismatch)")
        
    # 建立 DataFrame
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume_list
    }, index=pd.to_datetime(timestamps, unit='s'))
    
    # 填充缺失值 (停牌或假期造成的 None 值)
    df['Close'] = df['Close'].ffill().bfill()
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Volume'] = df['Volume'].fillna(0)
    
    # 轉換成香港時區
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Hong_Kong')
    return df

fig_all = go.Figure()

for ticker, name in STOCKS.items():
    print(f"📊 正在下載並解析 {name} ({ticker})...")
    
    # 安全沙盒下載機制
    try:
        stock_data = fetch_stock_data_raw(ticker)
    except Exception as e:
        print(f"❌ {ticker} 獲取失敗: {e}，自動跳過此股。")
        continue
        
    if stock_data.empty or len(stock_data) < 250:
        print(f"⚠️ {ticker} 數據不足，跳過。")
        continue
        
    df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # ---------------------------------------------------------
    # 1. 技術指標計算 (物理防線)
    # ---------------------------------------------------------
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA250'] = df['Close'].rolling(window=250).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['MA20'] - 2 * df['STD20']
    df['BB_Upper'] = df['MA20'] + 2 * df['STD20']
    
    # 計算 RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 波動率與偏離度
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std() * 100
    df['MA_Dev'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
    
    df.dropna(inplace=True)
    
    if len(df) < 50:
        print(f"⚠️ {ticker} 清洗數據後樣本太少，跳過。")
        continue
    
    # ---------------------------------------------------------
    # 2. AI 決策樹訓練 (精準預報未來5天動能)
    # ---------------------------------------------------------
    features = ['RSI', 'Volatility', 'MA_Dev']
    df['Future_5d_Return'] = df['Close'].pct_change(5).shift(-5)
    
    df['Label'] = 0
    df.loc[df['Future_5d_Return'] > 0.02, 'Label'] = 1
    df.loc[df['Future_5d_Return'] < -0.02, 'Label'] = -1
    
    # 排除包含 NaNs 的訓練集
    train_df = df.dropna(subset=['Future_5d_Return']).copy()
    
    if len(train_df) < 30:
        print(f"⚠️ {ticker} 可訓練樣本太少，跳過。")
        continue
        
    ai_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    ai_model.fit(train_df[features], train_df['Label'])
    
    # ---------------------------------------------------------
    # 3. 實時推演今日最新狀態
    # ---------------------------------------------------------
    latest_row = df.iloc[-1]
    current_price = round(latest_row['Close'], 2)
    
    best_buy_min = round(min(latest_row['BB_Lower'], latest_row['MA250']), 2)
    best_buy_max = round(latest_row['BB_Lower'], 2)
    
    # 用 DataFrame 包裹特徵，消除 scikit-learn 的 Feature Names 警告
    latest_features = pd.DataFrame([[latest_row['RSI'], latest_row['Volatility'], latest_row['MA_Dev']]], columns=features)
    ai_pred_proba = ai_model.predict_proba(latest_features)[0]
    
    classes = ai_model.classes_
    buy_idx = np.where(classes == 1)[0]
    buy_probability = round(ai_pred_proba[buy_idx[0]] * 100, 1) if len(buy_idx) > 0 else 0.0
    
    # 綜合評級判定
    if buy_probability >= 60 and current_price <= best_buy_max:
        rating = "💎 強烈建議分批買入 (AI與物理超賣共振)"
        rating_color = "#2ecc71"
    elif buy_probability >= 50:
        rating = "🟢 偏向多頭 (時機成熟，可試探買入)"
        rating_color = "#a3e4d7"
    elif current_price <= best_buy_max:
        rating = "🟡 到達物理支撐位 (價格便宜，但動能未轉強)"
        rating_color = "#f1c40f"
    elif latest_row['RSI'] > 65:
        rating = "🔴 短線超買 (建議分批止盈/切勿追高)"
        rating_color = "#e74c3c"
    else:
        rating = "⚪ 觀望中 (未達最佳買入區間)"
        rating_color = "#bdc3c7"
        
    results.append({
        "代號": ticker,
        "名稱": name,
        "當前收市價 (HKD)": current_price,
        "最佳買入區間 (HKD)": f"{best_buy_min} - {best_buy_max}",
        "AI 買入信心度 (%)": f"{buy_probability}%",
        "綜合操作建議": f"<span style='color:{rating_color}; font-weight:bold;'>{rating}</span>",
        "20天均線": round(latest_row['MA20'], 2),
        "14天RSI": round(latest_row['RSI'], 1)
    })

    # ---------------------------------------------------------
    # 4. 🚀 記憶體切片優化：直接切出 3 個月畫圖 (免除二次網路請求，100% 解決空白線圖)
    # ---------------------------------------------------------
    three_months_ago = stock_data.index.max() - pd.Timedelta(days=90)
    hist_3m = stock_data[stock_data.index >= three_months_ago]
    
    formatted_dates = hist_3m.index.strftime('%Y-%m-%d')
    
    fig_all.add_trace(go.Scatter(
        x=formatted_dates, 
        y=hist_3m['Close'], 
        name=f"{name} ({ticker})"
    ))

# ---------------------------------------------------------
# 5. 安全防範
# ---------------------------------------------------------
if not results:
    print("❌ 致命錯誤：所有股票數據獲取失敗！請檢查 Yahoo API 狀態。")
    exit(1)

df_report = pd.DataFrame(results)

# 調整圖表美化外觀
fig_all.update_layout(
    title="📈 近三個月核心港股走勢對比 (實時Yahoo數據)",
    xaxis_title="日期",
    yaxis_title="股價 (HKD)",
    hovermode="x unified",
    template="plotly_dark",
    paper_bgcolor="#121212",
    plot_bgcolor="#161c24",
    margin=dict(l=40, r=40, t=60, b=40)
)

chart_html = fig_all.to_html(full_html=False, include_plotlyjs='cdn')
table_html = df_report.to_html(index=False, border=0, escape=False)

# ---------------------------------------------------------
# 6. 生成網頁 index.html
# ---------------------------------------------------------
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>香港股市 AI 雙軌量化推演系統</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background-color: #0c0f12; color: #e0e6ed; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 30px 0; border-bottom: 2px solid #1f2937; }}
        h1 {{ margin: 0; color: #00ffcc; font-size: 26px; letter-spacing: 1px; }}
        .update-time {{ color: #88a0b0; font-size: 13px; margin-top: 8px; }}
        .intro-box {{ background: #161c24; padding: 20px; border-radius: 8px; margin: 20px 0; font-size: 14px; line-height: 1.6; border-left: 4px solid #00ffcc; }}
        .table-container {{ margin-top: 30px; overflow-x: auto; background: #161c24; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
        th {{ background-color: #1f2937; color: #00ffcc; font-weight: 600; padding: 12px; border-bottom: 2px solid #374151; }}
        td {{ padding: 14px 12px; border-bottom: 1px solid #1f2937; color: #cbd5e1; }}
        tr:hover {{ background-color: #1e293b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 香港股市 AI 雙軌實時量化推演系統</h1>
            <div class="update-time">每日自動更新（香港時間 HKT 收市後）: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        
        <div class="intro-box">
            💡 <b>運作原理：</b> 本系統採用「雙軌量化推演模式」。
            <b>【最佳買入價（價格便宜）】</b>由布林通道下軌（極端超賣）及 250天年線形成的雙重強支撐位決定；
            <b>【最佳買入時間（動能起跑）】</b>由 AI 決策樹模型實時讀取 RSI、波動率、均線偏離度進行深度掃描，預測未來 5 天上漲機率。當 AI 與 物理支撐位發生共振，即為完美的黃金左側/右側交易買點。
        </div>

        <div style="background:#161c24; padding:15px; border-radius:10px;">
            {chart_html}
        </div>
        
        <div class="table-container">
            <h3>📋 今日核心港股 AI 雙軌推演決策表</h3>
            {table_html}
        </div>
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("🎉 恭喜！原生 API 超穩版 index.html 生成完畢！")
