import numpy as np
import pandas as pd
import requests
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go
from datetime import datetime

# 🤖 40隻全球核心股票追蹤名單 (20隻港股 + 20隻美股)
STOCKS_HK = {
    "2800.HK": "盈富基金 (Tracker Fund)", "0005.HK": "匯豐控股 (HSBC)",
    "1299.HK": "友邦保險 (AIA)", "0388.HK": "香港交易所 (HKEX)",
    "0700.HK": "騰訊控股 (Tencent)", "9988.HK": "阿里巴巴 (Alibaba)",
    "3690.HK": "美團 (Meituan)", "1810.HK": "小米集團 (Xiaomi)",
    "1024.HK": "快手 (Kuaishou)", "9618.HK": "京東集團 (JD.com)",
    "9888.HK": "百度集團 (Baidu)", "1211.HK": "比亞迪股份 (BYD)",
    "2015.HK": "理想汽車 (Li Auto)", "0941.HK": "中國移動 (China Mobile)",
    "0883.HK": "中國海洋石油 (CNOOC)", "1398.HK": "工商銀行 (ICBC)",
    "0939.HK": "建設銀行 (CCB)", "2269.HK": "藥明生物 (WuXi Bio)",
    "1928.HK": "金沙中國 (Sands China)", "0669.HK": "創科實業 (TTI)"
}

STOCKS_US = {
    "SPY": "標普500指數ETF (S&P 500)", "QQQ": "納斯達克100ETF (Nasdaq 100)",
    "AAPL": "蘋果公司 (Apple)", "MSFT": "微軟 (Microsoft)",
    "NVDA": "輝達 (NVIDIA)", "GOOGL": "谷歌 (Alphabet/Google)",
    "AMZN": "亞馬遜 (Amazon)", "META": "臉書 (Meta)",
    "TSLA": "特斯拉 (Tesla)", "AVGO": "博通 (Broadcom)",
    "AMD": "超微半導體 (AMD)", "PLTR": "Palantir AI (Palantir)",
    "BRK-B": "巴郡哈薩威 (Berkshire B)", "JPM": "摩根大通 (JPMorgan)",
    "LLY": "禮來藥廠 (Eli Lilly)", "COST": "好市多 (Costco)",
    "WMT": "沃爾瑪 (Walmart)", "NFLX": "網飛 (Netflix)",
    "MSTR": "微策略怪獸 (MicroStrategy)", "COIN": "硬幣基地 (Coinbase)"
}

# 合併兩個市場字典
ALL_STOCKS = {**STOCKS_HK, **STOCKS_US}

results = []
print(f"🚀 啟動【港美雙軌全球聯動版】量化推演引擎，當前香港時間: {datetime.now()}")

# 建立偽裝 Session，繞過防火牆
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7'
})

fig_all = go.Figure()

def fetch_stock_data_raw(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
    response = session.get(url, timeout=15)
    if response.status_code != 200:
        raise Exception(f"Yahoo API 響應錯誤，HTTP Code: {response.status_code}")
    
    data = response.json()
    if 'chart' not in data or 'result' not in data['chart'] or data['chart']['result'] is None:
        raise Exception("JSON 數據格式無效")
        
    result = data['chart']['result'][0]
    timestamps = result.get('timestamp') or []
    if not timestamps:
        raise Exception("無有效時間戳")
        
    indicators = result.get('indicators', {}).get('quote', [{}])[0]
    open_prices = indicators.get('open') or []
    high_prices = indicators.get('high') or []
    low_prices = indicators.get('low') or []
    close_prices = indicators.get('close') or []
    volume_list = indicators.get('volume') or []
    
    if not (len(timestamps) == len(open_prices) == len(high_prices) == len(low_prices) == len(close_prices) == len(volume_list)):
        raise Exception("數據長度不匹配")
        
    df = pd.DataFrame({
        'Open': open_prices, 'High': high_prices, 'Low': low_prices, 'Close': close_prices, 'Volume': volume_list
    }, index=pd.to_datetime(timestamps, unit='s'))
    
    df['Close'] = df['Close'].ffill().bfill()
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Volume'] = df['Volume'].fillna(0)
    
    # 統一轉換為香港時間進行推演
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Hong_Kong')
    return df

for ticker, name in ALL_STOCKS.items():
    market_type = "港股 🇭🇰" if ticker in STOCKS_HK else "美股 🇺🇸"
    currency_prefix = "HK$" if ticker in STOCKS_HK else "US$"
    
    print(f"📊 正在下載並解析 [{market_type}] {name} ({ticker})...")
    
    try:
        stock_data = fetch_stock_data_raw(ticker)
    except Exception as e:
        print(f"❌ {ticker} 獲取失敗: {e}，自動跳過。")
        continue
        
    if stock_data.empty or len(stock_data) < 250:
        print(f"⚠️ {ticker} 數據不足，跳過。")
        continue
        
    df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # 1. 技術指標計算
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA250'] = df['Close'].rolling(window=250).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['MA20'] - 2 * df['STD20']
    df['BB_Upper'] = df['MA20'] + 2 * df['STD20']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std() * 100
    df['MA_Dev'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
    df.dropna(inplace=True)
    
    # 2. AI 決策樹訓練
    features = ['RSI', 'Volatility', 'MA_Dev']
    df['Future_5d_Return'] = df['Close'].pct_change(5).shift(-5)
    
    df['Label'] = 0
    df.loc[df['Future_5d_Return'] > 0.025, 'Label'] = 1
    df.loc[df['Future_5d_Return'] < -0.025, 'Label'] = -1
    
    train_df = df.dropna(subset=['Future_5d_Return']).copy()
    if len(train_df) < 30:
        continue
        
    ai_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    ai_model.fit(train_df[features], train_df['Label'])
    
    # 3. 實時雙向推演
    latest_row = df.iloc[-1]
    current_price = round(latest_row['Close'], 2)
    
    best_buy_min = round(min(latest_row['BB_Lower'], latest_row['MA250']), 2)
    best_buy_max = round(latest_row['BB_Lower'], 2)
    best_sell_price = round(latest_row['BB_Upper'], 2)
    
    latest_features = pd.DataFrame([[latest_row['RSI'], latest_row['Volatility'], latest_row['MA_Dev']]], columns=features)
    ai_pred_proba = ai_model.predict_proba(latest_features)[0]
    classes = ai_model.classes_
    
    buy_idx = np.where(classes == 1)[0]
    buy_probability = round(ai_pred_proba[buy_idx[0]] * 100, 1) if len(buy_idx) > 0 else 0.0
    
    sell_idx = np.where(classes == -1)[0]
    sell_probability = round(ai_pred_proba[sell_idx[0]] * 100, 1) if len(sell_idx) > 0 else 0.0
    
    # 綜合評級邏輯
    if sell_probability >= 60 and current_price >= (best_sell_price * 0.98):
        rating = "🚨 強烈建議分批止盈 (AI與高位阻力共振！)"
        rating_color = "#e74c3c"
    elif buy_probability >= 60 and current_price <= best_buy_max:
        rating = "💎 強烈建議分批買入 (AI與物理超賣共振)"
        rating_color = "#2ecc71"
    elif buy_probability >= 50:
        rating = "🟢 偏向多頭 (時機成熟，可試探買入)"
        rating_color = "#2ecc71"
    elif sell_probability >= 50:
        rating = "⚠️ 轉向空頭 (警惕回調風險，建議減持)"
        rating_color = "#e67e22"
    elif current_price <= best_buy_max:
        rating = "🟡 到達物理支撐位 (價格便宜，但動能未轉強)"
        rating_color = "#f1c40f"
    elif latest_row['RSI'] > 68:
        rating = "🔴 短線超買 (建議分批止盈/切勿追高)"
        rating_color = "#e74c3c"
    else:
        rating = "⚪ 觀望中 (未達最佳買賣區間)"
        rating_color = "#bdc3c7"
        
    results.append({
        "市場": market_type,
        "代號": ticker,
        "名稱": name,
        "當前收市價": f"{currency_prefix} {current_price}",
        "最佳買入區間": f"{currency_prefix} {best_buy_min} - {best_buy_max}",
        "最佳賣出目標": f"&ge; {currency_prefix} {best_sell_price}",
        "AI 買入信心": f"{buy_probability}%",
        "AI 賣出風險": f"{sell_probability}%",
        "綜合操作建議": f"<span style='color:{rating_color}; font-weight:bold;'>{rating}</span>",
        "14天RSI": round(latest_row['RSI'], 1)
    })

    # 4. 提取最近3個月的數據畫線
    three_months_ago = stock_data.index.max() - pd.Timedelta(days=90)
    hist_3m = stock_data[stock_data.index >= three_months_ago]
    formatted_dates = hist_3m.index.strftime('%Y-%m-%d')
    
    fig_all.add_trace(go.Scatter(
        x=formatted_dates, y=hist_3m['Close'], 
        name=f"[{market_type[0]}] {name}"
    ))

if not results:
    print("❌ 致命錯誤：所有股票數據獲取失敗！")
    exit(1)

df_report = pd.DataFrame(results)

# 調整圖表美化外觀
fig_all.update_layout(
    title="📈 港美全球 40 隻核心股票走勢對比 (實時Yahoo數據)",
    xaxis_title="日期", yaxis_title="相對股價",
    hovermode="x unified", template="plotly_dark",
    paper_bgcolor="#121212", plot_bgcolor="#161c24",
    margin=dict(l=40, r=40, t=60, b=40)
)

chart_html = fig_all.to_html(full_html=False, include_plotlyjs='cdn')
table_html = df_report.to_html(index=False, border=0, escape=False)

# 5. 生成全新的 HTML 網頁
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>全球港美股市 AI 雙軌量化推演系統</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background-color: #0c0f12; color: #e0e6ed; margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 25px 0; border-bottom: 2px solid #1f2937; }}
        h1 {{ margin: 0; color: #00ffcc; font-size: 25px; letter-spacing: 1px; }}
        .update-time {{ color: #88a0b0; font-size: 13px; margin-top: 8px; }}
        .intro-box {{ background: #161c24; padding: 18px; border-radius: 8px; margin: 20px 0; font-size: 13.5px; line-height: 1.6; border-left: 4px solid #00ffcc; }}
        .table-container {{ margin-top: 30px; overflow-x: auto; background: #161c24; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 12.5px; }}
        th {{ background-color: #1f2937; color: #00ffcc; font-weight: 600; padding: 10px; border-bottom: 2px solid #374151; }}
        td {{ padding: 10px 8px; border-bottom: 1px solid #1f2937; color: #cbd5e1; }}
        tr:hover {{ background-color: #1e293b; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 全球港美股市 AI 雙軌實時量化推演系統</h1>
            <div class="update-time">每日自動更新（香港時間 HKT 收市後）: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        
        <div class="intro-box">
            💡 <b>跨市場量化運作原理：</b> 本系統已全面打通香港市場與美國市場，追蹤 40 隻全球頂級權重資產：
            <br>1. <b>【智能多貨幣自動格式化】</b>：港股自動顯示 <b>HK$</b>，美股自動顯示 <b>US$</b>，統一對齊香港時間進行大數據推演。
            <br>2. <b>【雙向共振操盤】</b>：當全場超買、AI 賣出風險飆升時，系統會亮起 <b>🚨 強烈建議分批止盈</b> 警報；當大盤急跌、特徵觸發物理年線與布林強支撐時，亮起 <b>💎 強烈建議分批買入</b> 抄底訊號。
        </div>

        <div style="background:#161c24; padding:15px; border-radius:10px;">
            {chart_html}
        </div>
        
        <div class="table-container">
            <h3>📋 今日全球 40 隻核心資產 AI 雙向推演決策大盤</h3>
            {table_html}
        </div>
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("🎉 恭喜！全球港美雙向版 40隻股 AI 量化大盤網頁 index.html 生成完畢！")
