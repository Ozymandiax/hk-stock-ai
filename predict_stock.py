import numpy as np
import pandas as pd
import yfinance as yf
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

print("🚀 啟動港股實時量化推演引擎...")

# 建立 Plotly 多圖表對比
fig_all = go.Figure()

for ticker, name in STOCKS.items():
    print(f"📊 正在獲取 {name} ({ticker}) 歷史數據...")
    # 使用 yf.Ticker 獲取 clean 數據，避免 MultiIndex 問題
    stock_data = yf.Ticker(ticker).history(period="2y")
    
    if len(stock_data) < 250:
        print(f"⚠️ {ticker} 數據不足，跳過。")
        continue
        
    df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # ---------------------------------------------------------
    # 1. 技術指標計算 (物理防線)
    # ---------------------------------------------------------
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA250'] = df['Close'].rolling(window=250).mean() # 年線 (牛熊分界線)
    
    # 計算布林通道 (Bollinger Bands) -> 用來鎖定「最佳物理買入價」
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['MA20'] - 2 * df['STD20'] # 下軌 (強支撐位)
    df['BB_Upper'] = df['MA20'] + 2 * df['STD20'] # 上軌 (強阻力位)
    
    # 計算 RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 波動率
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std() * 100
    df['MA_Dev'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
    
    df.dropna(inplace=True)
    
    # ---------------------------------------------------------
    # 2. AI 決策樹訓練 (預測未來 5 天上漲機率)
    # ---------------------------------------------------------
    features = ['RSI', 'Volatility', 'MA_Dev']
    df['Future_5d_Return'] = df['Close'].shift(-5).pct_change(5)
    
    # 標籤：未來 5 天升幅 > 2% 記為 1 (買入)，跌幅 > 2% 記為 -1 (賣出)，其餘為 0
    df['Label'] = 0
    df.loc[df['Future_5d_Return'] > 0.02, 'Label'] = 1
    df.loc[df['Future_5d_Return'] < -0.02, 'Label'] = -1
    
    train_df = df.iloc[:-5].copy() # 避開最後 5 天
    
    ai_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    ai_model.fit(train_df[features], train_df['Label'])
    
    # ---------------------------------------------------------
    # 3. 實時推演今天最新的狀態
    # ---------------------------------------------------------
    latest_row = df.iloc[-1]
    current_price = round(latest_row['Close'], 2)
    
    # 計算「最佳買入價區間」：
    # 結合布林下軌(BB_Lower) 與 MA250(年線) 作為雙重物理支撐位
    best_buy_min = round(min(latest_row['BB_Lower'], latest_row['MA250']), 2)
    best_buy_max = round(latest_row['BB_Lower'], 2)
    
    # AI 機率預測 (評估買入時機)
    latest_features = np.array([[latest_row['RSI'], latest_row['Volatility'], latest_row['MA_Dev']]])
    ai_pred_proba = ai_model.predict_proba(latest_features)[0]
    
    # class_labels 對應 [-1, 0, 1]。 index 2 是 1 (買入機率)
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

df_report = pd.DataFrame(results)

# ---------------------------------------------------------
# 4. 繪製 Plotly 多股趨勢對比圖
# ---------------------------------------------------------
print("⚡ 正在生成互動式分析圖表...")
for ticker, name in STOCKS.items():
    hist = yf.Ticker(ticker).history(period="3m")
    fig_all.add_trace(go.Scatter(x=hist.index.astype(str), y=hist['Close'], name=f"{name} 股價"))

fig_all.update_layout(
    title="📈 近三個月核心港股走勢對比 (實時Yahoo數據)",
    xaxis_title="日期",
    yaxis_title="股價 (HKD)",
    hovermode="x unified",
    template="plotly_dark",
    paper_bgcolor="#121212",
    plot_bgcolor="#1e1e1e"
)

chart_html = fig_all.to_html(full_html=False, include_plotlyjs='cdn')
table_html = df_report.to_html(index=False, border=0, escape=False)

# ---------------------------------------------------------
# 5. 生成精美的前端 HTML 網頁
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

print("🎉 恭喜！港股 AI 推演網頁 index.html 生成完畢！")
