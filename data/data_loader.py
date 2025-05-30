import os
import time
import random
import pandas as pd
import yfinance as yf
'''
Script created because yfinance was giving issues, and all it needed at the end was to pip install upgrade it!
But will work on this later one to download more data!
'''
def download_or_load_from_cache(ticker, start_date, end_date, cache_dir, max_retries=3, wait_time=10):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}.csv")

    for attempt in range(max_retries):
        try:
            print(f"\n🔽 Downloading {ticker} (attempt {attempt + 1})...")
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                threads=False,
                auto_adjust=True  # suppress warnings
            )

            if df.empty:
                raise ValueError("Downloaded DataFrame is empty.")

            # Handle MultiIndex or standard structure
            if isinstance(df.columns, pd.MultiIndex):
                col = ('Adj Close', ticker) if ('Adj Close', ticker) in df.columns else ('Close', ticker)
                series = df[col].rename(ticker)
            else:
                col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
                series = df[col].rename(ticker)

            series.to_frame().to_csv(cache_file)
            return series.to_frame()

        except Exception as e:
            print(f"⚠️ Error downloading {ticker}: {e}")
            if attempt < max_retries - 1:
                wait = wait_time + random.uniform(1, 5)
                print(f"🔁 Retrying in {wait:.1f} seconds...")
                time.sleep(wait)
            else:
                print(f"❌ Max retries reached for {ticker}. Checking for cached file...")

    if os.path.exists(cache_file):
        print(f"📂 Loading {ticker} from cache...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df
    else:
        print(f"🚫 No data_ingestion available for {ticker}")
        return None

# -------- Example usage --------
# start_date = "2024-01-02"
# end_date = "2025-05-01"
# tickers = ['SPY', 'GLD']
# cache_dir = "cache"
#
# all_data = []
#
# for ticker in tickers:
#     df = download_or_load_from_cache(ticker, start_date, end_date, cache_dir)
#     if df is not None:
#         all_data.append(df)
#
# if all_data:
#     combined_data = pd.concat(all_data, axis=1)
#     print("\n✅ Final Combined Data:")
#     print(combined_data.head())
# else:
#     print("❌ No data_ingestion available from any ticker.")