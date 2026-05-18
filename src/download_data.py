#!/usr/bin/env python3
"""Download market data. Run BEFORE experiment.py."""
import os, json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TICKERS = [
    'SPY','QQQ','IWM','DIA','ARKK','VTI','VOO',
    'EWG','EWU','EWQ','EWI','EWP','EWL','EWN','EWD','EWK',
    'EWJ','EWY','EWA','EWT','EWS','EWH','MCHI',
    'EEM','EWZ','INDA','EWW','TUR','EZA','ECH','EPOL',
    'XLK','XLF','XLE','XLV','XLRE','SMH','XLU','XLI','XLP','XLB','XLC',
    'GLD','SLV','TLT','HYG','LQD','TIP','VNQ','IBIT',
    'ACWI','^VIX','UUP','USO','DBC'
]
START, END = '2008-01-01', '2026-05-18'

def main():
    import yfinance as yf
    datadir = ROOT / 'data'
    datadir.mkdir(exist_ok=True)
    print(f"Downloading {len(TICKERS)} tickers...")
    data = yf.download(TICKERS, start=START, end=END, interval='1d',
                       auto_adjust=True, progress=True)
    prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
    if hasattr(prices.columns, 'droplevel'):
        try: prices.columns = prices.columns.droplevel(0)
        except: pass
    prices.to_csv(datadir / 'prices.csv')
    manifest = {'tickers': TICKERS, 'start': START, 'end': END,
                'download_date': datetime.now().isoformat(),
                'rows': int(prices.shape[0]), 'cols': int(prices.shape[1]),
                'source': 'Yahoo Finance via yfinance (personal/research use)'}
    with open(datadir / 'ticker_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: data/prices.csv ({prices.shape[0]} x {prices.shape[1]})")

if __name__ == '__main__': main()
