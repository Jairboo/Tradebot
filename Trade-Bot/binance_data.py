
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class BinanceDataProvider:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',  # We'll use public API only
            'secret': '',
            'sandbox': False,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        # Binance symbol mapping
        self.symbol_map = {
            "XAU/USD": "XAUUSDT",  # Gold
            "EUR/USD": "EURUSDT",
            "USD/JPY": "USDTJPY",  
            "GBP/USD": "GBPUSDT",
            "USD/CHF": "USDTCHF",  
            "AUD/USD": "AUDUSDT",
            "NZD/USD": "NZDUSDT",
            "EUR/CHF": "EURCHF",
            "EUR/GBP": "EURGBP",
            "BTC/USD": "BTCUSDT",
            "ETH/USD": "ETHUSDT"
        }
    
    def get_symbol(self, market_name):
        """Convert market name to Binance symbol"""
        return self.symbol_map.get(market_name, market_name.replace("/", ""))
    
    def fetch_ohlcv_data(self, symbol, timeframe='1h', limit=500):
        """Fetch OHLCV data from Binance"""
        try:
            binance_symbol = self.get_symbol(symbol)
            
            # Fetch data from Binance
            ohlcv = self.exchange.fetch_ohlcv(binance_symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching Binance data for {symbol}: {e}")
            return None
    
    def get_ticker_24hr(self, symbol):
        """Get 24hr ticker statistics"""
        try:
            binance_symbol = self.get_symbol(symbol)
            ticker = self.exchange.fetch_ticker(binance_symbol)
            return ticker
        except Exception as e:
            print(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    def get_order_book(self, symbol, limit=100):
        """Get order book depth"""
        try:
            binance_symbol = self.get_symbol(symbol)
            orderbook = self.exchange.fetch_order_book(binance_symbol, limit)
            return orderbook
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None
    
    def get_market_depth_analysis(self, symbol):
        """Analyze market depth for better entry/exit points"""
        try:
            orderbook = self.get_order_book(symbol)
            if not orderbook:
                return {}
            
            bids = np.array(orderbook['bids'])
            asks = np.array(orderbook['asks'])
            
            if len(bids) == 0 or len(asks) == 0:
                return {}
            
            # Calculate bid/ask strength
            total_bid_volume = np.sum(bids[:, 1])
            total_ask_volume = np.sum(asks[:, 1])
            
            # Calculate weighted average prices
            bid_weighted_price = np.sum(bids[:, 0] * bids[:, 1]) / total_bid_volume if total_bid_volume > 0 else 0
            ask_weighted_price = np.sum(asks[:, 0] * asks[:, 1]) / total_ask_volume if total_ask_volume > 0 else 0
            
            # Calculate depth imbalance
            depth_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1
            
            # Large order detection
            large_bids = bids[bids[:, 1] > np.percentile(bids[:, 1], 90)]
            large_asks = asks[asks[:, 1] > np.percentile(asks[:, 1], 90)]
            
            return {
                'bid_volume': total_bid_volume,
                'ask_volume': total_ask_volume,
                'depth_ratio': depth_ratio,
                'bid_weighted_price': bid_weighted_price,
                'ask_weighted_price': ask_weighted_price,
                'large_bid_count': len(large_bids),
                'large_ask_count': len(large_asks),
                'spread': asks[0][0] - bids[0][0] if len(asks) > 0 and len(bids) > 0 else 0
            }
            
        except Exception as e:
            print(f"Error analyzing market depth for {symbol}: {e}")
            return {}
    
    def get_volume_profile(self, symbol, timeframe='1h', periods=24):
        """Get volume profile for better support/resistance levels"""
        try:
            df = self.fetch_ohlcv_data(symbol, timeframe, periods)
            if df is None or df.empty:
                return {}
            
            # Calculate volume weighted average price (VWAP)
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Price levels with high volume (support/resistance)
            price_bins = np.linspace(df['Low'].min(), df['High'].max(), 20)
            volume_at_price = []
            
            for i in range(len(price_bins) - 1):
                mask = (df['Low'] <= price_bins[i+1]) & (df['High'] >= price_bins[i])
                volume_at_level = df[mask]['Volume'].sum()
                volume_at_price.append((price_bins[i], volume_at_level))
            
            # Find high volume nodes (HVN) and low volume nodes (LVN)
            volume_at_price.sort(key=lambda x: x[1], reverse=True)
            hvn_levels = volume_at_price[:3]  # Top 3 high volume levels
            
            return {
                'vwap': df['VWAP'].iloc[-1],
                'hvn_levels': [level[0] for level in hvn_levels],
                'total_volume': df['Volume'].sum(),
                'avg_volume': df['Volume'].mean(),
                'volume_trend': df['Volume'].tail(5).mean() / df['Volume'].head(5).mean()
            }
            
        except Exception as e:
            print(f"Error calculating volume profile for {symbol}: {e}")
            return {}
    
    def get_funding_rate(self, symbol):
        """Get funding rate for perpetual futures (market sentiment)"""
        try:
            binance_symbol = self.get_symbol(symbol)
            if not binance_symbol.endswith('USDT'):
                return None
                
            funding_rate = self.exchange.fetch_funding_rate(binance_symbol)
            return funding_rate
        except Exception as e:
            # Not all symbols have funding rates
            return None
    
    def get_enhanced_market_data(self, symbol):
        """Get comprehensive market data for analysis"""
        try:
            # Get multiple timeframes
            data_1h = self.fetch_ohlcv_data(symbol, '1h', 200)
            data_4h = self.fetch_ohlcv_data(symbol, '4h', 100)
            data_1d = self.fetch_ohlcv_data(symbol, '1d', 50)
            
            # Get market depth
            depth_analysis = self.get_market_depth_analysis(symbol)
            
            # Get volume profile
            volume_profile = self.get_volume_profile(symbol)
            
            # Get 24hr ticker
            ticker_24hr = self.get_ticker_24hr(symbol)
            
            # Get funding rate (if available)
            funding_rate = self.get_funding_rate(symbol)
            
            return {
                'ohlcv_1h': data_1h,
                'ohlcv_4h': data_4h,
                'ohlcv_1d': data_1d,
                'depth_analysis': depth_analysis,
                'volume_profile': volume_profile,
                'ticker_24hr': ticker_24hr,
                'funding_rate': funding_rate
            }
            
        except Exception as e:
            print(f"Error getting enhanced market data for {symbol}: {e}")
            return None
