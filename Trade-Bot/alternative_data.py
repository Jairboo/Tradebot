
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class AlternativeDataProvider:
    def __init__(self):
        # Free API endpoints (no key required for basic functionality)
        self.endpoints = {
            'polygon': 'https://api.polygon.io/v2',
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'fred': 'https://api.stlouisfed.org/fred/series/observations',
            'yahoo_finance': 'https://query1.finance.yahoo.com/v8/finance/chart',
            'forex_api': 'https://api.exchangerate-api.com/v4/latest',
            'crypto_api': 'https://api.coingecko.com/api/v3',
            'commodities_api': 'https://api.metals.live/v1/spot'
        }
        
        # Enhanced symbol mapping for multiple providers
        self.symbol_mapping = {
            "XAU/USD": {
                'yahoo': 'GC=F',
                'metals': 'gold',
                'fred': 'GOLDAMGBD228NLBM'
            },
            "EUR/USD": {
                'yahoo': 'EURUSD=X',
                'forex': 'EUR',
                'fred': 'DEXUSEU'
            },
            "USD/JPY": {
                'yahoo': 'USDJPY=X', 
                'forex': 'JPY',
                'fred': 'DEXJPUS'
            },
            "GBP/USD": {
                'yahoo': 'GBPUSD=X',
                'forex': 'GBP', 
                'fred': 'DEXUSUK'
            },
            "USD/CHF": {
                'yahoo': 'USDCHF=X',
                'forex': 'CHF',
                'fred': 'DEXSZUS'
            },
            "AUD/USD": {
                'yahoo': 'AUDUSD=X',
                'forex': 'AUD',
                'fred': 'DEXUSAL'
            },
            "NZD/USD": {
                'yahoo': 'NZDUSD=X',
                'forex': 'NZD'
            },
            "EUR/CHF": {
                'yahoo': 'EURCHF=X'
            },
            "EUR/GBP": {
                'yahoo': 'EURGBP=X'
            },
            "BTC/USD": {
                'yahoo': 'BTC-USD',
                'coingecko': 'bitcoin',
                'symbol': 'BTCUSD'
            },
            "ETH/USD": {
                'yahoo': 'ETH-USD',
                'coingecko': 'ethereum', 
                'symbol': 'ETHUSD'
            }
        }

    def get_enhanced_yahoo_data(self, symbol, timeframe='1h', limit=200):
        """Enhanced Yahoo Finance data with multiple attempts"""
        try:
            # Map timeframe
            tf_map = {'1h': '1h', '4h': '1h', '1d': '1d'}
            period_map = {'1h': '30d', '4h': '60d', '1d': '1y'}
            
            yf_tf = tf_map.get(timeframe, '1h')
            yf_period = period_map.get(timeframe, '30d')
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(symbol, interval=yf_tf, period=yf_period, 
                                 auto_adjust=True, progress=False)
            
            if data.empty:
                return None
                
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # For 4h data, resample 1h to 4h
            if timeframe == '4h' and yf_tf == '1h':
                data = data.resample('4h').agg({
                    'Open': 'first',
                    'High': 'max', 
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            
            return data.tail(limit) if len(data) > limit else data
            
        except Exception as e:
            print(f"Enhanced Yahoo data error for {symbol}: {e}")
            return None

    def get_crypto_data_coingecko(self, crypto_id, days=30):
        """Get cryptocurrency data from CoinGecko"""
        try:
            url = f"{self.endpoints['crypto_api']}/coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 30 else 'daily'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
                
            data = response.json()
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume data
            if volumes:
                vol_df = pd.DataFrame(volumes, columns=['timestamp', 'Volume'])
                vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
                vol_df.set_index('timestamp', inplace=True)
                df = df.join(vol_df, how='left')
            
            # Create OHLC from Close prices (approximate)
            df['Open'] = df['Close'].shift(1)
            df['High'] = df[['Close', 'Open']].max(axis=1)
            df['Low'] = df[['Close', 'Open']].min(axis=1)
            
            # Fill missing values
            df['Open'].fillna(df['Close'], inplace=True)
            df['Volume'].fillna(0, inplace=True)
            
            return df.dropna()
            
        except Exception as e:
            print(f"CoinGecko data error for {crypto_id}: {e}")
            return None

    def get_forex_rates(self, base_currency='USD'):
        """Get current forex rates"""
        try:
            url = f"{self.endpoints['forex_api']}/{base_currency}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json().get('rates', {})
                
        except Exception as e:
            print(f"Forex API error: {e}")
        
        return {}

    def get_metals_data(self):
        """Get precious metals spot prices from Yahoo Finance"""
        try:
            # Use Yahoo Finance for gold prices as backup
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gold_data = yf.download('GC=F', period='5d', interval='1d', progress=False)
            
            if not gold_data.empty and len(gold_data) >= 2:
                current_price = gold_data['Close'].iloc[-1]
                prev_price = gold_data['Close'].iloc[-2]
                change = current_price - prev_price
                
                return {
                    'gold': {
                        'price': float(current_price),
                        'change': float(change),
                        'change_percent': float((change / prev_price) * 100)
                    }
                }
                
        except Exception as e:
            print(f"Alternative metals data error: {e}")
            
        return {}

    def get_economic_indicators(self, series_id):
        """Get economic data from FRED (Federal Reserve Economic Data)"""
        try:
            url = self.endpoints['fred']
            params = {
                'series_id': series_id,
                'api_key': 'demo',  # Using demo key for basic access
                'file_type': 'json',
                'limit': 100
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                if observations:
                    df = pd.DataFrame(observations)
                    df['date'] = pd.to_datetime(df['date'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df.set_index('date', inplace=True)
                    return df.dropna()
                    
        except Exception as e:
            print(f"FRED data error for {series_id}: {e}")
            
        return None

    def get_market_sentiment_data(self, symbol):
        """Get market sentiment indicators"""
        sentiment_data = {}
        
        try:
            # Fear & Greed Index (approximation)
            if 'BTC' in symbol or 'ETH' in symbol:
                # Crypto sentiment
                sentiment_data['asset_class'] = 'crypto'
                sentiment_data['volatility_regime'] = 'high'
            elif 'USD' in symbol:
                # Forex sentiment  
                sentiment_data['asset_class'] = 'forex'
                sentiment_data['volatility_regime'] = 'medium'
            elif 'XAU' in symbol:
                # Gold sentiment
                sentiment_data['asset_class'] = 'commodity'
                sentiment_data['volatility_regime'] = 'medium'
            
            # Get current forex rates for sentiment
            forex_rates = self.get_forex_rates()
            if forex_rates:
                sentiment_data['usd_strength'] = self.calculate_usd_strength(forex_rates)
            
            # Get metals data
            metals_data = self.get_metals_data()
            if metals_data:
                sentiment_data['gold_sentiment'] = self.analyze_gold_sentiment(metals_data)
                
        except Exception as e:
            print(f"Market sentiment error: {e}")
            
        return sentiment_data

    def calculate_usd_strength(self, forex_rates):
        """Calculate USD strength index from forex rates"""
        try:
            major_pairs = ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']
            usd_values = []
            
            for currency in major_pairs:
                if currency in forex_rates:
                    # For JPY, higher number means weaker USD
                    if currency == 'JPY':
                        usd_values.append(1 / forex_rates[currency] * 100)
                    else:
                        usd_values.append(forex_rates[currency])
            
            if usd_values:
                avg_strength = np.mean(usd_values)
                return 'strong' if avg_strength > 1.1 else 'weak' if avg_strength < 0.9 else 'neutral'
                
        except Exception as e:
            print(f"USD strength calculation error: {e}")
            
        return 'neutral'

    def analyze_gold_sentiment(self, metals_data):
        """Analyze gold market sentiment"""
        try:
            gold_info = metals_data.get('gold', {})
            gold_price = gold_info.get('price', 0)
            gold_change = gold_info.get('change', 0)
            
            # Price level analysis
            if gold_price > 2000:
                price_sentiment = 'bullish'
            elif gold_price < 1800:
                price_sentiment = 'bearish'
            else:
                price_sentiment = 'neutral'
            
            # Change analysis
            if gold_change > 20:  # $20+ increase
                return 'bullish'
            elif gold_change < -20:  # $20+ decrease
                return 'bearish'
            else:
                return price_sentiment
                
        except Exception as e:
            print(f"Gold sentiment error: {e}")
            
        return 'neutral'

    def get_comprehensive_data(self, market_name, timeframes=['1h', '4h', '1d']):
        """Get comprehensive market data from multiple sources"""
        data_dict = {}
        
        try:
            symbols = self.symbol_mapping.get(market_name, {})
            
            # Primary data source - Enhanced Yahoo Finance
            yahoo_symbol = symbols.get('yahoo', market_name.replace('/', ''))
            
            for tf in timeframes:
                data = self.get_enhanced_yahoo_data(yahoo_symbol, tf)
                
                if data is not None and not data.empty and len(data) >= 20:
                    data_dict[tf] = data
                    print(f"✅ Yahoo Finance data for {market_name} ({tf})")
                else:
                    print(f"⚠️ No Yahoo data for {market_name} ({tf})")
            
            # Alternative crypto data
            if market_name in ['BTC/USD', 'ETH/USD']:
                coingecko_id = symbols.get('coingecko')
                if coingecko_id:
                    crypto_data = self.get_crypto_data_coingecko(coingecko_id)
                    if crypto_data is not None:
                        data_dict['crypto_backup'] = crypto_data
                        print(f"✅ CoinGecko backup data for {market_name}")
            
            # Market sentiment and additional indicators
            sentiment = self.get_market_sentiment_data(market_name)
            if sentiment:
                data_dict['market_sentiment'] = sentiment
            
            # Economic indicators for forex pairs
            fred_series = symbols.get('fred')
            if fred_series:
                econ_data = self.get_economic_indicators(fred_series)
                if econ_data is not None:
                    data_dict['economic_data'] = econ_data
                    print(f"✅ Economic data for {market_name}")
            
            # Real-time rates for forex
            if 'forex' in symbols:
                forex_rates = self.get_forex_rates()
                if forex_rates:
                    data_dict['forex_rates'] = forex_rates
            
            # Metals data for gold
            if market_name == 'XAU/USD':
                metals_data = self.get_metals_data()
                if metals_data:
                    data_dict['metals_data'] = metals_data
                    
        except Exception as e:
            print(f"Comprehensive data error for {market_name}: {e}")
        
        return data_dict

    def enhance_signal_with_alternative_data(self, signal_data, market_data):
        """Enhance trading signal with alternative data sources"""
        if not market_data:
            return signal_data
            
        enhancement_score = 0
        additional_reasons = []
        
        try:
            # Market sentiment analysis
            sentiment = market_data.get('market_sentiment', {})
            
            if sentiment.get('usd_strength') == 'strong' and 'USD' in signal_data['market']:
                if signal_data['market'].startswith('USD'):
                    enhancement_score += 3
                    additional_reasons.append("🇺🇸 Strong USD momentum")
                else:
                    enhancement_score -= 3
                    additional_reasons.append("🇺🇸 USD strength pressure")
            
            # Economic data confirmation
            if 'economic_data' in market_data:
                econ_data = market_data['economic_data']
                if not econ_data.empty:
                    recent_change = econ_data['value'].pct_change().iloc[-1]
                    if abs(recent_change) > 0.02:  # 2% change
                        if recent_change > 0:
                            enhancement_score += 2
                            additional_reasons.append("📊 Positive economic data")
                        else:
                            enhancement_score -= 2
                            additional_reasons.append("📊 Negative economic data")
            
            # Forex rates confirmation
            if 'forex_rates' in market_data:
                rates = market_data['forex_rates']
                base_currency = signal_data['market'][:3]
                quote_currency = signal_data['market'][-3:]
                
                if base_currency in rates and quote_currency in rates:
                    # Cross rate calculation for validation
                    enhancement_score += 1
                    additional_reasons.append("💱 Real-time forex validation")
            
            # Crypto backup data validation
            if 'crypto_backup' in market_data:
                crypto_data = market_data['crypto_backup']
                if not crypto_data.empty:
                    recent_volume = crypto_data['Volume'].tail(24).mean()
                    if recent_volume > crypto_data['Volume'].mean() * 1.5:
                        enhancement_score += 3
                        additional_reasons.append("🚀 High crypto volume surge")
            
            # Metals data for gold
            if 'metals_data' in market_data and signal_data['market'] == 'XAU/USD':
                metals = market_data['metals_data']
                gold_sentiment = self.analyze_gold_sentiment(metals)
                
                if gold_sentiment == 'bullish' and signal_data['direction'] == 'BUY':
                    enhancement_score += 4
                    additional_reasons.append("🥇 Bullish gold fundamentals")
                elif gold_sentiment == 'bearish' and signal_data['direction'] == 'SELL':
                    enhancement_score += 4
                    additional_reasons.append("🥇 Bearish gold fundamentals")
            
            # Apply enhancements
            if enhancement_score != 0:
                signal_data['confidence'] = min(98, signal_data['confidence'] + enhancement_score)
                signal_data['reasons'].extend(additional_reasons[:3])  # Limit additional reasons
                signal_data['enhanced'] = True
                
        except Exception as e:
            print(f"Signal enhancement error: {e}")
        
        return signal_data
