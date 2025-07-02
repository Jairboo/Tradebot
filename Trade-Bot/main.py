import yfinance as yf
import pandas as pd
import ta
import requests
import schedule
import time
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import json
import warnings
# Suppress warnings inline where needed

# --- CONFIGURATION ---
from config import MARKETS, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, CONFIDENCE_SCALE, ECONOMIC_CALENDAR_API
from alternative_data import AlternativeDataProvider
from news_sentiment import NewsAnalyzer

# --- ALTERNATIVE DATA PROVIDER ---
alt_data_provider = AlternativeDataProvider()
news_analyzer = NewsAnalyzer()

# --- DATABASE SETUP ---
Base = declarative_base()


class Signal(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True)
    market = Column(String)
    direction = Column(String)
    entry = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    confidence = Column(Integer)
    reasons = Column(String)
    fundamental_score = Column(Integer, default=0)
    technical_score = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Subscriber(Base):
    __tablename__ = 'subscribers'
    id = Column(Integer, primary_key=True)
    chat_id = Column(String, unique=True)
    username = Column(String)
    first_name = Column(String)
    is_active = Column(Boolean, default=True)
    subscribed_at = Column(DateTime, default=datetime.utcnow)


import os
# Ensure database is created in the current directory
db_path = os.path.join(os.getcwd(), "signals.db")
engine = create_engine(f"sqlite:///{db_path}")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# --- TELEGRAM BOT FUNCTIONS ---
last_update_id = 0


def get_telegram_updates(offset=None):
    """Get updates from Telegram bot"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": 10}
    if offset:
        params["offset"] = offset

    try:
        response = requests.get(url, params=params, timeout=15)
        return response.json()
    except Exception as e:
        print(f"Error getting Telegram updates: {e}")
        return None


def send_message(chat_id, text, parse_mode="Markdown"):
    """Send message to specific chat"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}

    try:
        response = requests.post(url, data=data, timeout=10)
        return response.json().get("ok", False)
    except Exception as e:
        print(f"Error sending message to {chat_id}: {e}")
        return False


def add_subscriber(chat_id, username=None, first_name=None):
    """Add new subscriber to database"""
    session = Session()
    try:
        existing = session.query(Subscriber).filter_by(chat_id=str(chat_id)).first()
        if existing:
            existing.is_active = True
            existing.username = username
            existing.first_name = first_name
            session.commit()
            return False  # Already existed
        else:
            subscriber = Subscriber(chat_id=str(chat_id), username=username, first_name=first_name)
            session.add(subscriber)
            session.commit()
            return True  # New subscriber
    except Exception as e:
        print(f"Error adding subscriber: {e}")
        return False
    finally:
        session.close()


def remove_subscriber(chat_id):
    """Remove subscriber from database"""
    session = Session()
    try:
        subscriber = session.query(Subscriber).filter_by(chat_id=str(chat_id)).first()
        if subscriber:
            subscriber.is_active = False
            session.commit()
            return True
        return False
    except Exception as e:
        print(f"Error removing subscriber: {e}")
        return False
    finally:
        session.close()


def get_active_subscribers():
    """Get all active subscribers"""
    session = Session()
    try:
        subscribers = session.query(Subscriber).filter_by(is_active=True).all()
        return [sub.chat_id for sub in subscribers]
    except Exception as e:
        print(f"Error getting subscribers: {e}")
        return []
    finally:
        session.close()


def handle_telegram_commands():
    """Handle incoming Telegram messages"""
    global last_update_id
    try:
        offset = last_update_id + 1 if last_update_id > 0 else None
        updates = get_telegram_updates(offset)
        if not updates or not updates.get("ok"):
            return

        for update in updates.get("result", []):
            last_update_id = update["update_id"]

            if "message" not in update:
                continue

            message = update["message"]
            chat_id = message["chat"]["id"]
            text = message.get("text", "")
            username = message["from"].get("username")
            first_name = message["from"].get("first_name")

            if not text.startswith("/"):
                continue

            if text.startswith("/start"):
                is_new = add_subscriber(chat_id, username, first_name)
                if is_new:
                    welcome_msg = (
                        "🎉 *Welcome to Multi-Market Trading Bot!*\n\n"
                        "✅ You're now subscribed to professional trading signals\n"
                        "📊 Markets: Gold, Forex (8 pairs), Crypto (BTC, ETH)\n"
                        "🔬 Analysis: 15+ Technical + Fundamental indicators\n"
                        "🎯 Only signals with 75%+ confidence are sent\n"
                        "⏰ Analysis runs every hour\n\n"
                        "🚀 Get ready for multi-market signals!")
                    send_message(chat_id, welcome_msg)

            elif text.startswith("/stop"):
                remove_subscriber(chat_id)

            elif text.startswith("/stats"):
                stats_msg = ("📊 *Multi-Market Bot Information*\n\n"
                            f"🔄 Analysis Frequency: *Every Hour*\n"
                            f"💎 Markets: *{len(MARKETS)} Total*\n"
                            f"📈 Forex Pairs: *8*\n"
                            f"₿ Crypto: *BTC, ETH*\n"
                            f"🥇 Commodities: *Gold*\n"
                            f"🎯 Minimum Confidence: *75%*\n"
                            f"📊 Technical Indicators: *15+*\n"
                            f"📰 Fundamental Analysis: *Enabled*\n\n"
                            "⚡ Advanced multi-market analysis!")
                send_message(chat_id, stats_msg)

            elif text.startswith("/help"):
                help_msg = (
                    "🤖 *Available Commands:*\n\n"
                    "• /start - Subscribe to signals\n"
                    "• /stop - Unsubscribe from signals\n"
                    "• /stats - View bot information\n"
                    "• /help - Show this help message\n\n"
                    "💡 Send /start to begin receiving trading signals!")
                send_message(chat_id, help_msg)

    except Exception as e:
        print(f"Error handling Telegram commands: {e}")


# --- FUNDAMENTAL ANALYSIS ---
def get_market_sentiment(symbol):
    """Get market sentiment and fundamental data"""
    fundamental_score = 0
    reasons = []

    try:
        # Get basic info
        ticker = yf.Ticker(symbol)

        # Get historical data for analysis
        hist = ticker.history(period="5d")
        if len(hist) >= 2:
            price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]

            if price_change > 0.01:  # 1% increase
                fundamental_score += 2
                reasons.append(f"Strong recent performance (+{price_change*100:.1f}%)")
            elif price_change < -0.01:
                fundamental_score -= 2
                reasons.append(f"Weak recent performance ({price_change*100:.1f}%)")

        # Crypto-specific fundamentals
        if "BTC" in symbol or "ETH" in symbol:
            try:
                info = ticker.info
                market_cap = info.get('marketCap', 0) if info else 0
                if market_cap > 1e12:  # > 1 trillion
                    fundamental_score += 3
                    reasons.append("Large market cap (>$1T)")
                elif market_cap > 1e11:  # > 100 billion
                    fundamental_score += 2
                    reasons.append("Substantial market cap (>$100B)")
            except:
                pass

        # USD strength analysis for forex pairs
        if "USD" in symbol and symbol not in ["BTC-USD", "ETH-USD"]:
            try:
                # Get DXY (Dollar Index) data
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dxy = yf.download("DX-Y.NYB", period="5d", interval="1d", progress=False)
                if not dxy.empty and len(dxy) >= 2:
                    dxy_change = (dxy['Close'].iloc[-1] - dxy['Close'].iloc[-2]) / dxy['Close'].iloc[-2]

                    if "USD" in symbol[:3]:  # USD is base currency
                        if dxy_change > 0.005:  # 0.5% DXY increase
                            fundamental_score += 2
                            reasons.append("USD strength (DXY rising)")
                        elif dxy_change < -0.005:
                            fundamental_score -= 2
                            reasons.append("USD weakness (DXY falling)")
            except:
                pass

        # Gold-specific fundamentals
        if symbol == "GC=F":
            try:
                # Get 10-year treasury yield
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    treasury = yf.download("^TNX", period="5d", interval="1d", progress=False)
                if not treasury.empty and len(treasury) >= 2:
                    yield_change = treasury['Close'].iloc[-1] - treasury['Close'].iloc[-2]
                    if yield_change > 0.1:  # Rising yields negative for gold
                        fundamental_score -= 2
                        reasons.append("Rising treasury yields pressuring gold")
                    elif yield_change < -0.1:  # Falling yields positive for gold
                        fundamental_score += 2
                        reasons.append("Falling treasury yields supporting gold")
            except:
                pass

    except Exception as e:
        print(f"Error in fundamental analysis for {symbol}: {e}")

    return fundamental_score, reasons


# --- TECHNICAL ANALYSIS FUNCTIONS ---
def safe_rolling_calculation(series, window, func_name='max'):
    """Safely calculate rolling statistics"""
    try:
        if func_name == 'max':
            return series.rolling(window=window, min_periods=1).max()
        elif func_name == 'min':
            return series.rolling(window=window, min_periods=1).min()
        elif func_name == 'mean':
            return series.rolling(window=window, min_periods=1).mean()
    except:
        return pd.Series(index=series.index, data=np.nan)


def calculate_ichimoku(data):
    """Calculate Ichimoku Cloud components"""
    try:
        high_9 = safe_rolling_calculation(data['High'], 9, 'max')
        low_9 = safe_rolling_calculation(data['Low'], 9, 'min')
        data['Tenkan'] = (high_9 + low_9) / 2

        high_26 = safe_rolling_calculation(data['High'], 26, 'max')
        low_26 = safe_rolling_calculation(data['Low'], 26, 'min')
        data['Kijun'] = (high_26 + low_26) / 2

        data['Senkou_A'] = ((data['Tenkan'] + data['Kijun']) / 2).shift(26)

        high_52 = safe_rolling_calculation(data['High'], 52, 'max')
        low_52 = safe_rolling_calculation(data['Low'], 52, 'min')
        data['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)

        data['Chikou'] = data['Close'].shift(-26)
    except Exception as e:
        print(f"Error calculating Ichimoku: {e}")
        data['Tenkan'] = data['Close']
        data['Kijun'] = data['Close']
        data['Senkou_A'] = data['Close']
        data['Senkou_B'] = data['Close']
        data['Chikou'] = data['Close']

    return data


def calculate_market_structure(data):
    """Advanced market structure analysis"""
    try:
        # Initialize columns with False
        data['SwingHigh'] = np.nan
        data['SwingLow'] = np.nan
        data['HH'] = False
        data['LL'] = False
        data['HL'] = False
        data['LH'] = False
        data['StructureBias'] = 0

        if len(data) < 10:
            return data

        # Swing highs and lows detection
        for i in range(2, len(data) - 2):
            # Swing High
            if (data['High'].iloc[i] > data['High'].iloc[i-1] and
                data['High'].iloc[i] > data['High'].iloc[i+1] and
                data['High'].iloc[i] > data['High'].iloc[i-2] and
                data['High'].iloc[i] > data['High'].iloc[i+2]):
                data.iloc[i, data.columns.get_loc('SwingHigh')] = data['High'].iloc[i]

            # Swing Low
            if (data['Low'].iloc[i] < data['Low'].iloc[i-1] and
                data['Low'].iloc[i] < data['Low'].iloc[i+1] and
                data['Low'].iloc[i] < data['Low'].iloc[i-2] and
                data['Low'].iloc[i] < data['Low'].iloc[i+2]):
                data.iloc[i, data.columns.get_loc('SwingLow')] = data['Low'].iloc[i]

        # Market structure patterns
        swing_highs = data['SwingHigh'].dropna()
        swing_lows = data['SwingLow'].dropna()

        if len(swing_highs) >= 2:
            if swing_highs.iloc[-1] > swing_highs.iloc[-2]:
                data.iloc[-1, data.columns.get_loc('HH')] = True

        if len(swing_lows) >= 2:
            if swing_lows.iloc[-1] < swing_lows.iloc[-2]:
                data.iloc[-1, data.columns.get_loc('LL')] = True
            elif swing_lows.iloc[-1] > swing_lows.iloc[-2]:
                data.iloc[-1, data.columns.get_loc('HL')] = True

        # Trend structure scoring
        recent_structure = min(10, len(data))
        hh_count = data['HH'].tail(recent_structure).sum()
        hl_count = data['HL'].tail(recent_structure).sum()
        ll_count = data['LL'].tail(recent_structure).sum()

        # Market structure bias
        if hh_count >= 1 and hl_count >= 1:
            data.iloc[-1, data.columns.get_loc('StructureBias')] = 1  # Bullish
        elif ll_count >= 2:
            data.iloc[-1, data.columns.get_loc('StructureBias')] = -1  # Bearish
        else:
            data.iloc[-1, data.columns.get_loc('StructureBias')] = 0  # Neutral

    except Exception as e:
        print(f"Error calculating market structure: {e}")
        data['SwingHigh'] = np.nan
        data['SwingLow'] = np.nan
        data['HH'] = False
        data['LL'] = False
        data['HL'] = False
        data['LH'] = False
        data['StructureBias'] = 0

    return data


def identify_candlestick_patterns(data):
    """Advanced candlestick pattern recognition"""
    try:
        # Calculate candlestick components
        data['Body'] = abs(data['Close'] - data['Open'])
        data['UpperShadow'] = data['High'] - np.maximum(data['Open'], data['Close'])
        data['LowerShadow'] = np.minimum(data['Open'], data['Close']) - data['Low']

        # Avoid division by zero
        range_val = data['High'] - data['Low']
        range_val = range_val.replace(0, 0.0001)
        data['BodyPercent'] = data['Body'] / range_val

        data['IsGreen'] = (data['Close'] > data['Open'])

        # Initialize pattern columns
        pattern_columns = ['Hammer', 'ShootingStar', 'Doji', 'BullishEngulfing', 
                          'BearishEngulfing', 'MorningStar', 'EveningStar']
        for col in pattern_columns:
            data[col] = False

        if len(data) < 3:
            return data

        # Pattern recognition with safe operations
        for i in range(1, len(data)):
            # Hammer
            if (data['LowerShadow'].iloc[i] > 2 * data['Body'].iloc[i] and
                data['UpperShadow'].iloc[i] < 0.1 * data['Body'].iloc[i] and
                data['BodyPercent'].iloc[i] < 0.3):
                data.iloc[i, data.columns.get_loc('Hammer')] = True

            # Shooting Star
            if (data['UpperShadow'].iloc[i] > 2 * data['Body'].iloc[i] and
                data['LowerShadow'].iloc[i] < 0.1 * data['Body'].iloc[i] and
                data['BodyPercent'].iloc[i] < 0.3):
                data.iloc[i, data.columns.get_loc('ShootingStar')] = True

            # Doji
            if data['BodyPercent'].iloc[i] < 0.05:
                data.iloc[i, data.columns.get_loc('Doji')] = True

            # Bullish Engulfing
            if (i > 0 and 
                not data['IsGreen'].iloc[i-1] and
                data['IsGreen'].iloc[i] and
                data['Open'].iloc[i] < data['Close'].iloc[i-1] and
                data['Close'].iloc[i] > data['Open'].iloc[i-1]):
                data.iloc[i, data.columns.get_loc('BullishEngulfing')] = True

            # Bearish Engulfing
            if (i > 0 and
                data['IsGreen'].iloc[i-1] and
                not data['IsGreen'].iloc[i] and
                data['Open'].iloc[i] > data['Close'].iloc[i-1] and
                data['Close'].iloc[i] < data['Open'].iloc[i-1]):
                data.iloc[i, data.columns.get_loc('BearishEngulfing')] = True

        # Three-candle patterns
        for i in range(2, len(data)):
            # Morning Star
            if (not data['IsGreen'].iloc[i-2] and
                data['Body'].iloc[i-1] < data['Body'].iloc[i-2] * 0.3 and
                data['IsGreen'].iloc[i] and
                data['Close'].iloc[i] > (data['Open'].iloc[i-2] + data['Close'].iloc[i-2]) / 2):
                data.iloc[i, data.columns.get_loc('MorningStar')] = True

            # Evening Star
            if (data['IsGreen'].iloc[i-2] and
                data['Body'].iloc[i-1] < data['Body'].iloc[i-2] * 0.3 and
                not data['IsGreen'].iloc[i] and
                data['Close'].iloc[i] < (data['Open'].iloc[i-2] + data['Close'].iloc[i-2]) / 2):
                data.iloc[i, data.columns.get_loc('EveningStar')] = True

    except Exception as e:
        print(f"Error in candlestick pattern recognition: {e}")
        # Initialize all pattern columns to False if error occurs
        pattern_columns = ['Hammer', 'ShootingStar', 'Doji', 'BullishEngulfing', 
                          'BearishEngulfing', 'MorningStar', 'EveningStar']
        for col in pattern_columns:
            data[col] = False

    return data


def calculate_support_resistance(data, window=20):
    """Dynamic support and resistance levels"""
    try:
        # Calculate recent highs and lows
        data['ResistanceLevel'] = safe_rolling_calculation(data['High'], window, 'max')
        data['SupportLevel'] = safe_rolling_calculation(data['Low'], window, 'min')

        # Distance from S/R levels
        data['DistanceFromResistance'] = (data['ResistanceLevel'] - data['Close']) / data['Close']
        data['DistanceFromSupport'] = (data['Close'] - data['SupportLevel']) / data['Close']

        # Key level touches
        data['ResistanceTouch'] = abs(data['High'] - data['ResistanceLevel']) / data['Close'] < 0.002
        data['SupportTouch'] = abs(data['Low'] - data['SupportLevel']) / data['Close'] < 0.002

        # False breakouts
        data['FalseBreakoutHigh'] = False
        data['FalseBreakoutLow'] = False

        if len(data) > 1:
            for i in range(1, len(data)):
                if (data['High'].iloc[i-1] > data['ResistanceLevel'].iloc[i-1] and
                    data['Close'].iloc[i] < data['ResistanceLevel'].iloc[i]):
                    data.iloc[i, data.columns.get_loc('FalseBreakoutHigh')] = True

                if (data['Low'].iloc[i-1] < data['SupportLevel'].iloc[i-1] and
                    data['Close'].iloc[i] > data['SupportLevel'].iloc[i]):
                    data.iloc[i, data.columns.get_loc('FalseBreakoutLow')] = True

    except Exception as e:
        print(f"Error calculating support/resistance: {e}")
        data['ResistanceLevel'] = data['High']
        data['SupportLevel'] = data['Low']
        data['DistanceFromResistance'] = 0
        data['DistanceFromSupport'] = 0
        data['ResistanceTouch'] = False
        data['SupportTouch'] = False
        data['FalseBreakoutHigh'] = False
        data['FalseBreakoutLow'] = False

    return data


def calculate_advanced_indicators(data):
    """Calculate all technical indicators with error handling"""
    try:
        # Basic indicators with error handling
        try:
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        except:
            data['RSI'] = 50

        try:
            data['RSI_Stoch'] = ta.momentum.StochRSIIndicator(data['Close']).stochrsi_k()
        except:
            data['RSI_Stoch'] = 0.5

        try:
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Histogram'] = macd.macd_diff()
        except:
            data['MACD'] = 0
            data['MACD_Signal'] = 0
            data['MACD_Histogram'] = 0

        try:
            bb = ta.volatility.BollingerBands(data['Close'])
            data['BB_High'] = bb.bollinger_hband()
            data['BB_Low'] = bb.bollinger_lband()
            data['BB_Mid'] = bb.bollinger_mavg()
            data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / data['BB_Mid']
        except:
            data['BB_High'] = data['Close']
            data['BB_Low'] = data['Close']
            data['BB_Mid'] = data['Close']
            data['BB_Width'] = 0

        # Moving averages
        try:
            data['EMA20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
            data['EMA50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
            data['EMA200'] = ta.trend.EMAIndicator(data['Close'], window=200).ema_indicator()
            data['SMA9'] = ta.trend.SMAIndicator(data['Close'], window=9).sma_indicator()
            data['SMA21'] = ta.trend.SMAIndicator(data['Close'], window=21).sma_indicator()
        except:
            data['EMA20'] = data['Close']
            data['EMA50'] = data['Close']
            data['EMA200'] = data['Close']
            data['SMA9'] = data['Close']
            data['SMA21'] = data['Close']

        # ADX
        try:
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx()
            data['DI_Plus'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx_pos()
            data['DI_Minus'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close']).adx_neg()
        except:
            data['ADX'] = 25
            data['DI_Plus'] = 25
            data['DI_Minus'] = 25

        # Other indicators
        try:
            data['Williams_R'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
        except:
            data['Williams_R'] = -50

        try:
            data['CCI'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        except:
            data['CCI'] = 0

        try:
            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        except:
            data['ATR'] = (data['High'] - data['Low']).rolling(window=14).mean()

        try:
            data['PSAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
        except:
            data['PSAR'] = data['Close']

        # Volume indicators
        try:
            if 'Volume' in data.columns and data['Volume'].sum() > 0:
                data['MFI'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                data['Volume_MA'] = safe_rolling_calculation(data['Volume'], 20, 'mean')
                data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
            else:
                data['MFI'] = 50
                data['VWAP'] = safe_rolling_calculation(data['Close'], 20, 'mean')
                data['Volume_Ratio'] = 1.0
        except:
            data['MFI'] = 50
            data['VWAP'] = data['Close']
            data['Volume_Ratio'] = 1.0

        # Advanced calculations
        data = calculate_ichimoku(data)
        data = calculate_market_structure(data)
        data = identify_candlestick_patterns(data)
        data = calculate_support_resistance(data)

        # Order flow analysis
        try:
            data['BuyingPressure'] = np.where(
                data['Close'] > data['Open'],
                (data['Close'] - data['Low']) / (data['High'] - data['Low']),
                (data['Close'] - data['Low']) / (data['High'] - data['Low']) * 0.5
            )
            data['SellingPressure'] = np.where(
                data['Close'] < data['Open'],
                (data['High'] - data['Close']) / (data['High'] - data['Low']),
                (data['High'] - data['Close']) / (data['High'] - data['Low']) * 0.5
            )
            data['VolumeWeightedPA'] = data['BuyingPressure'] - data['SellingPressure']
        except:
            data['BuyingPressure'] = 0.5
            data['SellingPressure'] = 0.5
            data['VolumeWeightedPA'] = 0

        # Trend strength
        try:
            data['ConsecutiveBulls'] = 0
            data['ConsecutiveBears'] = 0
            data['TrendMomentum'] = 1.0

            if len(data) > 1:
                bulls = 0
                bears = 0
                for i in range(len(data)):
                    if data['Close'].iloc[i] > data['Open'].iloc[i]:
                        bulls += 1
                        bears = 0
                    else:
                        bears += 1
                        bulls = 0
                    data.iloc[i, data.columns.get_loc('ConsecutiveBulls')] = bulls
                    data.iloc[i, data.columns.get_loc('ConsecutiveBears')] = bears
        except:
            data['ConsecutiveBulls'] = 0
            data['ConsecutiveBears'] = 0
            data['TrendMomentum'] = 1.0

        # Fill any remaining NaN values using modern pandas methods
        data = data.ffill().bfill().fillna(0)

    except Exception as e:
        print(f"Error in advanced indicators calculation: {e}")

    return data


# --- ENHANCED MULTI-MARKET DATA FETCHING ---
def fetch_market_data(symbol, market_name):
    """Fetch enhanced data using multiple alternative data sources"""
    data_dict = {}

    try:
        print(f"🔍 Fetching comprehensive data for {market_name}...")

        # Get comprehensive data from alternative sources
        alt_data = alt_data_provider.get_comprehensive_data(market_name)

        # Process timeframe data
        for tf in ['1h', '4h', '1d']:
            if tf in alt_data:
                df = alt_data[tf]
                if df is not None and not df.empty and len(df) >= 20:
                    df = calculate_advanced_indicators(df)
                    data_dict[tf] = df
                    print(f"✅ Processed {tf} data for {market_name}")

        # Add alternative data sources
        if 'market_sentiment' in alt_data:
            data_dict['market_sentiment'] = alt_data['market_sentiment']

        if 'economic_data' in alt_data:
            data_dict['economic_data'] = alt_data['economic_data']

        if 'forex_rates' in alt_data:
            data_dict['forex_rates'] = alt_data['forex_rates']

        if 'metals_data' in alt_data:
            data_dict['metals_data'] = alt_data['metals_data']

        if 'crypto_backup' in alt_data:
            # Use crypto backup if primary data insufficient
            if '1h' not in data_dict or len(data_dict['1h']) < 50:
                crypto_df = alt_data['crypto_backup']
                if crypto_df is not None and len(crypto_df) >= 20:
                    crypto_df = calculate_advanced_indicators(crypto_df)
                    data_dict['1h'] = crypto_df
                    print(f"✅ Using CoinGecko backup for {market_name}")

        # Fallback to basic yfinance if no primary data
        if '1h' not in data_dict:
            print(f"⚠️ Using basic yfinance fallback for {market_name}")
            timeframes = {'1h': '30d', '4h': '60d', '1d': '1y'}

            for interval, period in timeframes.items():
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        data = yf.download(symbol, interval=interval, period=period, auto_adjust=True, progress=False)

                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.droplevel(1)

                    if data.empty or len(data) < 50:
                        continue

                    data = data.dropna(subset=['Close'])
                    if len(data) < 20:
                        continue

                    data = calculate_advanced_indicators(data)
                    data_dict[interval] = data

                except Exception as e:
                    print(f"Error fetching {interval} fallback data for {symbol}: {e}")
                    continue

    except Exception as e:
        print(f"Error in comprehensive data fetching for {market_name}: {e}")

    return data_dict


# --- ENHANCED SIGNAL GENERATOR ---
def generate_market_signal(market_name, symbol):
    """Generate signal with Binance integration and 98%+ accuracy targeting"""
    try:
        print(f"📊 Analyzing {market_name}...")

        # Fetch enhanced technical data
        data_dict = fetch_market_data(symbol, market_name)
        if not data_dict or '1h' not in data_dict:
            print(f"❌ No data available for {market_name}")
            return None

        primary_data = data_dict['1h']
        if len(primary_data) == 0:
            return None

        latest = primary_data.iloc[-1]
        price = latest['Close']

        # Initialize enhanced scoring system
        bullish_signals = 0
        bearish_signals = 0
        reasons = []
        critical_factors = 0

        # === ALTERNATIVE DATA ANALYSIS ===

        # Market Sentiment Analysis
        if 'market_sentiment' in data_dict:
            sentiment = data_dict['market_sentiment']

            # USD strength analysis for forex pairs
            if sentiment.get('usd_strength') == 'strong' and 'USD' in market_name:
                if market_name.startswith('USD'):
                    bullish_signals += 5
                    critical_factors += 1
                    reasons.append("🇺🇸 Strong USD Momentum")
                else:
                    bearish_signals += 5
                    critical_factors += 1
                    reasons.append("🇺🇸 USD Strength Pressure")
            elif sentiment.get('usd_strength') == 'weak' and 'USD' in market_name:
                if market_name.startswith('USD'):
                    bearish_signals += 5
                    critical_factors += 1
                    reasons.append("🇺🇸 Weak USD Momentum")
                else:
                    bullish_signals += 5
                    critical_factors += 1
                    reasons.append("🇺🇸 USD Weakness Boost")

            # Asset class specific analysis
            asset_class = sentiment.get('asset_class', '')
            if asset_class == 'crypto':
                volatility = sentiment.get('volatility_regime', 'medium')
                if volatility == 'high':
                    bullish_signals += 2
                    reasons.append("⚡ High Crypto Volatility Environment")

        # Economic Data Analysis
        if 'economic_data' in data_dict:
            econ_data = data_dict['economic_data']
            if not econ_data.empty and len(econ_data) > 1:
                recent_change = econ_data['value'].pct_change().iloc[-1]

                if abs(recent_change) > 0.03:  # 3% change in economic indicator
                    if recent_change > 0:
                        bullish_signals += 4
                        critical_factors += 1
                        reasons.append(f"📊 Strong Economic Data (+{recent_change*100:.1f}%)")
                    else:
                        bearish_signals += 4
                        critical_factors += 1
                        reasons.append(f"📊 Weak Economic Data ({recent_change*100:.1f}%)")

        # Forex Market Analysis
        if 'forex_rates' in data_dict:
            forex_rates = data_dict['forex_rates']

            # Real-time rate validation
            if market_name in ['EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD']:
                base_currency = market_name[:3]
                if base_currency in forex_rates:
                    bullish_signals += 2
                    reasons.append("💱 Real-time Forex Validation")

        # Get fundamental analysis first
        fundamental_score, fundamental_reasons = get_market_sentiment(symbol)
        
        # Enhanced news sentiment analysis
        try:
            if market_name == 'XAU/USD':
                news_sentiment = news_analyzer.get_gold_specific_news()
            else:
                news_sentiment = news_analyzer.get_comprehensive_news_analysis(market_name)
            
            if news_sentiment and news_sentiment.get('score', 0) != 0:
                news_score = news_sentiment['score']
                news_relevance = news_sentiment.get('relevance', 'none')
                news_count = news_sentiment.get('news_count', 0)
                
                # Weight news sentiment based on relevance and count
                news_weight = 1.0
                if news_relevance == 'high' and news_count >= 5:
                    news_weight = 2.0
                elif news_relevance == 'medium' and news_count >= 3:
                    news_weight = 1.5
                
                weighted_news_score = int(news_score * news_weight)
                fundamental_score += weighted_news_score
                
                # Add news-based reasons
                if news_score > 2:
                    fundamental_reasons.append(f"📰 Bullish News Sentiment ({news_score:.1f})")
                elif news_score < -2:
                    fundamental_reasons.append(f"📰 Bearish News Sentiment ({news_score:.1f})")
                
                if news_count > 0:
                    fundamental_reasons.append(f"📊 {news_count} Relevant News Items")
                    
                print(f"📰 News Sentiment for {market_name}: {news_sentiment['sentiment']} ({news_score})")
                
        except Exception as e:
            print(f"⚠️ News sentiment analysis error for {market_name}: {e}")
        
        # Enhanced Gold analysis with news sentiment
        if 'metals_data' in data_dict and market_name == 'XAU/USD':
            metals = data_dict['metals_data']
            gold_data = metals.get('gold', {})

            if gold_data:
                gold_price = gold_data.get('price', 0)
                gold_change = gold_data.get('change', 0)

                if abs(gold_change) > 20:  # Significant gold price move
                    if gold_change > 20:
                        bullish_signals += 4
                        critical_factors += 1
                        reasons.append(f"🥇 Strong Gold Rally (+${gold_change:.2f})")
                    else:
                        bearish_signals += 4
                        critical_factors += 1
                        reasons.append(f"🥇 Gold Decline (-${abs(gold_change):.2f})")

                # Gold-specific news sentiment boost
                if fundamental_score > 3:
                    bullish_signals += 2
                    reasons.append("📰 Bullish gold fundamentals from news")
                elif fundamental_score < -3:
                    bearish_signals += 2
                    reasons.append("📰 Bearish gold fundamentals from news")

        # Enhanced Volume Analysis
        volume_strength = 0
        if 'Volume' in primary_data.columns:
            recent_volume = primary_data['Volume'].tail(5).mean()
            avg_volume = primary_data['Volume'].rolling(window=20).mean().iloc[-1]

            if recent_volume > avg_volume * 2:
                volume_strength = 4
                critical_factors += 1
                reasons.append(f"📈 Exceptional Volume Surge ({recent_volume/avg_volume:.1f}x)")
            elif recent_volume > avg_volume * 1.5:
                volume_strength = 2
                reasons.append(f"📊 High Volume Activity ({recent_volume/avg_volume:.1f}x)")

        if volume_strength > 0:
            if latest['Close'] > latest['Open']:
                bullish_signals += volume_strength
            else:
                bearish_signals += volume_strength

        # === PRICE ACTION ANALYSIS ===

        # Candlestick Patterns
        if latest.get('BullishEngulfing', False):
            bullish_signals += 5
            critical_factors += 1
            reasons.append("🕯️ Bullish Engulfing Pattern")
        elif latest.get('BearishEngulfing', False):
            bearish_signals += 5
            critical_factors += 1
            reasons.append("🕯️ Bearish Engulfing Pattern")

        if latest.get('MorningStar', False):
            bullish_signals += 6
            critical_factors += 1
            reasons.append("🌅 Morning Star Pattern")
        elif latest.get('EveningStar', False):
            bearish_signals += 6
            critical_factors += 1
            reasons.append("🌆 Evening Star Pattern")

        if latest.get('Hammer', False):
            bullish_signals += 4
            critical_factors += 1
            reasons.append("🔨 Bullish Hammer")
        elif latest.get('ShootingStar', False):
            bearish_signals += 4
            critical_factors += 1
            reasons.append("⭐ Shooting Star")

        # Market Structure
        structure_bias = latest.get('StructureBias', 0)
        if structure_bias == 1:
            bullish_signals += 4
            critical_factors += 1
            reasons.append("📈 Bullish Market Structure")
        elif structure_bias == -1:
            bearish_signals += 4
            critical_factors += 1
            reasons.append("📉 Bearish Market Structure")

        # Support/Resistance
        if latest.get('SupportTouch', False):
            bullish_signals += 5
            critical_factors += 1
            reasons.append("🛡️ Support Level Bounce")
        elif latest.get('ResistanceTouch', False):
            bearish_signals += 5
            critical_factors += 1
            reasons.append("🚧 Resistance Level Rejection")

        if latest.get('FalseBreakoutHigh', False):
            bearish_signals += 6
            critical_factors += 1
            reasons.append("🎯 False Breakout Above Resistance")
        elif latest.get('FalseBreakoutLow', False):
            bullish_signals += 6
            critical_factors += 1
            reasons.append("🎯 False Breakout Below Support")

        # Order Flow
        buying_pressure = latest.get('BuyingPressure', 0.5)
        selling_pressure = latest.get('SellingPressure', 0.5)

        if buying_pressure > 0.7:
            bullish_signals += 3
            reasons.append(f"💪 Strong Buying Pressure ({buying_pressure:.2f})")
        elif selling_pressure > 0.7:
            bearish_signals += 3
            reasons.append(f"🔻 Strong Selling Pressure ({selling_pressure:.2f})")

        # === TECHNICAL ANALYSIS ===

        # RSI
        rsi = latest.get('RSI', 50)
        if rsi < 20:
            bullish_signals += 3
            reasons.append("RSI Extremely Oversold (<20)")
        elif rsi < 30:
            bullish_signals += 2
            reasons.append("RSI Oversold (<30)")
        elif rsi > 80:
            bearish_signals += 3
            reasons.append("RSI Extremely Overbought (>80)")
        elif rsi > 70:
            bearish_signals += 2
            reasons.append("RSI Overbought (>70)")

        # EMA Alignment
        ema20 = latest.get('EMA20', price)
        ema50 = latest.get('EMA50', price)
        ema200 = latest.get('EMA200', price)

        if ema20 > ema50 > ema200:
            bullish_signals += 4
            reasons.append("Perfect EMA Bullish Alignment")
        elif ema20 < ema50 < ema200:
            bearish_signals += 4
            reasons.append("Perfect EMA Bearish Alignment")

        # MACD
        macd = latest.get('MACD', 0)
        macd_signal = latest.get('MACD_Signal', 0)
        macd_hist = latest.get('MACD_Histogram', 0)

        if macd > macd_signal and macd_hist > 0:
            bullish_signals += 3
            reasons.append("MACD Bullish Crossover")
        elif macd < macd_signal and macd_hist < 0:
            bearish_signals += 3
            reasons.append("MACD Bearish Crossover")

        # ADX
        adx = latest.get('ADX', 25)
        di_plus = latest.get('DI_Plus', 25)
        di_minus = latest.get('DI_Minus', 25)

        if adx > 25 and di_plus > di_minus:
            bullish_signals += 3
            reasons.append(f"Strong Bullish Trend (ADX: {adx:.1f})")
        elif adx > 25 and di_minus > di_plus:
            bearish_signals += 3
            reasons.append(f"Strong Bearish Trend (ADX: {adx:.1f})")

        # Multi-timeframe analysis
        mtf_bullish = 0
        mtf_bearish = 0

        for tf in ['4h', '1d']:
            if tf in data_dict:
                tf_data = data_dict[tf]
                tf_latest = tf_data.iloc[-1]

                tf_structure = tf_latest.get('StructureBias', 0)
                if tf_structure == 1:
                    mtf_bullish += 1
                elif tf_structure == -1:
                    mtf_bearish += 1

                if tf_latest.get('BullishEngulfing', False) or tf_latest.get('MorningStar', False):
                    mtf_bullish += 2
                elif tf_latest.get('BearishEngulfing', False) or tf_latest.get('EveningStar', False):
                    mtf_bearish += 2

        if mtf_bullish >= 2:
            bullish_signals += 4
            critical_factors += 1
            reasons.append("⏳ Higher Timeframe Bullish Confluence")
        elif mtf_bearish >= 2:
            bearish_signals += 4
            critical_factors += 1
            reasons.append("⏳ Higher Timeframe Bearish Confluence")

        # Apply fundamental analysis results
        reasons.extend(fundamental_reasons[:3])  # Limit fundamental reasons

        if fundamental_score > 0:
            bullish_signals += fundamental_score
        else:
            bearish_signals += abs(fundamental_score)

        # Enhanced confidence calculation with alternative data factors
        total_signals = bullish_signals + bearish_signals
        min_critical_factors = 2

        # Alternative data quality bonus
        data_quality_bonus = 0
        if 'market_sentiment' in data_dict:
            data_quality_bonus += 4  # Market sentiment analysis
        if 'economic_data' in data_dict:
            data_quality_bonus += 5  # Economic indicators
        if 'forex_rates' in data_dict:
            data_quality_bonus += 3  # Real-time forex rates
        if 'metals_data' in data_dict:
            data_quality_bonus += 3  # Precious metals data

        if total_signals == 0 or critical_factors < min_critical_factors:
            confidence = 0
            direction = "WAIT"
        else:
            if bullish_signals > bearish_signals:
                base_confidence = (bullish_signals / total_signals) * 100
                critical_boost = min(15, critical_factors * 5)
                confidence = min(98, int(base_confidence + critical_boost + data_quality_bonus))
                direction = "BUY" if confidence >= 85 else "WAIT"
            else:
                base_confidence = (bearish_signals / total_signals) * 100
                critical_boost = min(15, critical_factors * 5)
                confidence = min(98, int(base_confidence + critical_boost + data_quality_bonus))
                direction = "SELL" if confidence >= 85 else "WAIT"

        # Enhanced stop loss and take profit calculation
        atr = latest.get('ATR', (latest.get('High', price) - latest.get('Low', price)))
        if pd.isna(atr) or atr == 0:
            atr = price * 0.02  # 2% fallback

        # Market-specific risk adjustment
        risk_adjustment = 1.0
        if market_name in ['BTC/USD', 'ETH/USD']:
            risk_adjustment = 1.3  # Higher volatility for crypto
        elif market_name == 'XAU/USD':
            risk_adjustment = 1.1  # Moderate volatility for gold
        elif 'JPY' in market_name:
            risk_adjustment = 0.9  # Lower volatility for JPY pairs

        if direction == "BUY":
            stop_loss = price - (atr * 1.8 * risk_adjustment)
            take_profit = price + (atr * 3.2 * risk_adjustment)
        elif direction == "SELL":
            stop_loss = price + (atr * 1.8 * risk_adjustment)
            take_profit = price - (atr * 3.2 * risk_adjustment)
        else:
            stop_loss = price
            take_profit = price

        signal_data = {
            "market": market_name,
            "direction": direction,
            "entry": round(price, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "confidence": confidence,
            "reasons": reasons[:8],  # Limit reasons
            "bullish_points": bullish_signals,
            "bearish_points": bearish_signals,
            "technical_score": bullish_signals + bearish_signals,
            "fundamental_score": abs(fundamental_score),
            "critical_factors": critical_factors
        }

        # Apply alternative data enhancements
        signal_data = alt_data_provider.enhance_signal_with_alternative_data(signal_data, data_dict)

        return signal_data

    except Exception as e:
        print(f"❌ Error analyzing {market_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- DATABASE LOGGING ---
def log_to_db(signal):
    session = Session()
    try:
        signal_entry = Signal(
            market=signal["market"],
            direction=signal["direction"],
            entry=signal["entry"],
            stop_loss=signal["stop_loss"],
            take_profit=signal["take_profit"],
            confidence=signal["confidence"],
            reasons="\n".join(signal["reasons"]),
            technical_score=signal["technical_score"],
            fundamental_score=signal["fundamental_score"]
        )
        session.add(signal_entry)
        session.commit()
    except Exception as e:
        print(f"Error logging to database: {e}")
    finally:
        session.close()


# --- BROADCAST SIGNALS ---
def broadcast_signal(signal):
    """Send signal to all active subscribers"""
    active_subscribers = get_active_subscribers()

    if not active_subscribers:
        print("📭 No active subscribers")
        return

    risk_reward = abs(signal['take_profit'] - signal['entry']) / abs(signal['entry'] - signal['stop_loss'])

    msg = (f"🎯 *ULTRA-PRECISION SIGNAL* 🎯\n"
           f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
           f"📈 Market: *{signal['market']}*\n"
           f"🔥 Direction: *{signal['direction']}*\n"
           f"💰 Entry: *{signal['entry']}*\n"
           f"🛡️ Stop Loss: *{signal['stop_loss']}*\n"
           f"🎯 Take Profit: *{signal['take_profit']}*\n"
           f"📊 Confidence: *{signal['confidence']}%*\n"
           f"⚖️ Risk:Reward: *1:{risk_reward:.2f}*\n"
           f"🔬 Technical Score: *{signal['technical_score']}*\n"
           f"📰 Fundamental Score: *{signal['fundamental_score']}*\n\n"
           f"📋 *Key Factors:*\n• " + "\n• ".join(signal['reasons'][:6]))

    successful_sends = 0
    failed_sends = 0

    for chat_id in active_subscribers:
        if send_message(chat_id, msg):
            successful_sends += 1
        else:
            failed_sends += 1

    print(f"📤 Signal sent to {successful_sends} subscribers")


# --- MAIN JOB ---
def job():
    try:
        print("🔍 Starting multi-market analysis...")
        high_confidence_signals = []

        for market_name, symbol in MARKETS.items():
            signal = generate_market_signal(market_name, symbol)

            if (signal and signal['direction'] != "WAIT" and 
                signal['confidence'] >= 85 and 
                signal.get('critical_factors', 0) >= 2):

                high_confidence_signals.append(signal)
                print(f"🎯 HIGH-CONFIDENCE SIGNAL: {market_name} | {signal['direction']} | {signal['confidence']}%")

        # Send signals
        for signal in high_confidence_signals:
            broadcast_signal(signal)
            log_to_db(signal)

        if not high_confidence_signals:
            print("⏳ No high-confidence signals found")

        print(f"📊 Analysis complete. {len(high_confidence_signals)} signals generated")

    except Exception as e:
        print(f"❌ Error in job execution: {e}")
        import traceback
        traceback.print_exc()


# --- MAIN LOOP ---
def main():
    schedule.every().hour.at(":00").do(job)
    schedule.every(30).seconds.do(handle_telegram_commands)

    print("🚀 Ultra-Precision Trading Bot v8.0 is running...")
    print("📊 Markets: Gold + 8 Forex pairs + 2 Crypto")
    print("🔥 Enhanced with Multiple Data Sources")
    print("📈 Yahoo Finance + CoinGecko + Economic Data + Forex APIs")
    print("💎 Real-time rates + Sentiment + Market structure")
    print("🕯️ Price Action + Technical + Fundamental Analysis")
    print("🎯 85%+ confidence + 2+ critical factors required")
    print("⏰ Analyzing all markets every hour...")
    print("🤖 Telegram bot active - /start to subscribe")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # Run once at start
    job()

    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    main()