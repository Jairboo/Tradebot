
import os

# Telegram Bot Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8018851274:AAHWZEhTpC5GMIdDmnqAUApEVS2bdxlIZUA")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7104833764")

# Trading Configuration
CONFIDENCE_SCALE = 10
ECONOMIC_CALENDAR_API = "https://api.tradingeconomics.com/calendar"

# Markets Configuration
MARKETS = {
    "XAU/USD": "GC=F",  # Gold
    "EUR/USD": "EURUSD=X",  # Euro/USD
    "USD/JPY": "USDJPY=X",  # USD/Japanese Yen
    "GBP/USD": "GBPUSD=X",  # British Pound/USD
    "USD/CHF": "USDCHF=X",  # USD/Swiss Franc
    "AUD/USD": "AUDUSD=X",  # Australian Dollar/USD
    "NZD/USD": "NZDUSD=X",  # New Zealand Dollar/USD
    "EUR/CHF": "EURCHF=X",  # Euro/Swiss Franc
    "EUR/GBP": "EURGBP=X",  # Euro/British Pound
    "BTC/USD": "BTC-USD",  # Bitcoin
    "ETH/USD": "ETH-USD"  # Ethereum
}
