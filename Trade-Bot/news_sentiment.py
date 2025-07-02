
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import time
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class NewsAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # News sources configuration
        self.news_sources = {
            'forex_factory': 'https://www.forexfactory.com/calendar',
            'investing': 'https://www.investing.com/news/forex-news',
            'marketwatch': 'https://www.marketwatch.com/markets/currencies',
            'reuters': 'https://www.reuters.com/markets/currencies',
            'bloomberg': 'https://www.bloomberg.com/markets/currencies'
        }
        
        # Currency impact keywords
        self.currency_keywords = {
            'USD': ['dollar', 'fed', 'federal reserve', 'interest rate', 'inflation', 'employment', 'gdp'],
            'EUR': ['euro', 'ecb', 'european central bank', 'eurozone', 'inflation', 'employment'],
            'GBP': ['pound', 'sterling', 'boe', 'bank of england', 'brexit', 'uk'],
            'JPY': ['yen', 'boj', 'bank of japan', 'japan', 'inflation'],
            'CHF': ['franc', 'snb', 'swiss national bank', 'switzerland'],
            'AUD': ['australian dollar', 'rba', 'reserve bank australia', 'australia'],
            'NZD': ['new zealand dollar', 'rbnz', 'reserve bank new zealand'],
            'XAU': ['gold', 'precious metals', 'safe haven', 'inflation hedge', 'treasury yields']
        }
        
        # Sentiment keywords
        self.bullish_keywords = [
            'surge', 'rally', 'strengthen', 'boost', 'rise', 'gain', 'bullish', 'positive',
            'optimistic', 'confidence', 'growth', 'recovery', 'improvement', 'higher',
            'support', 'upgrade', 'beat expectations', 'strong data', 'resilient'
        ]
        
        self.bearish_keywords = [
            'fall', 'decline', 'weaken', 'drop', 'bearish', 'negative', 'pessimistic',
            'concern', 'worry', 'recession', 'slowdown', 'deterioration', 'lower',
            'pressure', 'downgrade', 'miss expectations', 'weak data', 'volatile'
        ]

    def get_forex_factory_news(self):
        """Scrape news from Forex Factory"""
        try:
            print("📰 Fetching Forex Factory news...")
            response = requests.get(self.news_sources['forex_factory'], 
                                  headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Find calendar events
            events = soup.find_all('tr', class_='calendar_row')
            
            for event in events[:20]:  # Limit to recent events
                try:
                    time_elem = event.find('td', class_='calendar__time')
                    event_elem = event.find('td', class_='calendar__event')
                    impact_elem = event.find('td', class_='calendar__impact')
                    
                    if event_elem and impact_elem:
                        event_text = event_elem.get_text(strip=True)
                        impact = impact_elem.find('span')
                        impact_level = impact.get('title', '') if impact else ''
                        
                        if 'High' in impact_level or 'Medium' in impact_level:
                            news_items.append({
                                'source': 'Forex Factory',
                                'title': event_text,
                                'impact': impact_level,
                                'time': datetime.now(),
                                'relevance': 'high' if 'High' in impact_level else 'medium'
                            })
                except:
                    continue
            
            print(f"✅ Found {len(news_items)} Forex Factory events")
            return news_items
            
        except Exception as e:
            print(f"❌ Error fetching Forex Factory news: {e}")
            return []

    def get_investing_news(self):
        """Get news from Investing.com"""
        try:
            print("📰 Fetching Investing.com news...")
            response = requests.get(self.news_sources['investing'], 
                                  headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            news_items = []
            
            # Find news articles
            articles = soup.find_all('div', class_='largeTitle')
            
            for article in articles[:15]:
                try:
                    title_elem = article.find('a')
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        news_items.append({
                            'source': 'Investing.com',
                            'title': title,
                            'impact': 'medium',
                            'time': datetime.now(),
                            'relevance': 'medium'
                        })
                except:
                    continue
            
            print(f"✅ Found {len(news_items)} Investing.com articles")
            return news_items
            
        except Exception as e:
            print(f"❌ Error fetching Investing.com news: {e}")
            return []

    def analyze_news_sentiment(self, news_items, currency_pair):
        """Analyze sentiment for specific currency pair"""
        if not news_items:
            return {'score': 0, 'relevance': 'none', 'news_count': 0, 'sentiment': 'neutral'}
        
        base_currency = currency_pair[:3]
        quote_currency = currency_pair[4:7] if len(currency_pair) > 6 else 'USD'
        
        relevant_news = []
        total_sentiment = 0
        
        for news in news_items:
            title = news['title'].lower()
            relevance_score = 0
            
            # Check relevance to currencies
            base_keywords = self.currency_keywords.get(base_currency, [])
            quote_keywords = self.currency_keywords.get(quote_currency, [])
            
            for keyword in base_keywords:
                if keyword in title:
                    relevance_score += 2
            
            for keyword in quote_keywords:
                if keyword in title:
                    relevance_score -= 1  # Negative for quote currency strength
            
            # General forex relevance
            forex_terms = ['forex', 'currency', 'exchange rate', 'central bank', 'monetary policy']
            for term in forex_terms:
                if term in title:
                    relevance_score += 1
            
            if relevance_score > 0:
                # Analyze sentiment using TextBlob
                blob = TextBlob(news['title'])
                polarity = blob.sentiment.polarity
                
                # Enhance with keyword analysis
                sentiment_score = polarity
                
                for keyword in self.bullish_keywords:
                    if keyword in title:
                        sentiment_score += 0.2
                
                for keyword in self.bearish_keywords:
                    if keyword in title:
                        sentiment_score -= 0.2
                
                # Weight by impact and relevance
                impact_weight = 3 if news['impact'] == 'High' else 2 if news['impact'] == 'Medium' else 1
                weighted_sentiment = sentiment_score * relevance_score * impact_weight
                
                relevant_news.append({
                    'title': news['title'],
                    'sentiment': sentiment_score,
                    'relevance': relevance_score,
                    'weighted_sentiment': weighted_sentiment
                })
                
                total_sentiment += weighted_sentiment
        
        if not relevant_news:
            return {'score': 0, 'relevance': 'none', 'news_count': 0, 'sentiment': 'neutral'}
        
        # Calculate final sentiment score
        avg_sentiment = total_sentiment / len(relevant_news)
        
        # Normalize to -10 to +10 scale
        normalized_score = max(-10, min(10, avg_sentiment * 10))
        
        # Determine sentiment category
        if normalized_score > 3:
            sentiment_category = 'bullish'
        elif normalized_score < -3:
            sentiment_category = 'bearish'
        else:
            sentiment_category = 'neutral'
        
        return {
            'score': round(normalized_score, 2),
            'relevance': 'high' if len(relevant_news) > 5 else 'medium' if len(relevant_news) > 2 else 'low',
            'news_count': len(relevant_news),
            'sentiment': sentiment_category,
            'news_items': relevant_news[:5]  # Top 5 relevant news
        }

    def get_comprehensive_news_analysis(self, currency_pair):
        """Get comprehensive news analysis for currency pair"""
        try:
            print(f"📊 Analyzing news sentiment for {currency_pair}...")
            
            all_news = []
            
            # Collect news from multiple sources
            forex_factory_news = self.get_forex_factory_news()
            investing_news = self.get_investing_news()
            
            all_news.extend(forex_factory_news)
            all_news.extend(investing_news)
            
            # Analyze sentiment
            sentiment_analysis = self.analyze_news_sentiment(all_news, currency_pair)
            
            # Add additional context
            sentiment_analysis['total_news_sources'] = len([n for n in all_news if n])
            sentiment_analysis['analysis_time'] = datetime.now()
            
            print(f"✅ News analysis complete for {currency_pair}")
            print(f"   Sentiment: {sentiment_analysis['sentiment']} ({sentiment_analysis['score']})")
            print(f"   Relevant news: {sentiment_analysis['news_count']}")
            
            return sentiment_analysis
            
        except Exception as e:
            print(f"❌ Error in news analysis for {currency_pair}: {e}")
            return {'score': 0, 'relevance': 'none', 'news_count': 0, 'sentiment': 'neutral'}

    def get_gold_specific_news(self):
        """Get gold-specific news and analysis"""
        try:
            print("🥇 Analyzing gold-specific news...")
            
            gold_news = []
            
            # Gold-specific news sources (simplified approach)
            gold_keywords = ['gold', 'precious metals', 'safe haven', 'inflation', 'treasury yields', 'fed policy']
            
            # Get general news and filter for gold relevance
            all_news = []
            forex_factory_news = self.get_forex_factory_news()
            investing_news = self.get_investing_news()
            
            all_news.extend(forex_factory_news)
            all_news.extend(investing_news)
            
            for news in all_news:
                title_lower = news['title'].lower()
                for keyword in gold_keywords:
                    if keyword in title_lower:
                        gold_news.append(news)
                        break
            
            # Analyze sentiment specifically for gold
            sentiment_analysis = self.analyze_news_sentiment(gold_news, 'XAU/USD')
            
            # Gold-specific sentiment modifiers
            if sentiment_analysis['score'] != 0:
                # Inflation fears = bullish for gold
                inflation_keywords = ['inflation', 'cpi', 'pce', 'price pressure']
                risk_off_keywords = ['risk off', 'safe haven', 'uncertainty', 'crisis']
                
                for news in gold_news:
                    title_lower = news['title'].lower()
                    
                    for keyword in inflation_keywords:
                        if keyword in title_lower:
                            sentiment_analysis['score'] += 1
                    
                    for keyword in risk_off_keywords:
                        if keyword in title_lower:
                            sentiment_analysis['score'] += 1.5
            
            # Normalize again
            sentiment_analysis['score'] = max(-10, min(10, sentiment_analysis['score']))
            
            print(f"✅ Gold news analysis complete")
            print(f"   Gold sentiment: {sentiment_analysis['sentiment']} ({sentiment_analysis['score']})")
            
            return sentiment_analysis
            
        except Exception as e:
            print(f"❌ Error in gold news analysis: {e}")
            return {'score': 0, 'relevance': 'none', 'news_count': 0, 'sentiment': 'neutral'}
