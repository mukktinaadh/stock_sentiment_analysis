import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockSentimentAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.analyzer = SentimentIntensityAnalyzer()

    def get_stock_data(self, period='1mo'):
        """Fetch stock price data"""
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=period)
        return df

    def analyze_sentiment(self, texts):
        """Analyze sentiment of text data"""
        sentiments = []
        for text in texts:
            score = self.analyzer.polarity_scores(text)
            sentiments.append(score)
        return pd.DataFrame(sentiments)

    def generate_sample_headlines(self):
        """Generate sample financial headlines for demonstration"""
        headlines = [
            f"{self.ticker} reports strong quarterly earnings beating expectations",
            f"Market analysts downgrade {self.ticker} stock amid concerns",
            f"{self.ticker} announces major partnership deal",
            f"Investors worried about {self.ticker} future prospects",
            f"{self.ticker} stock hits new 52-week high",
            f"Regulatory challenges facing {self.ticker}",
            f"{self.ticker} CEO optimistic about growth trajectory",
            f"Weak guidance from {self.ticker} disappoints investors",
            f"{self.ticker} expands into new markets",
            f"Competition intensifies for {self.ticker}"
        ]
        return headlines

    def visualize_results(self, sentiment_df, stock_df):
        """Create visualization dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Sentiment distribution
        sentiment_df['compound'].plot(kind='hist', bins=20, ax=axes[0, 0], color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Sentiment Score Distribution')
        axes[0, 0].set_xlabel('Compound Sentiment Score')

        # Stock price trend
        stock_df['Close'].plot(ax=axes[0, 1], color='green', linewidth=2)
        axes[0, 1].set_title(f'{self.ticker} Stock Price Trend')
        axes[0, 1].set_ylabel('Price ($)')

        # Sentiment components
        sentiment_df[['pos', 'neu', 'neg']].mean().plot(kind='bar', ax=axes[1, 0], color=['green', 'gray', 'red'])
        axes[1, 0].set_title('Average Sentiment Components')
        axes[1, 0].set_xticklabels(['Positive', 'Neutral', 'Negative'], rotation=0)

        # Volume
        stock_df['Volume'].plot(ax=axes[1, 1], color='orange', alpha=0.7)
        axes[1, 1].set_title('Trading Volume')
        axes[1, 1].set_ylabel('Volume')

        plt.tight_layout()
        plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved as 'sentiment_analysis_results.png'")
        plt.show()

    def run_analysis(self):
        """Execute full analysis pipeline"""
        print(f"\n{'='*60}")
        print(f"Stock Sentiment Analysis for {self.ticker}")
        print(f"{'='*60}\n")

        # Get stock data
        print("📊 Fetching stock data...")
        stock_df = self.get_stock_data()
        print(f"✓ Retrieved {len(stock_df)} days of stock data")

        # Generate and analyze headlines
        print("\n📰 Analyzing news sentiment...")
        headlines = self.generate_sample_headlines()
        sentiment_df = self.analyze_sentiment(headlines)

        # Display results
        print(f"\n{'─'*60}")
        print("Sentiment Analysis Results:")
        print(f"{'─'*60}")
        print(f"Average Compound Score: {sentiment_df['compound'].mean():.3f}")
        print(f"Positive Sentiment: {sentiment_df['pos'].mean():.3f}")
        print(f"Neutral Sentiment: {sentiment_df['neu'].mean():.3f}")
        print(f"Negative Sentiment: {sentiment_df['neg'].mean():.3f}")

        # Save results
        results_df = pd.DataFrame({
            'headline': headlines,
            'compound': sentiment_df['compound'],
            'positive': sentiment_df['pos'],
            'neutral': sentiment_df['neu'],
            'negative': sentiment_df['neg']
        })
        results_df.to_csv('sentiment_results.csv', index=False)
        print(f"\n✓ Results saved to 'sentiment_results.csv'")

        # Visualize
        print("\n📈 Generating visualizations...")
        self.visualize_results(sentiment_df, stock_df)

        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    analyzer = StockSentimentAnalyzer('AAPL')
    analyzer.run_analysis()
