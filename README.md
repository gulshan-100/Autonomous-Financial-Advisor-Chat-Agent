# Financial Advisor Agent - Mock Dataset

This directory contains comprehensive mock data for the Autonomous Financial Advisor Agent challenge.

## Dataset Overview

| File | Description | Key Data Points |
|------|-------------|-----------------|
| `market_data.json` | Real-time market data snapshot | 40+ stocks, 5 indices, 10 sectors |
| `news_data.json` | Financial news feed | 25 articles with sentiment, scope, and entity tags |
| `portfolios.json` | User portfolio samples | 3 portfolios (Diversified, Sector-heavy, Conservative) |
| `mutual_funds.json` | Mutual fund details | 12 schemes with NAV, holdings, and returns |
| `historical_data.json` | 7-day historical trends | Index/stock history, FII/DII data, market breadth |
| `sector_mapping.json` | Sector-stock relationships | Macro correlations and sector characteristics |

## Data Scenarios Included

### Market Conditions (April 21, 2026)

The dataset simulates a **risk-off market day** with the following characteristics:

- **NIFTY 50**: -1.00% (Bearish)
- **Bank Nifty**: -2.33% (Strong selling due to RBI stance)
- **NIFTY IT**: +1.22% (Outperforming due to US tech earnings)
- **FII**: Net sellers of ₹4,500 crore
- **Market Breadth**: Weak (12 advances vs 38 declines in NIFTY 50)

### Sector Performance

| Sector | Day Change | Sentiment | Key Driver |
|--------|------------|-----------|------------|
| Banking | -2.45% | Bearish | RBI hawkish stance |
| IT | +1.35% | Bullish | US tech earnings, weak rupee |
| Pharma | +0.78% | Bullish | USFDA approvals |
| Metals | -1.50% | Bearish | China demand concerns |
| Realty | -2.10% | Bearish | Interest rate sensitivity |
| FMCG | +0.25% | Neutral | Defensive buying |

### News Categories

1. **Market-Wide** (5 articles)
   - RBI monetary policy
   - FII outflows
   - Global risk-off sentiment
   - Oil price movements

2. **Sector-Specific** (8 articles)
   - US tech earnings (IT positive)
   - China steel demand (Metals negative)
   - Housing sales vs rate concerns (Realty mixed)
   - Government capex push (Infra positive)

3. **Stock-Specific** (12 articles)
   - HDFC Bank results (mixed)
   - Sun Pharma USFDA approval (positive)
   - Infosys mega deal win (positive)
   - Tata Motors EV leadership (positive)

### Edge Cases for Agent Testing

The dataset includes several **conflict scenarios** to test the agent's reasoning:

1. **Positive news + Negative price action**
   - Bajaj Finance: Strong asset quality but stock falling due to sector sentiment
   - HUL: Slightly up despite weak volume growth (defensive buying)

2. **Mixed signals**
   - Reliance: Strong retail but weak Jio subscriber growth
   - Housing sales: Record high but rate concerns dominate
   - ICICI Bank: Improved asset quality but margin compression

3. **Sector vs Stock divergence**
   - Tata Motors: +0.79% vs Auto sector -1.85% (EV leadership)

## Portfolio Profiles

### Portfolio 1: Diversified (Rahul Sharma)
- **Type**: Well-balanced across sectors
- **Day P&L**: -0.44% (₹-12,785)
- **Concentration Risk**: None
- **Max Single Stock Weight**: 7.17% (TCS)
- **Asset Mix**: 38% Stocks, 62% Mutual Funds

### Portfolio 2: Sector-Concentrated (Priya Patel)
- **Type**: Banking & Financial Services heavy
- **Day P&L**: -2.73% (₹-57,390)
- **Concentration Risk**: CRITICAL (91.58% in Banking + FS)
- **Max Single Stock Weight**: 22.62% (HDFC Bank)
- **Asset Mix**: 91% Stocks, 9% Mutual Funds

### Portfolio 3: Conservative (Arun Krishnamurthy)
- **Type**: Mutual fund heavy with defensive stocks
- **Day P&L**: -0.04% (₹-1,758)
- **Concentration Risk**: None
- **Max Single Stock Weight**: 5.19% (ITC)
- **Asset Mix**: 21% Stocks, 79% Mutual Funds (34% Debt Funds)

## Usage in Agent

### Loading Data

```python
from data_loader import DataLoader

# Initialize loader
loader = DataLoader("./data")

# Load all data
market_data = loader.get_market_data()
news = loader.get_news()
portfolios = loader.get_portfolios()

# Get specific portfolio
portfolio = loader.get_portfolio("PORTFOLIO_002")

# Get sector info
sector = loader.get_sector_info("BANKING")

# Get stock with news impact
stock_analysis = loader.get_stock_with_context("HDFCBANK")
```

### Expected Agent Outputs

For **Portfolio 2** (Banking concentrated), the agent should identify:

1. **Primary Impact**: RBI's hawkish stance hitting all banking holdings
2. **Concentration Risk Alert**: 91.58% exposure to interest-rate sensitive sectors
3. **Causal Chain**: 
   ```
   RBI Hawkish Stance → Banking Sector -2.45% → 
   HDFC Bank -3.51% (largest holding) → Portfolio -2.73%
   ```
4. **Conflicting Signals**: 
   - ICICI Bank asset quality improved but NIM compressed
   - Bajaj Finance strong guidance but sector headwinds

## Data Schema Reference

### Stock Object
```json
{
  "symbol": "HDFCBANK",
  "name": "HDFC Bank Ltd",
  "sector": "BANKING",
  "current_price": 1542.30,
  "change_percent": -3.51,
  "volume": 15234500,
  "beta": 1.15
}
```

### News Object
```json
{
  "id": "NEWS001",
  "headline": "...",
  "sentiment": "NEGATIVE",
  "sentiment_score": -0.72,
  "scope": "MARKET_WIDE|SECTOR_SPECIFIC|STOCK_SPECIFIC",
  "impact_level": "HIGH|MEDIUM|LOW",
  "entities": {
    "sectors": ["BANKING"],
    "stocks": ["HDFCBANK"],
    "indices": ["BANKNIFTY"]
  },
  "causal_factors": ["..."]
}
```

### Portfolio Holding Object
```json
{
  "symbol": "HDFCBANK",
  "quantity": 100,
  "avg_buy_price": 1520.00,
  "current_price": 1542.30,
  "weight_in_portfolio": 5.36,
  "day_change_percent": -3.51
}
```

## Extending the Dataset

To add more scenarios:

1. **Add New Stocks**: Update `market_data.json` → `stocks` section
2. **Add News**: Update `news_data.json` → `news` array
3. **Add Portfolios**: Update `portfolios.json` → `portfolios` section
4. **Add Sector**: Update `sector_mapping.json` → `sectors` section
