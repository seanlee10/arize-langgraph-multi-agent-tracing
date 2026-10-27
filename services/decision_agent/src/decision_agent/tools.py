"""LangChain tools for the decision agent; bodies delegate to the indicator and market-data modules."""

from typing import Annotated, Any, Literal

import pandas as pd
from langchain_core.tools import BaseTool, tool
from pydantic import Field

from decision_agent import market_data
from decision_agent.indicators import bollinger_bands as compute_bollinger_bands
from decision_agent.indicators import force_index as compute_force_index
from decision_agent.indicators import summarize_insider_transactions

Ticker = Annotated[str, Field(description="Stock ticker symbol, e.g. AAPL")]


@tool
def force_index(ticker: Ticker, period: Annotated[int, Field(ge=1, le=50)] = 13) -> dict[str, Any]:
    """Elder's Force Index (price change x volume) over ~6 months of daily data, smoothed with an EMA. Positive values mean buying pressure, negative values mean selling pressure."""
    history = market_data.fetch_price_history(ticker)
    return {"ticker": ticker, **compute_force_index(history["Close"], history["Volume"], period)}


@tool
def bollinger_bands(
    ticker: Ticker,
    window: Annotated[int, Field(ge=2, le=100)] = 20,
    num_std: Annotated[float, Field(ge=0.5, le=4)] = 2.0,
) -> dict[str, Any]:
    """Bollinger Bands on daily closes: upper/middle/lower band, %B and bandwidth. %B >= 1 is overbought, %B <= 0 is oversold."""
    history = market_data.fetch_price_history(ticker)
    return {"ticker": ticker, **compute_bollinger_bands(history["Close"], window, num_std)}


@tool
def insider_trades(ticker: Ticker, days: Annotated[int, Field(ge=1, le=365)] = 90) -> dict[str, Any]:
    """Insider purchases and sales over the last N days: counts, net shares, dollar values and the largest transactions."""
    transactions = market_data.fetch_insider_transactions(ticker)
    summary = summarize_insider_transactions(transactions, days=days, today=pd.Timestamp.today())
    return {"ticker": ticker, **summary}


@tool
def submit_decision(
    decision: Literal["buy", "hold", "sell"],
    confidence: Annotated[float, Field(ge=0, le=1)],
    rationale: str,
) -> str:
    """Submit the final trading decision. Call this exactly once, when you are done analysing."""
    return "submitted"


INDICATOR_TOOLS: list[BaseTool] = [force_index, bollinger_bands, insider_trades]
