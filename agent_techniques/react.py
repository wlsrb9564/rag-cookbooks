# %% [markdown]
# # **ReAct (Reasoning and Action)**
#
# ReAct (Reasoning + Acting) is a paradigm for integrating reasoning and acting within large
# language models (LLMs) to enhance decision-making, task-solving efficiency, and external tool
# interaction. By interleaving reasoning traces with actions, ReAct allows models to:
#
# - Maintain a coherent thought process throughout multi-step tasks.
# - Reduce hallucination by verifying facts through external knowledge retrieval.
# - Improve interpretability by making reasoning steps explicit.
# - Adapt dynamically to new information by integrating real-time data from APIs.
# - Enable complex decision-making in uncertain environments by iteratively refining outputs.
#
# ### Key Characteristics of ReAct:
# 1. **Reasoning**: The model generates intermediate thoughts to break down problems into logical steps.
# 2. **Acting**: The model interacts with tools, APIs, or external knowledge bases to gather and process relevant data.
# 3. **Interleaved Execution**: Reasoning and acting steps alternate, ensuring the model adjusts based on observations.
# 4. **Enhanced Transparency**: Users can inspect both reasoning traces and action logs.
# 5. **Versatile Application**: Used in finance, healthcare, research, and various AI-driven workflows.
#
# Paper: https://arxiv.org/pdf/2210.03629

# %% [markdown]
# ## **Initial Setup**

# %%
import os
from dotenv import load_dotenv
load_dotenv()

# %% [markdown]
# ## **Tools**

# %%
# web search tool
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(k=3)

# %%
# define the LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

# %%
# define tools
from langchain_core.tools import tool
import yfinance as yf
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from datetime import datetime

@tool
def get_current_date(tool_input=None) -> str:
    """Returns the current date in YYYY-MM-DD format.

    The LLM can use this to understand time-based queries such as
    'past 1 month' or 'last 3 weeks' in relation to today's date.
    """
    return datetime.now().strftime("%Y-%m-%d")

@tool
def web_search(query: str) -> str:
    """Tool for performing web search."""
    return search.invoke(query)

@tool
def get_stock_price(ticker: str) -> float:
    """Fetches the latest closing price of a given stock ticker.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL').
    """
    ticker_data = yf.Ticker(ticker)
    history = ticker_data.history(period="1y")
    return history['Close']

@tool
def yahoo_finance_news(query: str) -> str:
    """Fetches news articles related to a query from Yahoo Finance.

    Args:
        query: The search query (e.g., company name or ticker).
    """
    news_tool = YahooFinanceNewsTool()
    return news_tool.invoke(query)

# %%
# list of tools
tools = [get_stock_price, web_search, yahoo_finance_news, get_current_date]

# %% [markdown]
# ## **LangChain ReAct Agent**

# %%
# get the ReAct prompt from LangChain Hub (requires LANGCHAIN_API_KEY)
from langchain import hub
prompt = hub.pull("hwchase17/react")

# %%
# construct the ReAct agent
from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools, prompt)

# %%
# create agent executor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# %%
agent_executor.invoke({"input": "stocks analysis for Elon musk company for last month and latest news about company?"})
