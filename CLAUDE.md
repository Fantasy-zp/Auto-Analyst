# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Auto-Analyst is a multi-agent industry research system that generates in-depth analysis reports. It combines CrewAI for agent orchestration, Tavily for web search, ChromaDB for vector storage, and FlashRank for reranking results.

## Running the Application

```bash
# Run the Streamlit web interface
streamlit run app.py
```

## Required Environment Variables

Create a `.env` file with:
- `DEEPSEEK_API_KEY` - DeepSeek API key (used as LLM backend via OpenAI-compatible API)
- `TAVILY_API_KEY` - Tavily search API key

## Architecture

### Core Components

- **app.py** - Streamlit frontend that orchestrates the analysis workflow
- **agent_manager.py** - CrewAI multi-agent setup with two agents:
  - Researcher agent: Uses RAG tool to collect industry data
  - Writer agent: Transforms research into structured reports
- **core_utils.py** - `AnalystCore` class providing DeepSeek chat and Tavily search APIs
- **rag_processor.py** - `AdvancedRAG` class implementing two-stage retrieval:
  1. Vector search via ChromaDB (retrieves 10 candidates)
  2. Reranking via FlashRank ms-marco-MiniLM-L-12-v2 (returns top-k)

### Data Flow

1. User enters research topic in Streamlit UI
2. `IndustryAnalystCrew` creates a sequential CrewAI workflow
3. Researcher agent calls `AdvancedRAGSearchTool` which:
   - Uses Tavily to search the web
   - Stores results in ChromaDB
   - Retrieves and reranks for relevant context
4. Writer agent produces final report from research data
5. Results displayed with downloadable markdown

### Key Design Decisions

- DeepSeek API is accessed via OpenAI-compatible client by setting `OPENAI_API_KEY` and `OPENAI_API_BASE` environment variables
- ChromaDB persists to `./chroma_db` directory
- FlashRank model cached in `./opt` directory
- CrewAI telemetry disabled via environment variables
