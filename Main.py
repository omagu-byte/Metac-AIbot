# -*- coding: utf-8 -*-
"""
Pre-Mortem Analysis Bot (v7 - Final)

This script runs a multi-stage forecasting process on Metaculus questions.
FIXED: Implements the required abstract methods and uses the correct `run_forecast_bot`
entry point from the forecasting-tools library.
"""
import argparse
import asyncio
import logging
import os
from datetime import datetime

import aiohttp
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
# FIX 1: Import the new 'run_forecast_bot' function
from forecasting_tools import (ForecastBot, MetaculusApi, MetaculusQuestion,
                               clean_indents, run_forecast_bot)
from newsapi import NewsApiClient

# --- Setup and Configuration ---
load_dotenv()

# API Keys
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")

# Basic Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PreMortemAnalysisBot")

# --- LLM Client (using OpenRouter) ---
class OpenRouterLlm:
    """A simple async client for the OpenRouter API."""
    def __init__(self, model: str):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        self.model = model
        self.api_key = OPENROUTER_API_KEY
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def invoke(self, prompt: str, temperature=0.7, max_tokens=3000) -> str:
        """Sends a prompt to the specified model via OpenRouter."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=headers, json=data, timeout=240) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"OpenRouter Error: Status {resp.status} for model {self.model}. Response: {error_text}")
                        return f"[{self.model} error: {resp.status}] - {error_text}"
                    result = await resp.json()
                    return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"An exception occurred while calling OpenRouter for model {self.model}: {e}")
            return f"Exception during API call to {self.model}: {e}"

# --- Research Clients & Collector ---
class LinkupClient:
    """Client for the LinkUp Job Search API."""
    async def search(self, query: str) -> str:
        if not LINKUP_API_KEY: return "LinkUp API key not set."
        headers = {"Authorization": f"Bearer {LINKUP_API_KEY}"}
        params = {"q": query, "limit": 5}
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.linkup.com/v1/jobs/search", headers=headers, params=params) as resp:
                if resp.status != 200: return f"[LinkUp error: {resp.status}]"
                jobs = (await resp.json()).get("jobs", [])
                return "\n".join([f"- {job.get('title','')} at {job.get('company','')} ({job.get('location','')})" for job in jobs])

class SerpApiClient:
    """Client for SerpAPI."""
    async def search(self, query: str) -> str:
        if not SERPAPI_API_KEY: return "SerpAPI key not set."
        params = {"q": query, "api_key": SERPAPI_API_KEY, "num": 5, "engine": "google"}
        async with aiohttp.ClientSession() as session:
            async with session.get("https://serpapi.com/search", params=params) as resp:
                if resp.status != 200: return f"[SerpAPI error: {resp.status}]"
                items = (await resp.json()).get("organic_results", [])
                return "\n".join([f"- {i.get('title','')}: {i.get('snippet','')}" for i in items])

class WebScraper:
    """Synchronous web scraper using DuckDuckGo and BeautifulSoup."""
    def perform_web_scrape(self, query: str) -> str:
        logger.info(f"[Web Scraper] Searching for: {query}")
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            top_result = soup.find('a', class_='result__a')
            if not top_result or 'href' not in top_result.attrs: return "Web scraping found no clear top result."
            page_url = top_result['href']
            logger.info(f"[Web Scraper] Scraping content from: {page_url}")
            page_response = requests.get(page_url, headers=headers, timeout=10)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            paragraphs = page_soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs[:5]])
            return f"- Top result summary: {content[:1500]}..." if content else "No content found."
        except Exception as e:
            return f"Web scraping failed: {e}"

# --- Main Pre-Mortem Analysis Bot ---
class PreMortemAnalysisBot(ForecastBot):
    """A forecasting bot that uses a pre-mortem analysis framework."""
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    MODEL_CONFIG = {
        "narrative_primary": "anthropic/claude-4-opus",
        "narrative_secondary": "mistralai/mistral-large-latest",
        "analytical_synthesis": "openai/gpt-5",
        "analytical_judgment": "openai/gpt-5",
    }

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Gathers research from all sources."""
        logger.info(f"[{question.id}] Running Research...")
        loop = asyncio.get_running_loop()
        
        def query_newsapi():
            if not NEWSAPI_API_KEY: return "NewsAPI key not set."
            try:
                newsapi = NewsApiClient(api_key=NEWSAPI_API_KEY)
                response = newsapi.get_everything(q=question.question_text, language="en", sort_by="relevancy", page_size=5)
                return "\n".join([f"- {a['title']}: {a.get('description', '')}" for a in response.get("articles", [])])
            except Exception as e:
                return f"NewsAPI failed: {e}"

        tasks = {
            "newsapi": loop.run_in_executor(None, query_newsapi),
            "web_scrape": loop.run_in_executor(None, WebScraper().perform_web_scrape, question.question_text),
            "linkup": LinkupClient().search(question.question_text),
            "serpapi": SerpApiClient().search(question.question_text),
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        research_results = dict(zip(tasks.keys(), results))
        
        return (
            f"--- RESEARCH SUMMARY ---\n"
            f"NewsAPI:\n{research_results.get('newsapi','')}\n\n"
            f"LinkUp (Job Postings):\n{research_results.get('linkup', '')}\n\n"
            f"Web Scrape:\n{research_results.get('web_scrape','')}\n\n"
            f"SerpAPI:\n{research_results.get('serpapi','')}\n"
            f"--- END RESEARCH ---\n"
        )

    async def _run_forecast_on_binary(self, question: MetaculusQuestion, research: str):
        """Runs the pre-mortem for a binary (Yes/No) question."""
        logger.info(f"[{question.id}] Running BINARY analysis...")
        fail_prompt = clean_indents(f"""It is one day after this question's resolution date ({question.resolution_date}). The outcome of "{question.question_text}" was a surprising NO. Write a plausible news article explaining what went wrong. Use the provided research to ground your narrative. Research:\n{research}""")
        success_prompt = clean_indents(f"""It is one day after this question's resolution date ({question.resolution_date}). The outcome of "{question.question_text}" was a surprising YES. Write a plausible news article explaining what went right. Use the provided research to ground your narrative. Research:\n{research}""")
        narratives = await self._generate_narratives(self.MODEL_CONFIG["narrative_primary"], self.MODEL_CONFIG["narrative_secondary"], fail_prompt, success_prompt)
        synthesis_prompt = clean_indents(f"""Read the two narratives. Extract key insights into a Markdown table with two columns: 'Identified Risks (drives NO)' and 'Identified Opportunities (drives YES)'. Failure Narrative:\n{narratives[0]}\nSuccess Narrative:\n{narratives[1]}""")
        synthesis = await OpenRouterLlm(self.MODEL_CONFIG["analytical_synthesis"]).invoke(synthesis_prompt)
        judgment_prompt = clean_indents(f"""You are a super forecaster. Question: {question.question_text}. Based on the synthesized risks and opportunities below, provide a final probability that the question resolves YES. Weigh the risks against the opportunities in your rationale. Risks & Opportunities:\n{synthesis}\n\nFormat your output ONLY as:\nProbability: XX%\nRationale: <Your detailed reasoning>""")
        prediction = await OpenRouterLlm(self.MODEL_CONFIG["analytical_judgment"]).invoke(judgment_prompt)
        self.print_analysis_results(question, narratives, synthesis, prediction, ("Failure Narrative", "Success Narrative"))

    async def _run_forecast_on_numeric(self, question: MetaculusQuestion, research: str):
        """Runs the pre-mortem for a numeric question."""
        logger.info(f"[{question.id}] Running NUMERIC analysis...")
        low_prompt = clean_indents(f"""It is one day after the resolution date. The final number for "{question.question_text}" was surprisingly LOW, far below the median expectation. Write a plausible news article explaining the constraints and factors that suppressed the outcome. Use the provided research. Research:\n{research}""")
        high_prompt = clean_indents(f"""It is one day after the resolution date. The final number for "{question.question_text}" was surprisingly HIGH, far exceeding the median expectation. Write a plausible news article explaining the catalysts and breakthroughs that drove the outcome. Use the provided research. Research:\n{research}""")
        narratives = await self._generate_narratives(self.MODEL_CONFIG["narrative_primary"], self.MODEL_CONFIG["narrative_secondary"], low_prompt, high_prompt)
        synthesis_prompt = clean_indents(f"""Read the two narratives. Extract key drivers into a Markdown table with two columns: 'Downward Drivers (pushes number lower)' and 'Upward Drivers (pushes number higher)'. Low Outcome Narrative:\n{narratives[0]}\nHigh Outcome Narrative:\n{narratives[1]}""")
        synthesis = await OpenRouterLlm(self.MODEL_CONFIG["analytical_synthesis"]).invoke(synthesis_prompt)
        judgment_prompt = clean_indents(f"""You are a super forecaster. Question: {question.question_text}. Based on the synthesized drivers below, provide a numeric forecast. Weigh the upward vs. downward drivers in your rationale. Downward & Upward Drivers:\n{synthesis}\n\nFormat your output ONLY as:\nPoint Forecast: [Your single best numeric estimate]\nConfidence Interval (90%): [Lower bound] to [Upper bound]\nRationale: <Your detailed reasoning>""")
        prediction = await OpenRouterLlm(self.MODEL_CONFIG["analytical_judgment"]).invoke(judgment_prompt)
        self.print_analysis_results(question, narratives, synthesis, prediction, ("Low Outcome Narrative", "High Outcome Narrative"))

    async def _run_forecast_on_multiple_choice(self, question: MetaculusQuestion, research: str):
        """Placeholder for multiple choice questions, as required by the abstract class."""
        logger.warning(f"Multiple choice question type not implemented. Skipping question: {question.id}")
        pass

    async def _generate_narratives(self, model1, model2, prompt1, prompt2) -> tuple[str, str]:
        """Generates two narratives in parallel."""
        tasks = [OpenRouterLlm(model1).invoke(prompt1), OpenRouterLlm(model2).invoke(prompt2)]
        results = await asyncio.gather(*tasks)
        return results[0], results[1]

    def print_analysis_results(self, question, narratives, synthesis, prediction, narrative_titles):
        """Formats and prints the final analysis."""
        print("\n" + "="*80)
        print(f"✅ ANALYSIS COMPLETE FOR: {question.question_text} (Type: {question.question_type})")
        print(f"   URL: {question.page_url}")
        print("="*80 + "\n")
        print(f"--- 📉 {narrative_titles[0]} ({self.MODEL_CONFIG['narrative_primary']}) ---\n{narratives[0]}\n")
        print(f"--- 📈 {narrative_titles[1]} ({self.MODEL_CONFIG['narrative_secondary']}) ---\n{narratives[1]}\n")
        print(f"--- 📊 Synthesized Drivers/Risks ({self.MODEL_CONFIG['analytical_synthesis']}) ---\n{synthesis}\n")
        print(f"--- 🎯 Final Judgment & Prediction ({self.MODEL_CONFIG['analytical_judgment']}) ---\n{prediction}\n")
        print("="*80 + "\n")

async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run Pre-Mortem Analysis Bot on Metaculus questions.")
    parser.add_argument("--tournament-ids", nargs='+', type=str, default=["minibench", "32813", "metaculus-cup-fall-2025"], help="Space-separated Metaculus tournament IDs.")
    args = parser.parse_args()
    
    if not METACULUS_TOKEN:
        logger.error("METACULUS_TOKEN is not set. The bot cannot fetch questions.")
        return

    bot = PreMortemAnalysisBot()
    # FIX 2: Use the new `run_forecast_bot` function and pass the bot instance to it.
    # This is the new, correct way to start the bot's main loop.
    await run_forecast_bot(bot, tournament_ids=args.tournament_ids)

if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot execution interrupted by user.")


