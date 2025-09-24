#Pre-Mortem Analysis Bot using forecasting-tools framework and custom research.
Multi-model, multi-source research for scenario analysis, risk/opportunity synthesis, and final judgment.#

Models:
- Narrative: Anthropic Claude 3 Opus, Mistral Large (via OpenRouter)
- Analytical: OpenAI GPT-4o (via OpenRouter), OpenAI GPT-O3 (via OpenRouter)
News/Research:
- NewsAPI, AskNews, SerpAPI (parallel fallback)
- Web scraping (DuckDuckGo + BeautifulSoup)
Metaculus API for question sourcing.

Environment:
- OPENROUTER_API_KEY, METACULUS_TOKEN, NEWSAPI_API_KEY, ASKNEWS_API_KEY, SERPAPI_API_KEY

import argparse
import asyncio
import logging
import os
from datetime import datetime

import numpy as np
import requests
from bs4 import BeautifulSoup
from forecasting_tools import (
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    clean_indents,
)
from newsapi import NewsApiClient

# API KEYS
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
ASKNEWS_API_KEY = os.getenv("ASKNEWS_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PreMortemAnalysisBot")

# --- LLM Clients (OpenRouter wrapper) ---

class OpenRouterLlm:
    def __init__(self, model: str):
        self.model = model
        self.api_key = OPENROUTER_API_KEY
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def invoke(self, prompt: str, temperature=0.7, max_tokens=1024):
        import aiohttp
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, headers=headers, json=data) as resp:
                if resp.status != 200:
                    return f"[{self.model} error: {resp.status}]"
                result = await resp.json()
                try:
                    return result['choices'][0]['message']['content']
                except Exception:
                    return str(result)

# --- Research Clients ---

class AskNewsClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.asknews.io/search"

    async def search(self, query: str) -> str:
        if not self.api_key:
            return "AskNewsAPI key not set."
        params = {
            "q": query,
            "api_key": self.api_key,
            "limit": 5
        }
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url, params=params) as resp:
                if resp.status != 200:
                    return f"[AskNews error: {resp.status}]"
                result = await resp.json()
                items = result.get("results", [])
                return "\n".join([f"- {i.get('title','')}: {i.get('snippet','')}" for i in items])

class SerpApiClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://serpapi.com/search"

    async def search(self, query: str) -> str:
        if not self.api_key:
            return "SerpAPI key not set."
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": 5,
            "engine": "google"
        }
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url, params=params) as resp:
                if resp.status != 200:
                    return f"[SerpAPI error: {resp.status}]"
                result = await resp.json()
                items = result.get("organic_results", [])
                return "\n".join([f"- {i.get('title','')}: {i.get('snippet','')}" for i in items])

class WebScraper:
    def perform_web_scrape(self, query: str) -> str:
        logger.info(f"[Web Scraper] Searching for: {query}")
        try:
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            top_result = soup.find('a', class_='result__a')
            if not top_result or 'href' not in top_result.attrs:
                return "Web scraping found no clear top result."
            page_url = top_result['href']
            logger.info(f"[Web Scraper] Scraping content from: {page_url}")
            page_response = requests.get(page_url, headers=headers, timeout=10)
            page_response.raise_for_status()
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            paragraphs = page_soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs[:5]])
            return f"- {content[:1500]}..."
        except Exception as e:
            return f"Web scraping failed: {e}"

# --- Hybrid Research Collector ---

async def gather_research(query: str):
    """
    Runs NewsAPI, AskNews, SerpAPI, and web scraping in parallel, returns all results.
    """
    results = {}
    loop = asyncio.get_running_loop()
    tasks = {
        "newsapi": NewsApiClient(api_key=NEWSAPI_API_KEY).get_everything(q=query, language="en", sort_by="relevancy", page_size=5),
        "web_scrape": loop.run_in_executor(None, WebScraper().perform_web_scrape, query),
        "asknews": AskNewsClient(ASKNEWS_API_KEY).search(query),
        "serpapi": SerpApiClient(SERPAPI_API_KEY).search(query),
    }
    # NewsAPI is sync, others async
    futures = [
        asyncio.create_task(tasks["asknews"]),
        asyncio.create_task(tasks["serpapi"]),
        tasks["newsapi"],  # sync
        tasks["web_scrape"],  # sync
    ]
    results_list = await asyncio.gather(*futures, return_exceptions=True)
    results['asknews'], results['serpapi'], newsapi_result, webscrape_result = results_list
    # Format NewsAPI
    try:
        if isinstance(newsapi_result, dict) and newsapi_result.get("articles"):
            articles = newsapi_result.get("articles", [])
            results['newsapi'] = "\n".join([f"- {a['title']}: {a.get('description', '')}" for a in articles])
        else:
            results['newsapi'] = str(newsapi_result)
    except Exception:
        results['newsapi'] = str(newsapi_result)
    results['web_scrape'] = webscrape_result
    return results

# --- MAIN BOT ---

class PreMortemAnalysisBot(ForecastBot):
    """
    Pre-Mortem Analysis forecasting bot.
    Uses narrative models for scenario generation,
    analytical models for synthesis and judgment.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        defaults = super()._llm_config_defaults()
        defaults.update({
            "narrative_1": "anthropic/claude-3-opus-20240229",
            "narrative_2": "mistral/mistral-large",
            "analytical_1": "openai/gpt-4o",
            "analytical_2": "openai/gpt-o3",  # <-- OpenAI GPT-O3 as analytic 2
        })
        return defaults

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forecaster_keys = ["narrative_1", "narrative_2", "analytical_1", "analytical_2"]

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Collects research from all sources for context.
        """
        async with self._concurrency_limiter:
            logger.info(f"--- Running Research for: {question.question_text} ---")
            research_results = await gather_research(question.question_text)
            research = (
                f"NewsAPI:\n{research_results.get('newsapi','')}\n\n"
                f"Web Scrape:\n{research_results.get('web_scrape','')}\n\n"
                f"AskNews:\n{research_results.get('asknews','')}\n\n"
                f"SerpAPI:\n{research_results.get('serpapi','')}\n\n"
            )
            logger.info(f"--- Research Complete for {question.page_url} ---")
            return research

    async def run_pre_mortem_analysis(self, question: MetaculusQuestion, research: str) -> dict:
        """
        Runs Pre-Mortem scenario generation, synthesis, and judgment.
        """
        # Step 1: Failure Narrative (Claude Opus)
        fail_prompt = clean_indents(
            f"""
            It is one day after {question.resolution_date}.
            The outcome of "{question.question_text}" was a surprising NO.
            Write a plausible, detailed news article or after-action report from the future explaining exactly what went wrong and the sequence of events that led to this failure.
            Research:\n{research}
            """
        )
        fail_narrative = await OpenRouterLlm("anthropic/claude-3-opus-20240229").invoke(fail_prompt)

        # Step 2: Success Narrative (Mistral Large)
        success_prompt = clean_indents(
            f"""
            It is one day after {question.resolution_date}.
            The outcome of "{question.question_text}" was a surprising YES.
            Write a plausible, detailed news article from the future explaining the key decisions, breakthroughs, and overlooked factors that led to this victory.
            Research:\n{research}
            """
        )
        success_narrative = await OpenRouterLlm("mistral/mistral-large").invoke(success_prompt)

        # Step 3: Synthesis (GPT-4o)
        synthesis_prompt = clean_indents(
            f"""
            Read the following two future histories (failure and success narratives):
            Failure Narrative:\n{fail_narrative}\n
            Success Narrative:\n{success_narrative}\n
            Extract a structured list of key insights. Populate two columns:
            - Identified Risks (from the failure story)
            - Identified Opportunities (from the success story)
            Format as a Markdown table with Risks and Opportunities columns.
            """
        )
        risks_opps_markdown = await OpenRouterLlm("openai/gpt-4o").invoke(synthesis_prompt)

        # Step 4: Final Judgment (GPT-O3)
        judgment_prompt = clean_indents(
            f"""
            You are the final super forecaster. Given the research and this specific list of risks and opportunities, what is your final probability that the question resolves YES?
            Your rationale must explicitly address how these risks and opportunities influenced your decision.
            Question: {question.question_text}
            Resolution Date: {question.resolution_date}
            Background: {question.background_info}
            Fine Print: {question.fine_print}
            Risks & Opportunities:\n{risks_opps_markdown}
            Format: Probability: XX% Rationale: <detailed rationale>
            """
        )
        final_prediction = await OpenRouterLlm("openai/gpt-o3").invoke(judgment_prompt)

        return {
            "failure_narrative": fail_narrative,
            "success_narrative": success_narrative,
            "risks_opps_markdown": risks_opps_markdown,
            "final_prediction": final_prediction,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pre-Mortem Analysis Bot on Metaculus questions.")
    parser.add_argument(
        "--mode", type=str, choices=["tournament", "test_questions"], default="tournament",
        help="Run mode: tournament or test_questions."
    )
    parser.add_argument(
        "--tournament-ids", nargs='+', type=str, default=[MetaculusApi.CURRENT_AI_COMPETITION_ID],
        help="Tournament IDs to run on."
    )
    args = parser.parse_args()

    bot = PreMortemAnalysisBot(
        llms={
            "narrative_1": GeneralLlm(model="anthropic/claude-3-opus-20240229"),
            "narrative_2": GeneralLlm(model="mistral/mistral-large"),
            "analytical_1": GeneralLlm(model="openai/gpt-4o"),
            "analytical_2": GeneralLlm(model="openai/gpt-o3"),
            "parser": GeneralLlm(model="openai/gpt-o3")
        }
    )

    if args.mode == "tournament":
        logger.info("Running in tournament mode...")
        tournament_ids = args.tournament_ids
        all_questions = []
        for tournament_id in tournament_ids:
            questions = MetaculusApi.get_questions_for_tournament(tournament_id)
            all_questions.extend(questions)
        for question in all_questions:
            research = asyncio.run(bot.run_research(question))
            result = asyncio.run(bot.run_pre_mortem_analysis(question, research))
            print("==== Pre-Mortem Analysis for Question ====")
            print(f"Q: {question.question_text}\nURL: {question.page_url}")
            print("=== Failure Narrative ===\n", result["failure_narrative"], "\n")
            print("=== Success Narrative ===\n", result["success_narrative"], "\n")
            print("=== Risks & Opportunities ===\n", result["risks_opps_markdown"], "\n")
            print("=== Final Judgment ===\n", result["final_prediction"], "\n")
    elif args.mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
        ]
        for url in EXAMPLE_QUESTIONS:
            question = MetaculusApi.get_question_by_url(url)
            research = asyncio.run(bot.run_research(question))
            result = asyncio.run(bot.run_pre_mortem_analysis(question, research))
            print("==== Pre-Mortem Analysis for Question ====")
            print(f"Q: {question.question_text}\nURL: {question.page_url}")
            print("=== Failure Narrative ===\n", result["failure_narrative"], "\n")
            print("=== Success Narrative ===\n", result["success_narrative"], "\n")
            print("=== Risks & Opportunities ===\n", result["risks_opps_markdown"], "\n")
            print("=== Final Judgment ===\n", result["final_prediction"], "\n")
