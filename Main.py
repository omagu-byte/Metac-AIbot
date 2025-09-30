# -*- coding: utf-8 -*-
"""
Pre-Mortem MultiChoice Forecasting Bot (OpenRouter default + GPT-5)
- Default LLM backend: OpenRouter (OPENROUTER_API_KEY required for OpenRouter)
- If a model string starts with "openai/" and OPENAI_API_KEY exists, the bot uses OpenAI for that model
- Research: NewsAPI, SerpAPI, LinkUp, DuckDuckGo scraping
- Forecasts all questions in tournaments by default (skip_previously_forecasted_questions = False)
"""
import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Tuple, Literal, List, Any

import aiohttp
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:
    from forecasting_tools import (
        ForecastBot,
        MetaculusApi,
        MetaculusQuestion,
        MultipleChoiceQuestion,
        NumericQuestion,
        BinaryQuestion,
        clean_indents,
    )
except ImportError:
    print("Error: 'forecasting_tools.py' not found.")
    print("Please ensure the file is in the same directory as this script.")
    exit(1)


# -----------------------
# Environment & Logging
# -----------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PreMortemOpenRouterGPT5Bot")

# -----------------------
# Async LLM clients
# -----------------------
class OpenRouterLlm:
    """Async OpenRouter client (default)."""
    def __init__(self, model: str):
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}

    async def invoke(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=self.headers, json=payload, timeout=240) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        logger.error(f"[OpenRouter {self.model}] status {resp.status}: {text[:500]}")
                        return f"[openrouter:{self.model} error {resp.status}] {text[:500]}"
                    
                    j = await resp.json()
                    content = j.get("choices", [{}])[0].get("message", {}).get("content")
                    if not content:
                        logger.error(f"OpenRouter response missing content: {j}")
                        return f"[openrouter:{self.model} error] Malformed response."
                    return content
        except Exception as e:
            logger.exception(f"OpenRouter invoke error for {self.model}")
            return f"[openrouter:{self.model} exception] {e}"

class OpenAIApiLlm:
    """Async OpenAI REST client. Used only when model string is 'openai/...' and OPENAI_API_KEY exists."""
    def __init__(self, model: str):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.model = model.split("/", 1)[-1]  # Use part after "openai/"
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

    async def invoke(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1500) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=self.headers, json=payload, timeout=240) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        logger.error(f"[OpenAI {self.model}] status {resp.status}: {text[:500]}")
                        return f"[openai:{self.model} error {resp.status}] {text[:500]}"
                    
                    j = await resp.json()
                    content = j.get("choices", [{}])[0].get("message", {}).get("content")
                    if not content:
                        logger.error(f"OpenAI response missing content: {j}")
                        return f"[openai:{self.model} error] Malformed response."
                    return content
        except Exception as e:
            logger.exception(f"OpenAIApiLlm.invoke error for {self.model}")
            return f"[openai:{self.model} exception] {e}"

def llm_factory(model: str):
    """Factory to return an LLM client instance."""
    m_lower = model.lower()
    if m_lower.startswith("openai/") and OPENAI_API_KEY:
        logger.info(f"Using OpenAI client for model: {model}")
        return OpenAIApiLlm(model)
    if OPENROUTER_API_KEY:
        logger.info(f"Using OpenRouter client for model: {model}")
        return OpenRouterLlm(model)
    if OPENAI_API_KEY:
        logger.warning(f"OpenRouter key not found. Falling back to OpenAI for model: {model}")
        return OpenAIApiLlm(model)
    raise RuntimeError("No LLM backend configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

# -----------------------
# Research clients
# -----------------------
class LinkupClient:
    """Simple LinkUp search (async)."""
    async def search(self, query: str) -> str:
        if not LINKUP_API_KEY:
            logger.warning("LINKUP_API_KEY not set.")
            return "LinkUp API key not set."
        headers = {"Authorization": f"Bearer {LINKUP_API_KEY}"}
        params = {"q": query, "limit": 5}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.linkup.com/v1/jobs/search", headers=headers, params=params, timeout=20) as resp:
                    if resp.status != 200:
                        return f"[LinkUp error: {resp.status}] {await resp.text()}"
                    jobs = (await resp.json()).get("jobs", [])
                    return "\n".join([f"- {job.get('title','')} at {job.get('company','')} ({job.get('location','')})" for job in jobs])
        except Exception as e:
            logger.exception("LinkUp search failed")
            return f"LinkUp search failed: {e}"

class SerpApiClient:
    """SerpAPI search (async)."""
    async def search(self, query: str) -> str:
        if not SERPAPI_API_KEY:
            logger.warning("SERPAPI_API_KEY not set.")
            return "SerpAPI key not set."
        params = {"q": query, "api_key": SERPAPI_API_KEY, "num": 5, "engine": "google"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://serpapi.com/search", params=params, timeout=20) as resp:
                    if resp.status != 200:
                        return f"[SerpAPI error: {resp.status}] {await resp.text()}"
                    data = await resp.json()
                    organic = data.get("organic_results", []) or data.get("organic", [])
                    results = []
                    for item in organic[:5]:
                        title = item.get("title") or item.get("position")
                        snippet = item.get("snippet") or item.get("snippet_text") or item.get("snippet_html", "")
                        results.append(f"- {title}: {snippet}")
                    return "\n".join(results)
        except Exception as e:
            logger.exception("SerpAPI search failed")
            return f"SerpAPI search failed: {e}"

class WebScraper:
    def perform_web_scrape(self, query: str) -> str:
        try:
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            top_link = soup.find("a", class_="result__a")
            if not top_link or not top_link.get("href"):
                return "No clear top result found on DuckDuckGo."
            
            page_url = top_link["href"]
            logger.info(f"Scraping top result: {page_url}")
            page_resp = requests.get(page_url, headers=headers, timeout=15)
            page_resp.raise_for_status()
            
            page_soup = BeautifulSoup(page_resp.text, "html.parser")
            paragraphs = page_soup.find_all("p")
            content = " ".join(p.get_text(strip=True) for p in paragraphs[:5])
            
            return content[:2000] if content else "No paragraph content found on the page."
        except requests.exceptions.RequestException as e:
            logger.error(f"WebScraper HTTP error: {e}")
            return f"Web scraping failed due to a network error: {e}"
        except Exception as e:
            logger.exception("WebScraper failed")
            return f"Web scraping failed: {e}"

# -----------------------
# Main Bot
# -----------------------
class PreMortemOpenRouterGPT5Bot(ForecastBot):
    """
    Pre-mortem bot that uses OpenRouter by default and includes multiple research sources.
    """
    MODEL_CONFIG = {
        "narrative_primary": os.getenv("NARRATIVE_PRIMARY_MODEL", "openrouter/gpt-o3"),
        "narrative_secondary": os.getenv("NARRATIVE_SECONDARY_MODEL", "openrouter/gpt-5"),
        "analytical_synthesis": os.getenv("ANALYTICAL_SYNTHESIS_MODEL", "openai/gpt-4o"),
        "analytical_judgment": os.getenv("ANALYTICAL_JUDGMENT_MODEL", "openrouter/gpt-5"),
    }

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Gather research from various sources concurrently."""
        logger.info(f"[{question.id}] Running combined research...")
        loop = asyncio.get_running_loop()

        def query_newsapi_sync():
            if not NEWSAPI_API_KEY:
                logger.warning("NEWSAPI_API_KEY not set.")
                return "NewsAPI key not set."
            try:
                from newsapi import NewsApiClient
                api = NewsApiClient(api_key=NEWSAPI_API_KEY)
                resp = api.get_everything(q=question.question_text, language="en", sort_by="relevancy", page_size=5)
                return "\n".join([f"- {a['title']}: {a.get('description','')}" for a in resp.get("articles", [])])
            except Exception as e:
                logger.exception("NewsAPI failed")
                return f"NewsAPI failed: {e}"

        tasks = {
            "web_scrape": loop.run_in_executor(None, WebScraper().perform_web_scrape, question.question_text),
            "newsapi": loop.run_in_executor(None, query_newsapi_sync),
            "serpapi": SerpApiClient().search(question.question_text),
            "linkup": LinkupClient().search(question.question_text),
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        research = dict(zip(tasks.keys(), results))
        
        return (
            f"--- RESEARCH SUMMARY ---\n\n"
            f"NewsAPI:\n{research.get('newsapi', 'N/A')}\n\n"
            f"SerpAPI:\n{research.get('serpapi', 'N/A')}\n\n"
            f"LinkUp:\n{research.get('linkup', 'N/A')}\n\n"
            f"Web Scrape:\n{research.get('web_scrape', 'N/A')}\n"
            f"--- END RESEARCH ---\n"
        )

    async def _generate_narratives(self, model1: str, model2: str, prompt1: str, prompt2: str) -> Tuple[str, str]:
        """Generate two narratives in parallel."""
        logger.info(f"Generating narratives using {model1} and {model2}")
        llm1 = llm_factory(model1)
        llm2 = llm_factory(model2)
        r1_task = llm1.invoke(prompt1, temperature=0.7, max_tokens=1200)
        r2_task = llm2.invoke(prompt2, temperature=0.7, max_tokens=1200)
        r1, r2 = await asyncio.gather(r1_task, r2_task)
        return r1, r2
    
    def _extract_json(self, text: str) -> Any:
        """Extracts a JSON object or array from a string."""
        # Find JSON enclosed in ```json ... ``` or the first '{' or '['
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1)
        
        # Find the first occurrence of '{' or '['
        first_bracket = -1
        first_curly = text.find('{')
        first_square = text.find('[')

        if first_curly != -1 and first_square != -1:
            first_bracket = min(first_curly, first_square)
        elif first_curly != -1:
            first_bracket = first_curly
        else:
            first_bracket = first_square
        
        if first_bracket == -1:
            return None # No JSON object/array found

        # Find the matching last bracket/curly
        text = text[first_bracket:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to be more lenient, find matching brackets
            open_braces = 0
            open_squares = 0
            last_index = -1
            
            for i, char in enumerate(text):
                if char == '{': open_braces += 1
                elif char == '}': open_braces -= 1
                elif char == '[': open_squares += 1
                elif char == ']': open_squares -= 1

                if open_braces == 0 and open_squares == 0:
                    last_index = i
                    break
            
            if last_index != -1:
                try:
                    return json.loads(text[:last_index+1])
                except json.JSONDecodeError:
                    return None
        return None


    async def _run_forecast_on_binary(self, question: MetaculusQuestion, research: str):
        logger.info(f"[{question.id}] Binary pre-mortem analysis")
        fail_prompt = clean_indents(f"""One day after the resolution date ({question.resolution_date}), the outcome of the question "{question.question_text}" was a surprising NO. Write a plausible news article explaining what went wrong. Use the research below.\nResearch:\n{research}""")
        success_prompt = clean_indents(f"""One day after the resolution date ({question.resolution_date}), the outcome of the question "{question.question_text}" was a surprising YES. Write a plausible news article explaining what went right. Use the research below.\nResearch:\n{research}""")

        narratives = await self._generate_narratives(self.MODEL_CONFIG["narrative_primary"], self.MODEL_CONFIG["narrative_secondary"], fail_prompt, success_prompt)
        synthesis_prompt = clean_indents(f"""Read the two narratives. Extract key insights into a Markdown table with columns: 'Identified Risks (drives NO)' and 'Identified Opportunities (drives YES)'.\n\nFailure Narrative:\n{narratives[0]}\n\nSuccess Narrative:\n{narratives[1]}""")
        synthesis = await llm_factory(self.MODEL_CONFIG["analytical_synthesis"]).invoke(synthesis_prompt, temperature=0.1, max_tokens=1200)
        
        judgment_prompt = clean_indents(f"""You are a superforecaster. Question: {question.question_text}\nBased on the synthesized risks and opportunities below, provide a final probability that the question resolves YES.\nFormat your response as a JSON object with two keys: "probability_percent" (a number from 0-100) and "rationale" (a string with your detailed reasoning).\nExample: {{"probability_percent": 75, "rationale": "The opportunities outweigh the risks because..."}}\n\nRisks & Opportunities:\n{synthesis}""")
        judgment_text = await llm_factory(self.MODEL_CONFIG["analytical_judgment"]).invoke(judgment_prompt, temperature=0.0, max_tokens=800)
        
        judgment_json = self._extract_json(judgment_text)
        self._print_analysis_results(question, narratives, synthesis, json.dumps(judgment_json, indent=2) if judgment_json else judgment_text, ("Failure Narrative", "Success Narrative"))

    async def _run_forecast_on_numeric(self, question: MetaculusQuestion, research: str):
        logger.info(f"[{question.id}] Numeric pre-mortem analysis")
        low_prompt = clean_indents(f"""One day after the resolution date ({question.resolution_date}), the final number for "{question.question_text}" was surprisingly LOW. Write a plausible news article explaining factors that suppressed the outcome. Use the research below.\nResearch:\n{research}""")
        high_prompt = clean_indents(f"""One day after the resolution date ({question.resolution_date}), the final number for "{question.question_text}" was surprisingly HIGH. Write a plausible news article explaining catalysts that drove the outcome. Use the research below.\nResearch:\n{research}""")

        narratives = await self._generate_narratives(self.MODEL_CONFIG["narrative_primary"], self.MODEL_CONFIG["narrative_secondary"], low_prompt, high_prompt)
        synthesis_prompt = clean_indents(f"""Read the low and high narratives. Extract key drivers into a Markdown table with columns: 'Downward Drivers' and 'Upward Drivers'.\n\nLow Narrative:\n{narratives[0]}\n\nHigh Narrative:\n{narratives[1]}""")
        synthesis = await llm_factory(self.MODEL_CONFIG["analytical_synthesis"]).invoke(synthesis_prompt, temperature=0.1, max_tokens=1200)

        judgment_prompt = clean_indents(f"""You are a superforecaster. Question: {question.question_text}\nBased on the synthesized drivers below, provide a numeric forecast.\nFormat your response as a JSON object with four keys: "point_forecast", "ci90_lower", "ci90_upper", and "rationale".\nExample: {{"point_forecast": 500, "ci90_lower": 200, "ci90_upper": 800, "rationale": "The upward drivers seem more potent..."}}\n\nDrivers:\n{synthesis}""")
        judgment_text = await llm_factory(self.MODEL_CONFIG["analytical_judgment"]).invoke(judgment_prompt, temperature=0.0, max_tokens=800)

        judgment_json = self._extract_json(judgment_text)
        self._print_analysis_results(question, narratives, synthesis, json.dumps(judgment_json, indent=2) if judgment_json else judgment_text, ("Low Narrative", "High Narrative"))

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str):
        logger.info(f"[{question.id}] Multiple choice analysis")
        
        # More robustly extract options, checking for different attribute names and types.
        options = getattr(question, "options", None) or getattr(question, "choices", None)
        
        if isinstance(options, dict):
            # Handle dictionary format, e.g., {0: "Option A", 1: "Option B"}
            option_lines = "\n".join([f"{i}: {v}" for i, v in options.items()])
        elif isinstance(options, (list, tuple)):
            # Handle list/tuple format, e.g., ["Option A", "Option B"]
            option_lines = "\n".join([f"{i}: {opt}" for i, opt in enumerate(options)])
        else:
            logger.warning(f"[{question.id}] Could not extract structured options. Will proceed without them.")
            option_lines = "Options not available in structured format. Analyze based on question text."
        
        prompt = clean_indents(f"""You are a forecasting assistant. Here is the question and the options:
Question: {question.question_text}
Options:\n{option_lines}
Research:\n{research}

Return a JSON array where each object has "index", "probability", and "rationale". Probabilities must sum to 100.
Example: [{{"index": 0, "probability": 20, "rationale":"..."}}, {{"index": 1, "probability": 80, "rationale":"..."}}]""")
        result_text = await llm_factory(self.MODEL_CONFIG["analytical_judgment"]).invoke(prompt, temperature=0.2, max_tokens=1500)

        json_result = self._extract_json(result_text)
        pretty_result = json.dumps(json_result, indent=2) if json_result else result_text

        self._print_analysis_results(question, [], research, pretty_result, ("Multiple Choice Analysis", ""))

    def _print_analysis_results(self, question: MetaculusQuestion, narratives: List[str], synthesis: str, prediction: str, titles: Tuple[str, str]):
        print("\n" + "=" * 90)
        print(f"✅ ANALYSIS FOR: {question.question_text} (Type: {getattr(question, 'question_type', 'unknown')})")
        print(f"   URL: {getattr(question, 'page_url', 'n/a')}")
        print("=" * 90 + "\n")
        if narratives:
            print(f"--- {titles[0]} ({self.MODEL_CONFIG['narrative_primary']}) ---\n{narratives[0]}\n")
            print(f"--- {titles[1]} ({self.MODEL_CONFIG['narrative_secondary']}) ---\n{narratives[1]}\n")
            print(f"--- Synthesized Drivers/Risks ({self.MODEL_CONFIG['analytical_synthesis']}) ---\n{synthesis}\n")
        else:
            # For multiple choice, synthesis contains research
            print(f"--- Research ---\n{synthesis}\n")
        print(f"--- FINAL JUDGMENT ({self.MODEL_CONFIG['analytical_judgment']}) ---\n{prediction}\n")
        print("=" * 90 + "\n")

# -----------------------
# Entrypoint
# -----------------------
async def main():
    parser = argparse.ArgumentParser(
        description="Run PreMortem OpenRouter+GPT Bot on Metaculus tournaments."
    )
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        default=[str(32813), str(MetaculusApi.CURRENT_MINIBENCH_ID)],
        help=(
            "Metaculus tournament IDs "
            f"(default: 32813 and current minibench {MetaculusApi.CURRENT_MINIBENCH_ID})"
        ),
    )
    parser.add_argument(
        "--models",
        nargs=4,
        metavar=("NARR_PRIMARY", "NARR_SECONDARY", "SYNTHESIS", "JUDGMENT"),
        help="Override models for the four stages.",
    )
    args = parser.parse_args()


    if args.models:
        PreMortemOpenRouterGPT5Bot.MODEL_CONFIG.update({
            "narrative_primary": args.models[0],
            "narrative_secondary": args.models[1],
            "analytical_synthesis": args.models[2],
            "analytical_judgment": args.models[3],
        })
        logger.info(f"Overrode MODEL_CONFIG via CLI: {PreMortemOpenRouterGPT5Bot.MODEL_CONFIG}")

    if not METACULUS_TOKEN:
        logger.error("METACULUS_TOKEN not set. The bot cannot fetch tournament questions.")
        return

    bot = PreMortemOpenRouterGPT5Bot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
    )

    # --- FIX ---
    # Loop through each tournament ID and call the singular method.
    for tournament_id in args.tournament_ids:
        logger.info(f"--- Starting forecast for tournament ID: {tournament_id} ---")
        try:
            await bot.forecast_on_tournament(tournament_id=int(tournament_id), return_exceptions=True)
        except ValueError:
            logger.error(f"Invalid tournament ID: '{tournament_id}'. Must be an integer.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing tournament {tournament_id}: {e}")
    # --- END FIX ---


if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
