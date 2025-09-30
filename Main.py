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

from forecasting_tools import (
    ForecastBot,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    BinaryQuestion,
    clean_indents,
)

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
                    return j["choices"][0]["message"]["content"]
        except Exception as e:
            logger.exception(f"OpenRouter invoke error for {self.model}")
            return f"[openrouter:{self.model} exception] {e}"

class OpenAIApiLlm:
    """Async OpenAI REST client. Used only when model string is 'openai/...' and OPENAI_API_KEY exists."""
    def __init__(self, model: str):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        # model here is the part after "openai/" (e.g., "gpt-5")
        self.model = model
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
                    return j["choices"][0]["message"]["content"]
        except Exception as e:
            logger.exception(f"OpenAIApiLlm.invoke error for {self.model}")
            return f"[openai:{self.model} exception] {e}"

def llm_factory(model: str):
    """
    Return an object with async .invoke(prompt, temperature, max_tokens).
    Policy:
      - If model starts with 'openai/' and OPENAI_API_KEY exists -> OpenAIApiLlm(model_after_prefix)
      - Else: prefer OpenRouterLlm (if OPENROUTER_API_KEY exists)
      - Else: fallback to OpenAIApiLlm if OPENAI_API_KEY exists
    """
    m = model.lower()
    # explicit OpenAI model requested AND key exists -> use OpenAI
    if m.startswith("openai/") and OPENAI_API_KEY:
        return OpenAIApiLlm(model.split("/", 1)[1])
    # default preference: OpenRouter if available
    if OPENROUTER_API_KEY:
        return OpenRouterLlm(model)
    # fallback: OpenAI if available
    if OPENAI_API_KEY:
        # model might not include "openai/" prefix; use as-is
        to_use = model.split("/", 1)[1] if m.startswith("openai/") else model
        return OpenAIApiLlm(to_use)
    raise RuntimeError("No LLM backend configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

# -----------------------
# Research clients
# -----------------------
class LinkupClient:
    """Simple LinkUp search (async)."""
    async def search(self, query: str) -> str:
        if not LINKUP_API_KEY:
            return "LinkUp API key not set."
        headers = {"Authorization": f"Bearer {LINKUP_API_KEY}"}
        params = {"q": query, "limit": 5}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.linkup.com/v1/jobs/search", headers=headers, params=params, timeout=20) as resp:
                    if resp.status != 200:
                        return f"[LinkUp error: {resp.status}]"
                    jobs = (await resp.json()).get("jobs", [])
                    return "\n".join([f"- {job.get('title','')} at {job.get('company','')} ({job.get('location','')})" for job in jobs])
        except Exception as e:
            logger.exception("LinkUp search failed")
            return f"LinkUp search failed: {e}"

class SerpApiClient:
    """SerpAPI search (async)."""
    async def search(self, query: str) -> str:
        if not SERPAPI_API_KEY:
            return "SerpAPI key not set."
        params = {"q": query, "api_key": SERPAPI_API_KEY, "num": 5, "engine": "google"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://serpapi.com/search", params=params, timeout=20) as resp:
                    if resp.status != 200:
                        return f"[SerpAPI error: {resp.status}]"
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
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            top = soup.find("a", class_="result__a")
            if not top or not top.get("href"):
                return "No clear top result."
            page_url = top["href"]
            page = requests.get(page_url, headers=headers, timeout=10)
            page.raise_for_status()
            page_soup = BeautifulSoup(page.text, "html.parser")
            paragraphs = page_soup.find_all("p")
            content = " ".join(p.get_text() for p in paragraphs[:5])
            return content[:1500] if content else "No content found."
        except Exception as e:
            logger.exception("WebScraper failed")
            return f"Web scraping failed: {e}"

# -----------------------
# Main Bot
# -----------------------
class PreMortemOpenRouterGPT5Bot(ForecastBot):
    """
    Pre-mortem bot that:
      - uses OpenRouter by default
      - uses GPT-5 (OpenAI) for final judgment when available
      - includes NewsAPI, SerpAPI, LinkUp research
      - supports Binary, Numeric, MultipleChoice
    """
    # default model config: narratives from OpenRouter 'gpt-o3', final judgment uses openai/gpt-5
    MODEL_CONFIG = {
        "narrative_primary": os.getenv("NARRATIVE_PRIMARY_MODEL", "openrouter/gpt-o3"),
        "narrative_secondary": os.getenv("NARRATIVE_SECONDARY_MODEL", "openrouter/gpt-o3"),
        # synthesis can be openrouter too
        "analytical_synthesis": os.getenv("ANALYTICAL_SYNTHESIS_MODEL", "openrouter/gpt-o3"),
        # explicit request to use GPT-5 for final judgment (if OPENAI_API_KEY present)
        "analytical_judgment": os.getenv("ANALYTICAL_JUDGMENT_MODEL", "openai/gpt-5"),
    }

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Gather research from NewsAPI, SerpAPI, LinkUp and web scraping concurrently."""
        logger.info(f"[{question.id}] Running combined research...")
        loop = asyncio.get_running_loop()

        def query_newsapi_sync():
            if not NEWSAPI_API_KEY:
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
            f"--- RESEARCH SUMMARY ---\n"
            f"NewsAPI:\n{research.get('newsapi','')}\n\n"
            f"SerpAPI:\n{research.get('serpapi','')}\n\n"
            f"LinkUp:\n{research.get('linkup','')}\n\n"
            f"Web Scrape:\n{research.get('web_scrape','')}\n"
            f"--- END RESEARCH ---\n"
        )

    async def _generate_narratives(self, model1: str, model2: str, prompt1: str, prompt2: str) -> Tuple[str, str]:
        """Generate two narratives in parallel with given models."""
        logger.info(f"Generating narratives using {model1} and {model2}")
        llm1 = llm_factory(model1)
        llm2 = llm_factory(model2)
        r1_task = llm1.invoke(prompt1, temperature=0.7, max_tokens=1200)
        r2_task = llm2.invoke(prompt2, temperature=0.7, max_tokens=1200)
        r1, r2 = await asyncio.gather(r1_task, r2_task)
        return r1, r2

    async def _run_forecast_on_binary(self, question: MetaculusQuestion, research: str):
        logger.info(f"[{question.id}] Binary pre-mortem analysis")
        fail_prompt = clean_indents(f"""One day after the resolution ({question.resolution_date}), the outcome of:
"{question.question_text}"
was a surprising NO. Write a plausible news article explaining what went wrong. Use the research below.\nResearch:\n{research}""")
        success_prompt = clean_indents(f"""One day after the resolution ({question.resolution_date}), the outcome of:
"{question.question_text}"
was a surprising YES. Write a plausible news article explaining what went right. Use the research below.\nResearch:\n{research}""")

        narratives = await self._generate_narratives(
            self.MODEL_CONFIG["narrative_primary"],
            self.MODEL_CONFIG["narrative_secondary"],
            fail_prompt,
            success_prompt,
        )

        synthesis_prompt = clean_indents(f"""Read the two narratives and extract key insights into a Markdown table with columns:
'Identified Risks (drives NO)' and 'Identified Opportunities (drives YES)'.

Failure Narrative:
{narratives[0]}

Success Narrative:
{narratives[1]}
""")
        synthesis = await llm_factory(self.MODEL_CONFIG["analytical_synthesis"]).invoke(
            synthesis_prompt, temperature=0.1, max_tokens=1200
        )

        judgment_prompt = clean_indents(f"""You are a superforecaster. Question: {question.question_text}
Based on the synthesized risks and opportunities below, provide a final probability that the question resolves YES.
Format EXACTLY as:
{{"probability_percent": XX, "rationale": "<detailed reasoning>" }}

Risks & Opportunities:
{synthesis}
""")
        judgment = await llm_factory(self.MODEL_CONFIG["analytical_judgment"]).invoke(
            judgment_prompt, temperature=0.0, max_tokens=800
        )

        self._print_analysis_results(question, narratives, synthesis, judgment, ("Failure Narrative", "Success Narrative"))

    async def _run_forecast_on_numeric(self, question: MetaculusQuestion, research: str):
        logger.info(f"[{question.id}] Numeric pre-mortem analysis")
        low_prompt = clean_indents(f"""One day after the resolution ({question.resolution_date}), the final number for:
"{question.question_text}"
was surprisingly LOW. Write a plausible news article explaining factors that suppressed the outcome. Use research below.\nResearch:\n{research}""")
        high_prompt = clean_indents(f"""One day after the resolution ({question.resolution_date}), the final number for:
"{question.question_text}"
was surprisingly HIGH. Write a plausible news article explaining catalysts that drove the outcome. Use research below.\nResearch:\n{research}""")

        narratives = await self._generate_narratives(
            self.MODEL_CONFIG["narrative_primary"],
            self.MODEL_CONFIG["narrative_secondary"],
            low_prompt,
            high_prompt,
        )

        synthesis_prompt = clean_indents(f"""Read the low and high narratives and extract key drivers into a Markdown table with columns: 'Downward Drivers' and 'Upward Drivers'.

Low Narrative:
{narratives[0]}

High Narrative:
{narratives[1]}
""")
        synthesis = await llm_factory(self.MODEL_CONFIG["analytical_synthesis"]).invoke(
            synthesis_prompt, temperature=0.1, max_tokens=1200
        )

        judgment_prompt = clean_indents(f"""You are a superforecaster. Question: {question.question_text}
Based on the synthesized drivers below, provide a numeric forecast.
Format EXACTLY as:
{{"point_forecast": <number>, "ci90_lower": <number>, "ci90_upper": <number>, "rationale": "<text>"}}

Drivers:
{synthesis}
""")
        judgment = await llm_factory(self.MODEL_CONFIG["analytical_judgment"]).invoke(
            judgment_prompt, temperature=0.0, max_tokens=800
        )

        self._print_analysis_results(question, narratives, synthesis, judgment, ("Low Narrative", "High Narrative"))

    async def _run_forecast_on_multiple_choice(self, question: MetaculusQuestion, research: str):
        """
        For Multiple Choice questions:
          - ask judgment model for a JSON array of objects:
              [{"index": 0, "probability": 30, "rationale":"..."}, ...]
          - Try to parse JSON; if parse fails, print raw output.
        """
        logger.info(f"[{question.id}] Multiple choice analysis")
        # attempt to get options from the question object
        options = getattr(question, "options", None) or getattr(question, "choices", None) or None
        # serialize options into numbered list
        if isinstance(options, dict):
            # some APIs return {index: label}
            option_lines = "\n".join([f"{i}: {v}" for i, v in options.items()])
        elif isinstance(options, (list, tuple)):
            option_lines = "\n".join([f"{i}: {opt}" for i, opt in enumerate(options)])
        else:
            # fallback: attempt to derive from question text (not ideal)
            option_lines = "Could not extract structured options; include question_text."
        prompt = clean_indents(f"""You are a forecasting assistant. Here is the question and the options:

Question:
{question.question_text}

Options:
{option_lines}

Research:
{research}

Return a JSON array where each element is:
{{"index": <option_index>, "probability": <0-100 numeric>, "rationale": "<brief justification>"}}
Ensure probabilities sum to ~100. Example:
[{{"index": 0, "probability": 20, "rationale":"..."}}, ...]
""")
        result_text = await llm_factory(self.MODEL_CONFIG["analytical_judgment"]).invoke(prompt, temperature=0.2, max_tokens=1200)

        # Try to extract JSON from LLM reply
        json_text = None
        try:
            # Look for a JSON array block in the response
            m = re.search(r"(\[.*\])", result_text, re.DOTALL)
            candidate = m.group(1) if m else result_text.strip()
            parsed = json.loads(candidate)
            # Validate structure minimally
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                json_text = parsed
        except Exception:
            json_text = None

        if json_text is None:
            # Could not parse JSON reliably; print raw output
            self._print_analysis_results(question, [], research, result_text, ("Multiple Choice (raw output)", ""))
            return

        # Print nicely
        pretty = json.dumps(json_text, indent=2)
        self._print_analysis_results(question, [], research, pretty, ("Multiple Choice (parsed)", ""))

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
            print(f"--- Research ---\n{synthesis}\n")
        print(f"--- FINAL JUDGMENT ({self.MODEL_CONFIG['analytical_judgment']}) ---\n{prediction}\n")
        print("=" * 90 + "\n")

# -----------------------
# Entrypoint
# -----------------------
async def main():
    parser = argparse.ArgumentParser(description="Run PreMortem OpenRouter+GPT5 Bot on Metaculus tournaments.")
    parser.add_argument("--tournament-ids", nargs="+", default=[str(32813), MetaculusApi.CURRENT_MINIBENCH_ID],
                        help="Metaculus tournament IDs (default: 32813 and minibench)")
    parser.add_argument("--models", nargs="+", help="Optional override models (primary secondary synthesis judgment)")
    args = parser.parse_args()

    if args.models and len(args.models) == 4:
        PreMortemOpenRouterGPT5Bot.MODEL_CONFIG = {
            "narrative_primary": args.models[0],
            "narrative_secondary": args.models[1],
            "analytical_synthesis": args.models[2],
            "analytical_judgment": args.models[3],
        }
        logger.info(f"Overrode MODEL_CONFIG via CLI: {PreMortemOpenRouterGPT5Bot.MODEL_CONFIG}")

    if not METACULUS_TOKEN:
        logger.error("METACULUS_TOKEN not set. The bot cannot fetch tournament questions.")
        return

    # instantiate bot with skip_previously_forecasted_questions = False to forecast ALL questions
    bot = PreMortemOpenRouterGPT5Bot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,  # change to True if you want to publish
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,  # IMPORTANT: forecast all questions in tournament
    )

    # run tournaments
    await bot.forecast_on_tournaments(tournament_ids=args.tournament_ids, return_exceptions=True)

if __name__ == "__main__":
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except Exception:
        pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
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

from forecasting_tools import (
    ForecastBot,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    BinaryQuestion,
    clean_indents,
)

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
                    return j["choices"][0]["message"]["content"]
        except Exception as e:
            logger.exception(f"OpenRouter invoke error for {self.model}")
            return f"[openrouter:{self.model} exception] {e}"

class OpenAIApiLlm:
    """Async OpenAI REST client. Used only when model string is 'openai/...' and OPENAI_API_KEY exists."""
    def __init__(self, model: str):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        # model here is the part after "openai/" (e.g., "gpt-5")
        self.model = model
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
                    return j["choices"][0]["message"]["content"]
        except Exception as e:
            logger.exception(f"OpenAIApiLlm.invoke error for {self.model}")
            return f"[openai:{self.model} exception] {e}"

def llm_factory(model: str):
    """
    Return an object with async .invoke(prompt, temperature, max_tokens).
    Policy:
      - If model starts with 'openai/' and OPENAI_API_KEY exists -> OpenAIApiLlm(model_after_prefix)
      - Else: prefer OpenRouterLlm (if OPENROUTER_API_KEY exists)
      - Else: fallback to OpenAIApiLlm if OPENAI_API_KEY exists
    """
    m = model.lower()
    # explicit OpenAI model requested AND key exists -> use OpenAI
    if m.startswith("openai/") and OPENAI_API_KEY:
        return OpenAIApiLlm(model.split("/", 1)[1])
    # default preference: OpenRouter if available
    if OPENROUTER_API_KEY:
        return OpenRouterLlm(model)
    # fallback: OpenAI if available
    if OPENAI_API_KEY:
        # model might not include "openai/" prefix; use as-is
        to_use = model.split("/", 1)[1] if m.startswith("openai/") else model
        return OpenAIApiLlm(to_use)
    raise RuntimeError("No LLM backend configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")

# -----------------------
# Research clients
# -----------------------
class LinkupClient:
    """Simple LinkUp search (async)."""
    async def search(self, query: str) -> str:
        if not LINKUP_API_KEY:
            return "LinkUp API key not set."
        headers = {"Authorization": f"Bearer {LINKUP_API_KEY}"}
        params = {"q": query, "limit": 5}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.linkup.com/v1/jobs/search", headers=headers, params=params, timeout=20) as resp:
                    if resp.status != 200:
                        return f"[LinkUp error: {resp.status}]"
                    jobs = (await resp.json()).get("jobs", [])
                    return "\n".join([f"- {job.get('title','')} at {job.get('company','')} ({job.get('location','')})" for job in jobs])
        except Exception as e:
            logger.exception("LinkUp search failed")
            return f"LinkUp search failed: {e}"

class SerpApiClient:
    """SerpAPI search (async)."""
    async def search(self, query: str) -> str:
        if not SERPAPI_API_KEY:
            return "SerpAPI key not set."
        params = {"q": query, "api_key": SERPAPI_API_KEY, "num": 5, "engine": "google"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://serpapi.com/search", params=params, timeout=20) as resp:
                    if resp.status != 200:
                        return f"[SerpAPI error: {resp.status}]"
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
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            top = soup.find("a", class_="result__a")
            if not top or not top.get("href"):
                return "No clear top result."
            page_url = top["href"]
            page = requests.get(page_url, headers=headers, timeout=10)
            page.raise_for_status()
            page_soup = BeautifulSoup(page.text, "html.parser")
            paragraphs = page_soup.find_all("p")
            content = " ".join(p.get_text() for p in paragraphs[:5])
            return content[:1500] if content else "No content found."
        except Exception as e:
            logger.exception("WebScraper failed")
            return f"Web scraping failed: {e}"

# -----------------------
# Main Bot
# -----------------------
class PreMortemOpenRouterGPT5Bot(ForecastBot):
    """
    Pre-mortem bot that:
      - uses OpenRouter by default
      - uses GPT-5 (OpenAI) for final judgment when available
      - includes NewsAPI, SerpAPI, LinkUp research
      - supports Binary, Numeric, MultipleChoice
    """
    # default model config: narratives from OpenRouter 'gpt-o3', final judgment uses openai/gpt-5
    MODEL_CONFIG = {
        "narrative_primary": os.getenv("NARRATIVE_PRIMARY_MODEL", "openrouter/gpt-o3"),
        "narrative_secondary": os.getenv("NARRATIVE_SECONDARY_MODEL", "openrouter/gpt-o3"),
        # synthesis can be openrouter too
        "analytical_synthesis": os.getenv("ANALYTICAL_SYNTHESIS_MODEL", "openrouter/gpt-o3"),
        # explicit request to use GPT-5 for final judgment (if OPENAI_API_KEY present)
        "analytical_judgment": os.getenv("ANALYTICAL_JUDGMENT_MODEL", "openai/gpt-5"),
    }

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Gather research from NewsAPI, SerpAPI, LinkUp and web scraping concurrently."""
        logger.info(f"[{question.id}] Running combined research...")
        loop = asyncio.get_running_loop()

        def query_newsapi_sync():
            if not NEWSAPI_API_KEY:
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
            f"--- RESEARCH SUMMARY ---\n"
            f"NewsAPI:\n{research.get('newsapi','')}\n\n"
            f"SerpAPI:\n{research.get('serpapi','')}\n\n"
            f"LinkUp:\n{research.get('linkup','')}\n\n"
            f"Web Scrape:\n{research.get('web_scrape','')}\n"
            f"--- END RESEARCH ---\n"
        )

    async def _generate_narratives(self, model1: str, model2: str, prompt1: str, prompt2: str) -> Tuple[str, str]:
        """Generate two narratives in parallel with given models."""
        logger.info(f"Generating narratives using {model1} and {model2}")
        llm1 = llm_factory(model1)
        llm2 = llm_factory(model2)
        r1_task = llm1.invoke(prompt1, temperature=0.7, max_tokens=1200)
        r2_task = llm2.invoke(prompt2, temperature=0.7, max_tokens=1200)
        r1, r2 = await asyncio.gather(r1_task, r2_task)
        return r1, r2

    async def _run_forecast_on_binary(self, question: MetaculusQuestion, research: str):
        logger.info(f"[{question.id}] Binary pre-mortem analysis")
        fail_prompt = clean_indents(f"""One day after the resolution ({question.resolution_date}), the outcome of:
"{question.question_text}"
was a surprising NO. Write a plausible news article explaining what went wrong. Use the research below.\nResearch:\n{research}""")
        success_prompt = clean_indents(f"""One day after the resolution ({question.resolution_date}), the outcome of:
"{question.question_text}"
was a surprising YES. Write a plausible news article explaining what went right. Use the research below.\nResearch:\n{research}""")

        narratives = await self._generate_narratives(
            self.MODEL_CONFIG["narrative_primary"],
            self.MODEL_CONFIG["narrative_secondary"],
            fail_prompt,
            success_prompt,
        )

        synthesis_prompt = clean_indents(f"""Read the two narratives and extract key insights into a Markdown table with columns:
'Identified Risks (drives NO)' and 'Identified Opportunities (drives YES)'.

Failure Narrative:
{narratives[0]}

Success Narrative:
{narratives[1]}
""")
        synthesis = await llm_factory(self.MODEL_CONFIG["analytical_synthesis"]).invoke(
            synthesis_prompt, temperature=0.1, max_tokens=1200
        )

        judgment_prompt = clean_indents(f"""You are a superforecaster. Question: {question.question_text}
Based on the synthesized risks and opportunities below, provide a final probability that the question resolves YES.
Format EXACTLY as:
{{"probability_percent": XX, "rationale": "<detailed reasoning>" }}

Risks & Opportunities:
{synthesis}
""")
        judgment = await llm_factory(self.MODEL_CONFIG["analytical_judgment"]).invoke(
            judgment_prompt, temperature=0.0, max_tokens=800
        )

        self._print_analysis_results(question, narratives, synthesis, judgment, ("Failure Narrative", "Success Narrative"))

    async def _run_forecast_on_numeric(self, question: MetaculusQuestion, research: str):
        logger.info(f"[{question.id}] Numeric pre-mortem analysis")
        low_prompt = clean_indents(f"""One day after the resolution ({question.resolution_date}), the final number for:
"{question.question_text}"
was surprisingly LOW. Write a plausible news article explaining factors that suppressed the outcome. Use research below.\nResearch:\n{research}""")
        high_prompt = clean_indents(f"""One day after the resolution ({question.resolution_date}), the final number for:
"{question.question_text}"
was surprisingly HIGH. Write a plausible news article explaining catalysts that drove the outcome. Use research below.\nResearch:\n{research}""")

        narratives = await self._generate_narratives(
            self.MODEL_CONFIG["narrative_primary"],
            self.MODEL_CONFIG["narrative_secondary"],
            low_prompt,
            high_prompt,
        )

        synthesis_prompt = clean_indents(f"""Read the low and high narratives and extract key drivers into a Markdown table with columns: 'Downward Drivers' and 'Upward Drivers'.

Low Narrative:
{narratives[0]}

High Narrative:
{narratives[1]}
""")
        synthesis = await llm_factory(self.MODEL_CONFIG["analytical_synthesis"]).invoke(
            synthesis_prompt, temperature=0.1, max_tokens=1200
        )

        judgment_prompt = clean_indents(f"""You are a superforecaster. Question: {question.question_text}
Based on the synthesized drivers below, provide a numeric forecast.
Format EXACTLY as:
{{"point_forecast": <number>, "ci90_lower": <number>, "ci90_upper": <number>, "rationale": "<text>"}}

Drivers:
{synthesis}
""")
        judgment = await llm_factory(self.MODEL_CONFIG["analytical_judgment"]).invoke(
            judgment_prompt, temperature=0.0, max_tokens=800
        )

        self._print_analysis_results(question, narratives, synthesis, judgment, ("Low Narrative", "High Narrative"))

    async def _run_forecast_on_multiple_choice(self, question: MetaculusQuestion, research: str):
        """
        For Multiple Choice questions:
          - ask judgment model for a JSON array of objects:
              [{"index": 0, "probability": 30, "rationale":"..."}, ...]
          - Try to parse JSON; if parse fails, print raw output.
        """
        logger.info(f"[{question.id}] Multiple choice analysis")
        # attempt to get options from the question object
        options = getattr(question, "options", None) or getattr(question, "choices", None) or None
        # serialize options into numbered list
        if isinstance(options, dict):
            # some APIs return {index: label}
            option_lines = "\n".join([f"{i}: {v}" for i, v in options.items()])
        elif isinstance(options, (list, tuple)):
            option_lines = "\n".join([f"{i}: {opt}" for i, opt in enumerate(options)])
        else:
            # fallback: attempt to derive from question text (not ideal)
            option_lines = "Could not extract structured options; include question_text."
        prompt = clean_indents(f"""You are a forecasting assistant. Here is the question and the options:

Question:
{question.question_text}

Options:
{option_lines}

Research:
{research}

Return a JSON array where each element is:
{{"index": <option_index>, "probability": <0-100 numeric>, "rationale": "<brief justification>"}}
Ensure probabilities sum to ~100. Example:
[{{"index": 0, "probability": 20, "rationale":"..."}}, ...]
""")
        result_text = await llm_factory(self.MODEL_CONFIG["analytical_judgment"]).invoke(prompt, temperature=0.2, max_tokens=1200)

        # Try to extract JSON from LLM reply
        json_text = None
        try:
            # Look for a JSON array block in the response
            m = re.search(r"(\[.*\])", result_text, re.DOTALL)
            candidate = m.group(1) if m else result_text.strip()
            parsed = json.loads(candidate)
            # Validate structure minimally
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                json_text = parsed
        except Exception:
            json_text = None

        if json_text is None:
            # Could not parse JSON reliably; print raw output
            self._print_analysis_results(question, [], research, result_text, ("Multiple Choice (raw output)", ""))
            return

        # Print nicely
        pretty = json.dumps(json_text, indent=2)
        self._print_analysis_results(question, [], research, pretty, ("Multiple Choice (parsed)", ""))

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
            print(f"--- Research ---\n{synthesis}\n")
        print(f"--- FINAL JUDGMENT ({self.MODEL_CONFIG['analytical_judgment']}) ---\n{prediction}\n")
        print("=" * 90 + "\n")

# -----------------------
# Entrypoint
# -----------------------
async def main():
    parser = argparse.ArgumentParser(description="Run PreMortem OpenRouter+GPT5 Bot on Metaculus tournaments.")
    parser.add_argument("--tournament-ids", nargs="+", default=[str(32813), MetaculusApi.CURRENT_MINIBENCH_ID],
                        help="Metaculus tournament IDs (default: 32813 and minibench)")
    parser.add_argument("--models", nargs="+", help="Optional override models (primary secondary synthesis judgment)")
    args = parser.parse_args()

    if args.models and len(args.models) == 4:
        PreMortemOpenRouterGPT5Bot.MODEL_CONFIG = {
            "narrative_primary": args.models[0],
            "narrative_secondary": args.models[1],
            "analytical_synthesis": args.models[2],
            "analytical_judgment": args.models[3],
        }
        logger.info(f"Overrode MODEL_CONFIG via CLI: {PreMortemOpenRouterGPT5Bot.MODEL_CONFIG}")

    if not METACULUS_TOKEN:
        logger.error("METACULUS_TOKEN not set. The bot cannot fetch tournament questions.")
        return

    # instantiate bot with skip_previously_forecasted_questions = False to forecast ALL questions
    bot = PreMortemOpenRouterGPT5Bot(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,  # change to True if you want to publish
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,  # IMPORTANT: forecast all questions in tournament
    )

    # run tournaments
    await bot.forecast_on_tournaments(tournament_ids=args.tournament_ids, return_exceptions=True)

if __name__ == "__main__":
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except Exception:
        pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
