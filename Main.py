import argparse
import asyncio
import logging
import os
import aiohttp
import requests
import json
import statistics
from datetime import datetime
from dotenv import load_dotenv

from forecasting_tools import (
    ForecastBot,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    BinaryQuestion,
    NumericQuestion,
    clean_indents,
)

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("PreMortemOpenRouterBot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# -------------------------------------------------------------------
# OpenRouter-only LLM client
# -------------------------------------------------------------------
class OpenRouterLlm:
    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        self.model = model
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    async def invoke(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=headers, json=payload, timeout=240) as resp:
                    txt = await resp.text()
                    if resp.status != 200:
                        logger.error(f"[OpenRouter {self.model}] status {resp.status}: {txt[:400]}")
                        return f"[error {resp.status}] {txt[:200]}"
                    data = await resp.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.exception(f"OpenRouter invoke error for {self.model}")
            return f"[exception] {e}"

# -------------------------------------------------------------------
# Bot implementation
# -------------------------------------------------------------------
class PreMortemOpenRouterBot(ForecastBot):
    def __init__(self, api_key: str, serpapi_key: str, newsapi_key: str, linkup_api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.serpapi_key = serpapi_key
        self.newsapi_key = newsapi_key
        self.linkup_api_key = linkup_api_key
        self.models = {
            "narrative_primary": "openrouter/gpt-5-mini",
            "judgment_models": [
                "openrouter/gpt-5",
                "openrouter/gpt-o3",
                "openrouter/gpt-5-nano"
            ]
        }

    def llm_factory(self, model: str) -> OpenRouterLlm:
        return OpenRouterLlm(model, self.api_key)

    def gather_research(self, query: str) -> str:
        serp_params = {
            "api_key": self.serpapi_key,
            "engine": "google",
            "q": query,
            "tbm": "nws"
        }
        news_params = {
            "apiKey": self.newsapi_key,
            "q": query,
            "pageSize": 10
        }

        serp_results = requests.get("https://serpapi.com/search", params=serp_params).json()
        news_results = requests.get("https://newsapi.org/v2/everything", params=news_params).json()

        texts = []
        for item in serp_results.get("news_results", []):
            texts.append(item.get("title", "") + ": " + item.get("snippet", ""))
        for article in news_results.get("articles", []):
            texts.append(article.get("title", "") + ": " + article.get("description", ""))

        return "\n".join(texts)

    async def forecast_question(self, question: MetaculusQuestion) -> str:
        logger.info(f"--- Forecasting question {question.id}: {question.title} ---")

        # Step 1: Gather research
        research_text = self.gather_research(question.title)
        logger.info(f"[Research] {question.id}:\n{research_text[:500]}...\n")

        # Step 2: Generate narrative
        prompt_narrative = f"Using the following research, write a compelling narrative for this forecasting question:\n\nQuestion: {question.title}\nURL: {question.url}\n\nResearch:\n{research_text}"
        narrative_llm = self.llm_factory(self.models["narrative_primary"])
        narrative = await narrative_llm.invoke(prompt_narrative, temperature=0.6)
        logger.info(f"[Narrative] {question.id}:\n{narrative}\n")

        # Step 3: Forecast using ensemble of models
        prompt_forecast = (
            f"Based on the narrative and research, provide a reasoned forecast.\n\n"
            f"Narrative:\n{narrative}\n\n"
            f"Research:\n{research_text}\n\n"
            f"Question:\n{question.title}\nType: {question.type}\nURL: {question.url}\n\n"
            f"Respond with a single numeric probability (0–100) or best guess value depending on question type."
        )

        forecasts = []
        raw_outputs = {}
        for model_name in self.models["judgment_models"]:
            llm = self.llm_factory(model_name)
            response = await llm.invoke(prompt_forecast, temperature=0.4)
            logger.info(f"[{model_name} Forecast] {question.id}:\n{response}\n")
            raw_outputs[model_name] = response
            try:
                value = float(response.strip().split()[0].replace("%", ""))
                forecasts.append(value)
            except Exception:
                logger.warning(f"Could not parse forecast from {model_name}: {response}")

        if forecasts:
            median_forecast = statistics.median(forecasts)
            disagreement = max(forecasts) - min(forecasts)
            logger.info(f"[Median Forecast] {question.id}: {median_forecast}")
            logger.info(f"[Model Disagreement] {question.id}: {disagreement:.2f}")

            # Step 4: Save to file
            output = {
                "question_id": question.id,
                "title": question.title,
                "url": question.url,
                "type": question.type,
                "narrative": narrative,
                "research": research_text,
                "forecasts": raw_outputs,
                "parsed_values": forecasts,
                "median_forecast": median_forecast,
                "disagreement": disagreement,
                "timestamp": datetime.utcnow().isoformat()
            }
            os.makedirs("forecasts", exist_ok=True)
            with open(f"forecasts/{question.id}.json", "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

            # Step 5: Post comment to Metaculus
            comment_text = (
                f"Forecast generated by ensemble of GPT models via OpenRouter.\n"
                f"Median forecast: {median_forecast:.2f}\n"
                f"Model disagreement: {disagreement:.2f}\n\n"
                f"Narrative:\n{narrative[:500]}..."
            )
            await self.metaculus_api.post_comment(question.id, comment_text)

            return f"Median forecast: {median_forecast:.2f} (disagreement: {disagreement:.2f})"
        else:
            return "Unable to compute forecast from model responses."

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(
        description="Run PreMortem OpenRouter ForecastBot on Metaculus tournaments."
    )
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        default=[str(32813), str(MetaculusApi.CURRENT_MINIBENCH_ID)],
        help="Metaculus tournament IDs"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    serpapi_key = os.getenv("SERP_API_KEY")
    newsapi_key = os.getenv("NEWS_API_KEY")
    linkup_api_key = os.getenv("LINKUP_API_KEY")

    if not all([api_key, serpapi_key, newsapi_key]):
        raise EnvironmentError("Missing one or more required API keys in environment.")

    bot = PreMortemOpenRouterBot(
        api_key=api_key,
        serpapi_key=serpapi_key,
        newsapi_key=newsapi_key,
        linkup_api_key=linkup_api_key,
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to="forecasts",
        skip_previously_forecasted_questions=False,
    )

    for tid in args.tournament_ids:
        logger.info
