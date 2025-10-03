# main.py
# Final Conservative Forecasting Bot — Tournament-Only, OpenRouter-Only

import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

import numpy as np
import requests
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)

# -----------------------------
# Environment & Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConservativeHybridBot")


class ConservativeHybridBot(ForecastBot):
    """
    Tournament-only bot using:
    - Research: SerpApi + NewsAPI + Linkup + Perplexity (deep)
    - Models: gpt-5 (default/summarizer), gpt-4.1-mini (parser), claude-sonnet-4.5, qwen3
    - Committee: 4 forecasters → median
    - Compliance: structure_output + NumericDistribution.from_question()
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/perplexity/sonar-deep-research",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.serpapi_key = os.getenv("SERP_API_KEY")
        self.newsapi_key = os.getenv("NEWS_API_KEY")
        self.linkup_api_key = os.getenv("LINKUP_API_KEY")
        if not all([self.serpapi_key, self.newsapi_key, self.linkup_api_key]):
            raise EnvironmentError("SERP_API_KEY, NEWS_API_KEY, and LINKUP_API_KEY must be set.")

    # -----------------------------
    # Multi-Source Research (SerpApi + NewsAPI + Linkup + Perplexity)
    # -----------------------------
    def call_serpapi(self, query: str) -> str:
        if not self.serpapi_key: return ""
        try:
            response = requests.get(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": self.serpapi_key, "tbm": "nws"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return "\n".join([
                f"- {item.get('title', '')}: {item.get('snippet', '')}"
                for item in data.get("news_results", [])
            ])
        except Exception as e:
            return f"SerpApi failed: {e}"

    def call_newsapi(self, query: str) -> str:
        if not self.newsapi_key: return ""
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "apiKey": self.newsapi_key, "pageSize": 5},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return "\n".join([
                f"- {a.get('title', '')}: {a.get('description', '')}"
                for a in data.get("articles", [])
            ])
        except Exception as e:
            return f"NewsAPI failed: {e}"

    def call_linkup(self, query: str) -> str:
        if not self.linkup_api_key: return ""
        try:
            response = requests.post(
                "https://api.linkup.so/v1/search",
                headers={"Authorization": f"Bearer {self.linkup_api_key}"},
                json={"query": query},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return "\n".join([
                f"- {r.get('title', '')}: {r.get('snippet', '')}"
                for r in data.get("results", [])
            ])
        except Exception as e:
            return f"Linkup failed: {e}"

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            loop = asyncio.get_running_loop()
            tasks = {
                "serpapi": loop.run_in_executor(None, self.call_serpapi, question.question_text),
                "newsapi": loop.run_in_executor(None, self.call_newsapi, question.question_text),
                "linkup": loop.run_in_executor(None, self.call_linkup, question.question_text),
                "perplexity": self.get_llm("researcher", "llm").invoke(question.question_text)
            }
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            raw_research = ""
            for i, result in enumerate(results):
                raw_research += f"--- SOURCE {list(tasks.keys())[i].upper()} ---\n{result}\n\n"
            return raw_research

    # -----------------------------
    # Conservative Forecasting with 4-Model Committee
    # -----------------------------
    async def _generate_narrative(self, question: MetaculusQuestion, research: str) -> str:
        prompt = clean_indents(f"""
        You are a senior analyst. Write a concise, evidence-based narrative that explains
        key drivers, uncertainties, and plausible scenarios for this question.

        Question: {question.question_text}
        Research: {research}
        Today: {datetime.now().strftime("%Y-%m-%d")}
        """)
        return await self.get_llm("summarizer", "llm").invoke(prompt)

    async def _single_forecast(self, question, narrative: str, research: str, model_override: str = None):
        if model_override:
            self._llms["default"] = GeneralLlm(model=model_override)

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            Conservative professional forecaster.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Narrative: {narrative}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            Favor status quo. Avoid overconfidence.

            End with: "Probability: ZZ%"
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            pred: BinaryPrediction = await structure_output(reasoning, BinaryPrediction, model=self.get_llm("parser", "llm"))
            result = max(0.01), min(0.99, pred.prediction_in_decimal)

        elif isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(f"""
            Question: {question.question_text}
            Options: {question.options}
            Narrative: {narrative}
            Research: {research}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            Assign probabilities. No 0% unless logically impossible.

            End with probabilities for each option in order.
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            result = await structure_output(
                reasoning, PredictedOptionList, model=self.get_llm("parser", "llm"),
                additional_instructions=f"Options must be exactly: {question.options}"
            )

        elif isinstance(question, NumericQuestion):
            lower_msg = f"Lower bound: {'open' if question.open_lower_bound else 'closed'} at {question.lower_bound or question.nominal_lower_bound}"
            upper_msg = f"Upper bound: {'open' if question.open_upper_bound else 'closed'} at {question.upper_bound or question.nominal_upper_bound}"
            prompt = clean_indents(f"""
            Conservative forecaster. Set wide 90/10 intervals.

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Infer'}
            Narrative: {narrative}
            Research: {research}
            {lower_msg}
            {upper_msg}
            Today: {datetime.now().strftime("%Y-%m-%d")}

            Provide percentiles: 10, 20, 40, 60, 80, 90.
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            percentile_list: list[Percentile] = await structure_output(reasoning, list[Percentile], model=self.get_llm("parser", "llm"))
            result = NumericDistribution.from_question(percentile_list, question)

        if model_override:
            self._llms["default"] = GeneralLlm(model="openrouter/openai/gpt-5")

        return result, reasoning

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        narrative = await self._generate_narrative(question, research)
        forecasts = []
        reasonings = []
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-sonnet-4.5",
            "openrouter/qwen/qwen3-235b-a22b-thinking-2507"
        ]
        for model in models:
            pred, reason = await self._single_forecast(question, narrative, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
        median_pred = float(np.median(forecasts))
        return ReasonedPrediction(prediction_value=median_pred, reasoning=" | ".join(reasonings))

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        narrative = await self._generate_narrative(question, research)
        forecasts = []
        reasonings = []
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-sonnet-4.5",
            "openrouter/qwen/qwen3-235b-a22b-thinking-2507"
        ]
        for model in models:
            pred, reason = await self._single_forecast(question, narrative, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
        all_probs = np.array([[opt["probability"] for opt in f.predicted_options] for f in forecasts])
        median_probs = np.median(all_probs, axis=0)
        if median_probs.sum() > 0:
            median_probs = median_probs / median_probs.sum()
        else:
            median_probs = np.full_like(median_probs, 1.0 / len(median_probs))
        options = forecasts[0].predicted_options
        median_forecast = PredictedOptionList([
            {"option": opt["option"], "probability": float(p)} for opt, p in zip(options, median_probs)
        ])
        return ReasonedPrediction(prediction_value=median_forecast, reasoning=" | ".join(reasonings))

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        narrative = await self._generate_narrative(question, research)
        forecasts = []
        reasonings = []
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-5",
            "openrouter/anthropic/claude-sonnet-4.5",
            "openrouter/qwen/qwen3-235b-a22b-thinking-2507"
        ]
        for model in models:
            pred, reason = await self._single_forecast(question, narrative, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
        target_percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        aggregated = []
        for p in target_percentiles:
            values = []
            for f in forecasts:
                for item in f.declared_percentiles:
                    if abs(item.percentile - p) < 0.01:
                        values.append(item.value)
                        break
                else:
                    values.append(0.0)
            median_val = float(np.median(values))
            aggregated.append(Percentile(percentile=p, value=median_val))
        distribution = NumericDistribution.from_question(aggregated, question)
        return ReasonedPrediction(prediction_value=distribution, reasoning=" | ".join(reasonings))


# -----------------------------
# Entrypoint — Tournament Only
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Conservative Hybrid Bot.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["32813", "market-pulse-25q4", MetaculusApi.CURRENT_MINIBENCH_ID],
    )
    args = parser.parse_args()

    bot = ConservativeHybridBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )

    try:
        all_reports = []
        for tid in args.tournament_ids:
            logger.info(f"Forecasting on tournament: {tid}")
            reports = asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
            all_reports.extend(reports)
        bot.log_report_summary(all_reports)
        logger.info("Run completed successfully.")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
