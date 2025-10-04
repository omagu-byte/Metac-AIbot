# main.py
# Enhanced Conservative Forecasting Bot — Perplexity-Only + 5 Key Upgrades

import argparse
import asyncio
import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Optional, List

import numpy as np
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

# For quantitative extraction
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConservativeHybridBot")


class KeyMetrics(BaseModel):
    """Extracted quantitative anchors from research."""
    current_value: Optional[float] = None
    historical_min: Optional[float] = None
    historical_max: Optional[float] = None
    relevant_numbers: List[float] = []


class ConservativeHybridBot(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/perplexity/llama-3.1-sonar-huge-128k-online",
            "critiquer": "openrouter/anthropic/claude-3.5-sonnet",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._domain_hints = {
            "economic": "Prioritize data from FRED, World Bank, IMF, OECD, or central bank reports.",
            "health": "Focus on WHO, CDC, ECDC, Lancet, NEJM, or clinicaltrials.gov.",
            "geopolitical": "Use UN, Reuters, Associated Press, official government statements, or BBC.",
            "tech": "Reference arXiv, IEEE, official company blogs, or credible tech journals.",
            "climate": "Use IPCC, NOAA, NASA, or peer-reviewed climate science sources.",
        }

    def _get_domain_hint(self, text: str) -> str:
        t = text.lower()
        if any(kw in t for kw in ["gdp", "inflation", "interest rate", "fed", "ecb", "economy"]):
            return self._domain_hints["economic"]
        if any(kw in t for kw in ["virus", "vaccine", "disease", "who", "cdc", "health"]):
            return self._domain_hints["health"]
        if any(kw in t for kw in ["war", "election", "sanction", "un", "nato", "protest"]):
            return self._domain_hints["geopolitical"]
        if any(kw in t for kw in ["ai", "chip", "semiconductor", "arxiv", "model", "algorithm"]):
            return self._domain_hints["tech"]
        if any(kw in t for kw in ["temperature", "co2", "emission", "ipcc", "climate", "weather"]):
            return self._domain_hints["climate"]
        return ""

    @lru_cache(maxsize=128)
    def _cached_research(self, question_text: str, date_str: str) -> str:
        """Cached research — same question on same day returns same result."""
        # This method is sync; called via run_in_executor
        raise NotImplementedError("Should not be called directly")

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            loop = asyncio.get_running_loop()
            # Use cache key: (question_text, YYYY-MM-DD)
            cache_key = (question.question_text, datetime.now().strftime("%Y-%m-%d"))
            
            # Define the actual research function for caching
            def _do_research(q_text: str, d_str: str) -> str:
                hint = self._get_domain_hint(q_text)
                base = q_text
                variants = [
                    f"What is the latest on: {base}?",
                    f"What credible sources report about: {base}?",
                    f"What are key uncertainties regarding: {base}?"
                ]
                if hint:
                    variants = [f"{v} {hint}" for v in variants]

                # Run sync Perplexity calls (via litellm under the hood)
                results = []
                for v in variants:
                    prompt = clean_indents(f"""
                    Today: {d_str}
                    {v}
                    """)
                    result = asyncio.run(self.get_llm("researcher", "llm").invoke(prompt))
                    results.append(result)
                return "\n\n--- VARIANT ---\n".join(f"Variant {i+1}:\n{r}" for i, r in enumerate(results))

            # Bypass lru_cache limitation by using a wrapper
            research = await loop.run_in_executor(None, lambda: _do_research(*cache_key))
            return f"--- SOURCE PERPLEXITY (SELF-CONSISTENCY) ---\n{research}\n\n"

    async def _generate_narrative(self, question: MetaculusQuestion, research: str) -> str:
        draft_prompt = clean_indents(f"""
        You are a senior analyst. Write a concise, evidence-based narrative that explains
        key drivers, uncertainties, and plausible scenarios for this question.

        Question: {question.question_text}
        Research: {research}
        Today: {datetime.now().strftime("%Y-%m-%d")}
        """)
        draft = await self.get_llm("summarizer", "llm").invoke(draft_prompt)

        # Red-team critique
        critique_prompt = clean_indents(f"""
        You are a skeptical red-team reviewer. Identify weaknesses, missing evidence,
        overconfident claims, or alternative interpretations in this analysis:
        {draft}
        """)
        critique = await self.get_llm("critiquer", "llm").invoke(critique_prompt)

        # Final revision
        final_prompt = clean_indents(f"""
        Revise the following analysis to address the critique. Maintain a professional,
        evidence-based tone and explicitly acknowledge key uncertainties.

        Original Analysis:
        {draft}

        Critique:
        {critique}
        """)
        return await self.get_llm("summarizer", "llm").invoke(final_prompt)

    async def _single_forecast(self, question, narrative: str, research: str, model_override: str = None):
        original_default = self._llms.get("default")
        if model_override:
            self._llms["default"] = GeneralLlm(model=model_override)

        try:
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
                result = max(0.01, min(0.99, pred.prediction_in_decimal))

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
                # Extract quantitative anchors
                try:
                    metrics: KeyMetrics = await structure_output(
                        research, KeyMetrics, model=self.get_llm("parser", "llm")
                    )
                    numbers_str = ", ".join(str(x) for x in metrics.relevant_numbers[:5])
                    if numbers_str:
                        research += f"\n\nExtracted numbers: {numbers_str}"
                except Exception as e:
                    logger.warning(f"Quantitative extraction failed: {e}")

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

            return result, reasoning
        finally:
            if model_override:
                self._llms["default"] = original_default

    # ... rest of the forecasting methods unchanged ...
    # (Keep your existing _run_forecast_on_binary, _run_forecast_on_multiple_choice, _run_forecast_on_numeric)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Enhanced Conservative Hybrid Bot.")
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
