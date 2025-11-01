# main.py
# Conservative Hybrid Forecasting Bot — Tournament-Only, OpenRouter-Only
# Research powered solely by GPT-5 and Claude Sonnet 4.5
# No external APIs. No numpy. No requests. No news.

import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import List

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
# Helper: Pure-Python median
# -----------------------------
def median(lst: List[float]) -> float:
    """Compute the median of a list of numbers without numpy."""
    if not lst:
        raise ValueError("median() arg is an empty sequence")
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0
    else:
        return float(sorted_lst[mid])

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Yrambot")


class Yrambot(ForecastBot):
    """
    Yrambot: Conservative hybrid forecaster using only GPT-5 and Claude Sonnet 4.5.
    No external news or web search — relies on model knowledge with strong temporal awareness.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher_gpt": "openrouter/openai/gpt-5",
            "researcher_claude": "openrouter/anthropic/claude-sonnet-4.5",
        }

    # -----------------------------
    # Dual-Model Research (GPT-5 + Claude Sonnet)
    # -----------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            gpt_prompt = clean_indents(f"""
            You are an expert researcher with deep world knowledge up to June 2024, but you understand that today is {today_str}.
            Analyze the following forecasting question with attention to recent developments, trends, and timing.

            Question: {question.question_text}
            Background: {question.background_info or 'None provided'}
            Resolution criteria: {question.resolution_criteria or 'Standard'}
            Fine print: {question.fine_print or 'None'}

            Consider:
            - How much time remains until resolution?
            - Has anything changed recently that affects this outcome?
            - What is the status quo? (World changes slowly.)
            - Are there known upcoming events (elections, product launches, policy deadlines)?
            - If uncertain, say so. Do not hallucinate.

            Provide a concise, factual summary for a professional forecaster.
            """)
            
            claude_prompt = clean_indents(f"""
            You are Claude Sonnet 4.5, a precise and cautious AI with knowledge cutoff in early 2024, but aware that today is {today_str}.
            Your task: analyze the forecasting question below with strong temporal reasoning.

            Question: {question.question_text}
            Context: {question.background_info or 'Not specified'}
            Resolution rules: {question.resolution_criteria or 'Default'}

            Focus on:
            - Recency: Is this question about a near-term or long-term event?
            - Plausibility given current date ({today_str})
            - Base rates and historical analogs
            - Known constraints or scheduled events before resolution

            Be honest about uncertainty. Avoid speculation beyond your knowledge.

            Output only relevant facts and reasoned considerations.
            """)

            try:
                gpt_response = await self.get_llm("researcher_gpt", "llm").invoke(gpt_prompt)
            except Exception as e:
                gpt_response = f"[GPT-5 research failed: {str(e)}]"

            try:
                claude_response = await self.get_llm("researcher_claude", "llm").invoke(claude_prompt)
            except Exception as e:
                claude_response = f"[Claude Sonnet research failed: {str(e)}]"

            return (
                f"--- RESEARCH FROM GPT-5 (as of {today_str}) ---\n{gpt_response}\n\n"
                f"--- RESEARCH FROM CLAUDE SONNET 4.5 (as of {today_str}) ---\n{claude_response}\n"
            )

    # -----------------------------
    # Conservative Forecasting with Committee 
    # -----------------------------
    async def _single_forecast(self, question, research: str, model_override: str = None):
        if model_override:
            self._llms["default"] = GeneralLlm(model=model_override)
            self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-4.1-mini")

        today_str = datetime.now().strftime("%Y-%m-%d")

        if isinstance(question, BinaryQuestion):
            prompt = clean_indents(f"""
            You are a professional forecaster known for conservative, well-calibrated predictions.
            Today is {today_str}.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Research: {research}

            Consider:
            (a) Time until resolution — near-term vs long-term
            (b) Status quo bias — the world changes slowly
            (c) Base rates (e.g., ~30% for major geopolitical disruptions)
            (d) Model disagreements in research

            Be humble. Avoid overconfidence. Anchor to community estimates if known.

            End with: "Probability: ZZ%"
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            pred: BinaryPrediction = await structure_output(reasoning, BinaryPrediction, model=self.get_llm("parser", "llm"))
            result = max(0.01, min(0.99, pred.prediction_in_decimal))

        elif isinstance(question, MultipleChoiceQuestion):
            prompt = clean_indents(f"""
            Conservative forecaster mode. Today: {today_str}.

            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            Research: {research}

            Assign probabilities. Do not assign 0% to any option unless logically impossible.
            Prefer modest shifts from uniform unless evidence strongly supports otherwise.

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
            Conservative forecaster. Today: {today_str}. Set wide 90/10 intervals.

            Question: {question.question_text}
            Units: {question.unit_of_measure or 'Infer from context'}
            Background: {question.background_info}
            Resolution: {question.resolution_criteria}
            {lower_msg}
            {upper_msg}
            Research: {research}

            Consider:
            - Status quo and recent trends
            - Expert consensus (if implied in research)
            - Black swans: allow tail risk but don't overweight
            - Time horizon: short-term → tighter; long-term → wider

            Provide percentiles: 10, 20, 40, 60, 80, 90.
            """)
            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            percentile_list: list[Percentile] = await structure_output(reasoning, list[Percentile], model=self.get_llm("parser", "llm"))
            result = NumericDistribution.from_question(percentile_list, question)

        if model_override:
            # Restore defaults
            self._llms["default"] = GeneralLlm(model="openrouter/openai/gpt-5")
            self._llms["parser"] = GeneralLlm(model="openrouter/openai/gpt-4.1-mini")

        return result, reasoning

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        forecasts = []
        reasonings = []
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-4o",
            "openrouter/anthropic/claude-sonnet-4.5"
        ]
        for model in models:
            pred, reason = await self._single_forecast(question, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
        median_pred = median(forecasts)
        return ReasonedPrediction(prediction_value=median_pred, reasoning=" | ".join(reasonings))

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        forecasts = []
        reasonings = []
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-4o",
            "openrouter/anthropic/claude-sonnet-4.5"
        ]
        for model in models:
            pred, reason = await self._single_forecast(question, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)

        num_options = len(forecasts[0].predicted_options)
        median_probs = []
        for i in range(num_options):
            probs = [f.predicted_options[i]["probability"] for f in forecasts]
            median_probs.append(median(probs))

        total = sum(median_probs)
        if total > 0:
            median_probs = [p / total for p in median_probs]
        else:
            uniform_prob = 1.0 / len(median_probs)
            median_probs = [uniform_prob] * len(median_probs)

        options = forecasts[0].predicted_options
        median_forecast = PredictedOptionList([
            {"option": opt["option"], "probability": float(p)} for opt, p in zip(options, median_probs)
        ])
        return ReasonedPrediction(prediction_value=median_forecast, reasoning=" | ".join(reasonings))

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str) -> ReasonedPrediction[NumericDistribution]:
        forecasts = []
        reasonings = []
        models = [
            "openrouter/openai/gpt-5",
            "openrouter/openai/gpt-4o",
            "openrouter/anthropic/claude-sonnet-4.5"
        ]
        for model in models:
            pred, reason = await self._single_forecast(question, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)

        target_percentiles = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        aggregated = []
        for p in target_percentiles:
            values = []
            for f in forecasts:
                matched = False
                for item in f.declared_percentiles:
                    if abs(item.percentile - p) < 0.01:
                        values.append(item.value)
                        matched = True
                        break
                if not matched:
                    values.append(0.0)
            median_val = median(values)
            aggregated.append(Percentile(percentile=p, value=median_val))
        distribution = NumericDistribution.from_question(aggregated, question)
        return ReasonedPrediction(prediction_value=distribution, reasoning=" | ".join(reasonings))


# -----------------------------
# Entrypoint — Tournament Only
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Conservative Hybrid Bot (GPT-5 + Claude only).")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["32813", "market-pulse-25q4", MetaculusApi.CURRENT_MINIBENCH_ID],
    )
    args = parser.parse_args()

    bot = Yrambot(
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
