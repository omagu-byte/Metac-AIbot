# main.py
# Enhanced Conservative Forecasting Bot — Confident & Risk-Aware Mode
# Model names preserved as requested

import argparse
import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import List

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConfidentConservativeBot")


_ENSEMBLE_MODELS: List[str] = [
    "openrouter/openai/gpt-5",
    "openrouter/anthropic/claude-sonnet-4.5",
    "openrouter/openai/gpt-4o",
    "openrouter/openai/gpt-5",
]


class ConfidentConservativeBot(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/openai/gpt-5",
            "critiquer": "openrouter/openai/gpt-5",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._domain_hints = {
            "economic": "Prioritize data from FRED, World Bank, IMF, OECD, or central bank reports.",
            "health": "Focus on WHO, CDC, ECDC, Lancet, NEJM, or clinicaltrials.gov.",
            "geopolitical": "Use UN, Reuters, Associated Press, official government statements, or BBC.",
            "tech": "Reference arXiv, IEEE, official company blogs, or credible tech journals.",
            "climate": "Use IPCC, NOAA, NASA, or peer-reviewed climate science sources.",
            "finance": "Use SEC filings, Bloomberg, Reuters, or official investor relations pages.",
        }
        self._domain_patterns = {
            "finance": re.compile(r"\b(revenue|earnings|stock|sp500|s&p 500|market cap|nasdaq|nyse)\b", re.IGNORECASE),
            "economic": re.compile(r"\b(gdp|inflation|interest rate|fed|ecb|economy)\b", re.IGNORECASE),
            "health": re.compile(r"\b(virus|vaccine|disease|who|cdc|health)\b", re.IGNORECASE),
            "geopolitical": re.compile(r"\b(war|election|sanction|un|nato|protest)\b", re.IGNORECASE),
            "tech": re.compile(r"\b(ai|chip|semiconductor|arxiv|model|algorithm|meta|msft|tsla|ares|mstr|app)\b", re.IGNORECASE),
            "climate": re.compile(r"\b(temperature|co2|emission|ipcc|climate|weather)\b", re.IGNORECASE),
        }

    def _get_domain_hint(self, text: str) -> str:
        for domain, pattern in self._domain_patterns.items():
            if pattern.search(text):
                return self._domain_hints[domain]
        return ""

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            hint = self._get_domain_hint(question.question_text)
            base = question.question_text
            variants = [
                f"What is the latest credible evidence on: {base}?",
                f"What would strongly update a rational forecaster's view on: {base}?",
                f"What are the most decisive indicators or inflection points for: {base}?"
            ]
            if hint:
                variants = [f"{v} {hint}" for v in variants]

            tasks = []
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            for v in variants:
                prompt = clean_indents(f"""
                Today: {today_str}
                {v}
                """)
                tasks.append(self.get_llm("researcher", "llm").invoke(prompt))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            safe_results = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    logger.warning(f"Research variant {i+1} failed: {r}")
                    safe_results.append(f"[ERROR: {str(r)}]")
                else:
                    safe_results.append(r)
            return "\n\n--- VARIANT ---\n".join(f"Variant {i+1}:\n{r}" for i, r in enumerate(safe_results))

    async def _generate_narrative(self, question: MetaculusQuestion, research: str) -> str:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        draft_prompt = clean_indents(f"""
        You are a top-tier geopolitical/economic/technical forecaster with a track record of bold, correct calls.
        Synthesize the research into a decisive, evidence-based narrative.
        Identify the MOST LIKELY outcome and key inflection points.
        Do not hedge unnecessarily—state clear conclusions when evidence supports them.

        Question: {repr(question.question_text)}
        Research: {research}
        Today: {today_str}
        """)
        draft = await self.get_llm("summarizer", "llm").invoke(draft_prompt)

        critique_prompt = clean_indents(f"""
        You are a skeptical red-team reviewer. Challenge overconfidence.
        Is the evidence truly sufficient for such a strong conclusion?
        What high-impact scenarios are being underweighted?
        """)
        critique = await self.get_llm("critiquer", "llm").invoke(critique_prompt)

        final_prompt = clean_indents(f"""
        Revise the analysis: keep bold conclusions ONLY if they survive scrutiny.
        Explicitly state when the evidence justifies high confidence vs. when uncertainty remains.
        Never claim certainty unless logically necessary.

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

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        try:
            if isinstance(question, BinaryQuestion):
                prompt = clean_indents(f"""
                You are a confident, evidence-driven forecaster.
                Make a decisive probability assessment—even if it strongly deviates from base rates.
                ONLY assign extreme probabilities (≤0.1% or ≥99.9%) if the outcome is virtually certain or impossible.

                Question: {repr(question.question_text)}
                Background: {repr(question.background_info or 'None')}
                Resolution: {repr(question.resolution_criteria or 'None')}
                Fine print: {repr(question.fine_print or 'None')}
                Narrative: {narrative}
                Research: {research}
                Today: {today_str}

                End with: "Probability: ZZ%"
                """)
                reasoning = await self.get_llm("default", "llm").invoke(prompt)
                try:
                    pred: BinaryPrediction = await structure_output(
                        reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
                    )
                    # Allow bolder range: [0.001, 0.999]
                    result = max(0.001, min(0.999, pred.prediction_in_decimal))
                except Exception as e:
                    logger.warning(f"Failed to parse binary prediction: {e}. Using 0.5.")
                    result = 0.5
                    reasoning += f"\n[PARSING FAILED: {e}]"

            elif isinstance(question, MultipleChoiceQuestion):
                options_repr = repr(question.options)
                prompt = clean_indents(f"""
                Assign decisive probabilities based on evidence—even if one option dominates.
                Avoid uniformity unless truly no signal exists.
                Zero probability ONLY if logically impossible.

                Question: {repr(question.question_text)}
                Options: {options_repr}
                Narrative: {narrative}
                Research: {research}
                Today: {today_str}
                """)
                reasoning = await self.get_llm("default", "llm").invoke(prompt)
                try:
                    result = await structure_output(
                        reasoning, PredictedOptionList, model=self.get_llm("parser", "llm"),
                        additional_instructions=f"Options must be exactly: {options_repr}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse MC prediction: {e}. Using uniform.")
                    uniform_prob = 1.0 / len(question.options)
                    result = PredictedOptionList([
                        {"option": opt, "probability": uniform_prob} for opt in question.options
                    ])
                    reasoning += f"\n[PARSING FAILED: {e}]"

            elif isinstance(question, NumericQuestion):
                lower_msg = f"Lower bound: {'open' if question.open_lower_bound else 'closed'} at {question.lower_bound or question.nominal_lower_bound}"
                upper_msg = f"Upper bound: {'open' if question.open_upper_bound else 'closed'} at {question.upper_bound or question.nominal_upper_bound}"
                prompt = clean_indents(f"""
                You may assign narrow intervals IF the narrative shows high confidence.
                Otherwise, default to wide, conservative intervals.
                Justify tight forecasts with specific evidence.

                Question: {repr(question.question_text)}
                Units: {repr(question.unit_of_measure or 'Infer')}
                Narrative: {narrative}
                Research: {research}
                {lower_msg}
                {upper_msg}
                Today: {today_str}

                Provide percentiles: 10, 20, 40, 60, 80, 90.
                """)
                reasoning = await self.get_llm("default", "llm").invoke(prompt)
                try:
                    percentile_list: list[Percentile] = await structure_output(
                        reasoning, list[Percentile], model=self.get_llm("parser", "llm")
                    )
                    result = NumericDistribution.from_question(percentile_list, question)
                except Exception as e:
                    logger.warning(f"Failed to parse numeric prediction: {e}. Using fallback.")
                    lb = question.nominal_lower_bound or 0
                    ub = question.nominal_upper_bound or 100
                    if question.open_lower_bound:
                        lb = ub - 1000 if ub else -1000
                    if question.open_upper_bound:
                        ub = lb + 1000 if lb else 1000
                    mid = (lb + ub) / 2
                    fallback_percentiles = [
                        Percentile(percentile=0.1, value=lb),
                        Percentile(percentile=0.2, value=lb + (mid - lb) * 0.3),
                        Percentile(percentile=0.4, value=mid - (ub - mid) * 0.3),
                        Percentile(percentile=0.6, value=mid + (ub - mid) * 0.3),
                        Percentile(percentile=0.8, value=ub - (ub - mid) * 0.3),
                        Percentile(percentile=0.9, value=ub),
                    ]
                    result = NumericDistribution.from_question(fallback_percentiles, question)
                    reasoning += f"\n[PARSING FAILED: {e}]"

            return result, reasoning
        finally:
            if model_override:
                self._llms["default"] = original_default

    # Keep the same ensemble logic as before (using _ENSEMBLE_MODELS)
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        narrative = await self._generate_narrative(question, research)
        forecasts = []
        reasonings = []
        for model in _ENSEMBLE_MODELS:
            pred, reason = await self._single_forecast(question, narrative, research, model_override=model)
            forecasts.append(pred)
            reasonings.append(reason)
        median_pred = float(np.median(forecasts))
        return ReasonedPrediction(prediction_value=median_pred, reasoning=" | ".join(reasonings))

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str) -> ReasonedPrediction[PredictedOptionList]:
        narrative = await self._generate_narrative(question, research)
        forecasts = []
        reasonings = []
        for model in _ENSEMBLE_MODELS:
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
        for model in _ENSEMBLE_MODELS:
            pred, reason = await self._single_forecast(question, narrative, research, model_override=model)
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
            median_val = float(np.median(values))
            aggregated.append(Percentile(percentile=p, value=median_val))
        distribution = NumericDistribution.from_question(aggregated, question)
        return ReasonedPrediction(prediction_value=distribution, reasoning=" | ".join(reasonings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Confident Conservative Hybrid Bot.")
    parser.add_argument(
        "--tournament-ids",
        nargs="+",
        type=str,
        default=["32813", "fiscal", "metaculus-cup-fall-2025", "market-pulse-25q4", MetaculusApi.CURRENT_MINIBENCH_ID],
    )
    args = parser.parse_args()

    bot = ConfidentConservativeBot(
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
