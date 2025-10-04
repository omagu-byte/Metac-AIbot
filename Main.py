# main.py
# Enhanced Conservative Forecasting Bot — Fully Corrected for OpenRouter

import argparse
import asyncio
import logging
import os
import re
from datetime import datetime

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
logger = logging.getLogger("ConservativeHybridBot")


class ConservativeHybridBot(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4.1-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/anthropic/claude-3.5-sonnet",
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
            "finance": "Use SEC filings, Bloomberg, Reuters, or official investor relations pages.",
        }

    def _get_domain_hint(self, text: str) -> str:
        t = text.lower()
        if any(kw in t for kw in ["revenue", "earnings", "stock", "sp500", "s&p 500", "market cap", "nasdaq", "nyse"]):
            return self._domain_hints["finance"]
        if any(kw in t for kw in ["gdp", "inflation", "interest rate", "fed", "ecb", "economy"]):
            return self._domain_hints["economic"]
        if any(kw in t for kw in ["virus", "vaccine", "disease", "who", "cdc", "health"]):
            return self._domain_hints["health"]
        if any(kw in t for kw in ["war", "election", "sanction", "un", "nato", "protest"]):
            return self._domain_hints["geopolitical"]
        if any(kw in t for kw in ["ai", "chip", "semiconductor", "arxiv", "model", "algorithm", "meta", "msft", "tsla", "ares", "mstr", "app"]):
            return self._domain_hints["tech"]
        if any(kw in t for kw in ["temperature", "co2", "emission", "ipcc", "climate", "weather"]):
            return self._domain_hints["climate"]
        return ""

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            hint = self._get_domain_hint(question.question_text)
            base = question.question_text
            variants = [
                f"What is the latest on: {base}?",
                f"What credible sources report about: {base}?",
                f"What are key uncertainties regarding: {base}?"
            ]
            if hint:
                variants = [f"{v} {hint}" for v in variants]

            tasks = []
            for v in variants:
                prompt = clean_indents(f"""
                Today: {datetime.now().strftime("%Y-%m-%d")}
                {v}
                """)
                tasks.append(self.get_llm("researcher", "llm").invoke(prompt))
            
            results = await asyncio.gather(*tasks)
            return "\n\n--- VARIANT ---\n".join(f"Variant {i+1}:\n{r}" for i, r in enumerate(results))

    async def _generate_narrative(self, question: MetaculusQuestion, research: str) -> str:
        draft_prompt = clean_indents(f"""
        You are a senior analyst. Write a concise, evidence-based narrative that explains
        key drivers, uncertainties, and plausible scenarios for this question.

        Question: {question.question_text}
        Research: {research}
        Today: {datetime.now().strftime("%Y-%m-%d")}
        """)
        draft = await self.get_llm("summarizer", "llm").invoke(draft_prompt)

        critique_prompt = clean_indents(f"""
        You are a skeptical red-team reviewer. Identify weaknesses, missing evidence,
        overconfident claims, or alternative interpretations in this analysis:
        {draft}
        """)
        critique = await self.get_llm("critiquer", "llm").invoke(critique_prompt)

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
                # Extract numbers using regex (no external deps)
                numbers = re.findall(r'\b\d+\.?\d*\b', research)
                float_numbers = []
                for n in numbers:
                    try:
                        val = float(n)
                        # Skip years (2000–2100) and extremely large outliers
                        if 0 <= val <= 1e9 and not (2000 <= val <= 2100):
                            float_numbers.append(val)
                    except ValueError:
                        continue
                
                if float_numbers:
                    numbers_str = ", ".join(f"{x:g}" for x in float_numbers[:5])
                    research += f"\n\nExtracted relevant numbers: {numbers_str}"

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
