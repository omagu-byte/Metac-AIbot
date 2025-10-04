# main.py
# Conservative Forecasting Bot — Tournament-Only
# Research: Tavily (primary) + Perplexity Sonar-Huge (fallback)
# Models: Claude 3.5 Sonnet (researcher), GPT-5 (summarizer), etc.

import argparse
import asyncio
import logging
import os
from datetime import datetime

import numpy as np
from tavily import TavilyClient
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
            "deep_researcher": "openrouter/perplexity/llama-3.1-sonar-huge-128k-online",  # fallback
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_key:
            raise EnvironmentError("TAVILY_API_KEY must be set.")
        self.tavily_client = TavilyClient(api_key=self.tavily_key)

    def _is_research_sufficient(self, text: str) -> bool:
        """Heuristic: Tavily result is sufficient if it has an answer and content."""
        if not text or not text.strip():
            return False
        has_answer = "Tavily Answer:" in text
        has_results = "Supporting Results:" in text
        min_length = len(text) > 150
        return (has_answer or has_results) and min_length

    def call_tavily(self, query: str) -> str:
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                include_answer=True,
                max_results=6,
            )
            answer = response.get("answer", "").strip()
            results = response.get("results", [])
            snippets = [
                f"- {res.get('title', '')}: {res.get('content', '')}"
                for res in results[:4]  # top 4
            ]
            parts = []
            if answer:
                parts.append(f"Tavily Answer:\n{answer}")
            if snippets:
                parts.append("Supporting Results:\n" + "\n".join(snippets))
            return "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"Tavily failed: {e}")
            return ""

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            loop = asyncio.get_running_loop()
            tavily_result = await loop.run_in_executor(None, self.call_tavily, question.question_text)

            if self._is_research_sufficient(tavily_result):
                logger.info("Tavily returned sufficient results.")
                return f"--- SOURCE TAVILY ---\n{tavily_result}\n\n"
            else:
                logger.info("Tavily result insufficient; falling back to Perplexity deep research.")
                fallback_prompt = (
                    f"Conduct a thorough, up-to-date investigation of the following question. "
                    f"Focus on credible sources, recent developments, and key uncertainties.\n\n"
                    f"Question: {question.question_text}\n\n"
                    f"Today's date: {datetime.now().strftime('%Y-%m-%d')}"
                )
                perplexity_result = await self.get_llm("deep_researcher", "llm").invoke(fallback_prompt)
                return f"--- SOURCE PERPLEXITY (FALLBACK) ---\n{perplexity_result}\n\n"

    # -----------------------------
    # Forecasting logic (unchanged)
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
