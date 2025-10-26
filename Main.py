import argparse
import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Literal, List

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

logger = logging.getLogger(__name__)

_ENSEMBLE_MODELS: List[str] = [
    "openrouter/openai/gpt-5",
    "openrouter/anthropic/claude-sonnet-4.5",
    "openrouter/openai/gpt-4o",
    "openrouter/openai/gpt-5",  # weighted twice
]


class ConfidentConservativeBot(ForecastBot):
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def _llm_config_defaults(self) -> dict[str, str]:
        return {
            "default": "openrouter/openai/gpt-5",
            "parser": "openrouter/openai/gpt-4o-mini",
            "summarizer": "openrouter/openai/gpt-5",
            "researcher": "openrouter/perplexity/llama-3.1-sonar-large-128k-online",
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
                ONLY assign extreme probabilities (≤0.1% or ≥99.9%) if the evidence is overwhelming and near-certain.

                Question: {question.question_text}
                Background: {question.background_info}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}
                Research Summary: {narrative}
                Today: {today_str}

                Before answering, consider:
                (a) Time until resolution
                (b) Status quo trajectory
                (c) Plausible Yes and No scenarios
                End with: "Probability: ZZ%"
                """)
                reasoning = await self.get_llm("default", "llm").invoke(prompt)
                pred: BinaryPrediction = await structure_output(
                    reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
                )
                decimal_pred = max(0.001, min(0.999, pred.prediction_in_decimal))
                return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

            elif isinstance(question, MultipleChoiceQuestion):
                prompt = clean_indents(f"""
                You are a confident, evidence-driven forecaster.
                Assign probabilities to each option based on the narrative. Avoid uniform distributions unless truly ignorant.

                Question: {question.question_text}
                Options: {question.options}
                Background: {question.background_info}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}
                Research Summary: {narrative}
                Today: {today_str}

                Before answering, consider status quo and plausible surprises.
                End with probabilities in this exact format:
                Option_A: Probability_A
                Option_B: Probability_B
                ...
                """)
                parsing_instructions = f"Valid options: {question.options}"
                reasoning = await self.get_llm("default", "llm").invoke(prompt)
                pred: PredictedOptionList = await structure_output(
                    reasoning,
                    PredictedOptionList,
                    model=self.get_llm("parser", "llm"),
                    additional_instructions=parsing_instructions,
                )
                return ReasonedPrediction(prediction_value=pred, reasoning=reasoning)

            elif isinstance(question, NumericQuestion):
                upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)
                prompt = clean_indents(f"""
                You are a confident, evidence-driven forecaster.
                Provide a calibrated numeric forecast with wide but justified intervals.

                Question: {question.question_text}
                Units: {question.unit_of_measure or 'Infer from context'}
                Background: {question.background_info}
                Resolution Criteria: {question.resolution_criteria}
                Fine Print: {question.fine_print}
                Research Summary: {narrative}
                Today: {today_str}
                {lower_msg}
                {upper_msg}

                Consider status quo, trends, expert views, and tail risks.
                End with:
                Percentile 10: XX
                Percentile 20: XX
                Percentile 40: XX
                Percentile 60: XX
                Percentile 80: XX
                Percentile 90: XX
                """)
                reasoning = await self.get_llm("default", "llm").invoke(prompt)
                percentiles: list[Percentile] = await structure_output(
                    reasoning, list[Percentile], model=self.get_llm("parser", "llm")
                )
                dist = NumericDistribution.from_question(percentiles, question)
                return ReasonedPrediction(prediction_value=dist, reasoning=reasoning)

        finally:
            if model_override and original_default:
                self._llms["default"] = original_default

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        narrative = await self._generate_narrative(question, research)
        forecasts = await asyncio.gather(*[
            self._single_forecast(question, narrative, research, model) for model in _ENSEMBLE_MODELS
        ], return_exceptions=True)

        valid_forecasts = [f for f in forecasts if not isinstance(f, Exception)]
        if not valid_forecasts:
            raise RuntimeError("All ensemble forecasts failed")

        avg_pred = sum(f.prediction_value for f in valid_forecasts) / len(valid_forecasts)
        combined_reasoning = "\n\n=== ENSEMBLE ===\n".join(f.reasoning for f in valid_forecasts)
        return ReasonedPrediction(prediction_value=avg_pred, reasoning=combined_reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        narrative = await self._generate_narrative(question, research)
        forecasts = await asyncio.gather(*[
            self._single_forecast(question, narrative, research, model) for model in _ENSEMBLE_MODELS
        ], return_exceptions=True)

        valid_forecasts = [f for f in forecasts if not isinstance(f, Exception)]
        if not valid_forecasts:
            raise RuntimeError("All ensemble forecasts failed")

        avg_probs = {}
        for opt in question.options:
            total = sum(f.prediction_value.get_probability_for_option(opt) for f in valid_forecasts)
            avg_probs[opt] = total / len(valid_forecasts)

        from forecasting_tools import PredictedOption
        avg_list = PredictedOptionList(predicted_options=[
            PredictedOption(option=opt, probability=prob) for opt, prob in avg_probs.items()
        ])
        combined_reasoning = "\n\n=== ENSEMBLE ===\n".join(f.reasoning for f in valid_forecasts)
        return ReasonedPrediction(prediction_value=avg_list, reasoning=combined_reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        narrative = await self._generate_narrative(question, research)
        forecasts = await asyncio.gather(*[
            self._single_forecast(question, narrative, research, model) for model in _ENSEMBLE_MODELS
        ], return_exceptions=True)

        valid_forecasts = [f for f in forecasts if not isinstance(f, Exception)]
        if not valid_forecasts:
            raise RuntimeError("All ensemble forecasts failed")

        avg_percentiles = []
        for i, p in enumerate([10, 20, 40, 60, 80, 90]):
            values = [f.prediction_value.declared_percentiles[i].value for f in valid_forecasts]
            avg_val = sum(values) / len(values)
            avg_percentiles.append(Percentile(percentile=p, value=avg_val))

        avg_dist = NumericDistribution.from_question(avg_percentiles, question)
        combined_reasoning = "\n\n=== ENSEMBLE ===\n".join(f.reasoning for f in valid_forecasts)
        return ReasonedPrediction(prediction_value=avg_dist, reasoning=combined_reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = f"The outcome cannot be higher than {upper_bound_number}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = f"The outcome cannot be lower than {lower_bound_number}."
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run ConfidentConservativeBot on tournaments 32813, 32831, and MiniBench"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournaments", "test_questions"],
        default="tournaments",
        help="Mode: 'tournaments' (32813, 32831, MiniBench) or 'test_questions'",
    )
    args = parser.parse_args()

    bot = ConfidentConservativeBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
    )

    if args.mode == "tournaments":
        logger.info("Forecasting on Tournament 32813...")
        reports_32813 = asyncio.run(bot.forecast_on_tournament(32813, return_exceptions=True))

        logger.info("Forecasting on Tournament 32831...")
        reports_32831 = asyncio.run(bot.forecast_on_tournament(32831, return_exceptions=True))

        logger.info("Forecasting on MiniBench...")
        reports_minibench = asyncio.run(
            bot.forecast_on_tournament(MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )

        forecast_reports = reports_32813 + reports_32831 + reports_minibench

    elif args.mode == "test_questions":
        # Corrected example questions based on knowledge base
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Multiple choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Numeric (discrete)
        ]
        bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            bot.forecast_questions(questions, return_exceptions=True)
        )

    bot.log_report_summary(forecast_reports)
