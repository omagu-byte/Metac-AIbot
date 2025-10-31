import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal

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


class FallTemplateBot2025(ForecastBot):
    """
    Custom forecasting bot for Fall 2025 Metaculus AI Tournament.
    
    Key changes from template:
    - NO external data retrieval (no web scraping, no news APIs)
    - All "research" is performed by gpt-5 using its internal knowledge
    - Uses only OpenRouter models:
        * researcher: openrouter/openai/gpt-5
        * default forecaster: openrouter/anthropic/claude-sonnet-4.5
        * parser: openrouter/openai/gpt-4o
    """

    _max_concurrent_questions = 3  # Adjust based on OpenRouter rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            # Use ONLY gpt-5 as researcher — no external tools
            researcher = self.get_llm("researcher")  # Should be gpt-5

            prompt = clean_indents(
                f"""
                You are an expert research assistant for a superforecaster.
                Your task is to provide a concise, factual, and up-to-date summary relevant to forecasting the following question.
                Use your internal knowledge (training cutoff included) to recall key facts, trends, historical precedents, expert opinions, and contextual background.
                Do NOT speculate. Only state what is reasonably known or inferable from public knowledge as of today.
                Do NOT make forecasts—only provide neutral, useful context.

                Question:
                {question.question_text}

                Resolution Criteria:
                {question.resolution_criteria}

                Fine Print:
                {question.fine_print}
                """
            )

            research = await researcher.invoke(prompt)
            logger.info(f"Research for {question.page_url}:\n{research}")
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # Rotate or select forecaster — here we use claude by default
        forecaster = self.get_llm("default")

        prompt = clean_indents(
            f"""
            You are a professional superforecaster.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution Criteria:
            {question.resolution_criteria}

            Fine Print:
            {question.fine_print}

            Research Summary (from your assistant):
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before giving your probability, analyze:
            (a) Time until resolution.
            (b) Status quo trajectory.
            (c) Plausible scenario leading to "No".
            (d) Plausible scenario leading to "Yes".

            Remember: The world changes slowly. Favor base rates and reference classes.
            Be calibrated—avoid overconfidence.

            End with: "Probability: ZZ%"
            """
        )
        reasoning = await forecaster.invoke(prompt)
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        forecaster = self.get_llm("default")
        prompt = clean_indents(
            f"""
            You are a professional superforecaster.

            Question:
            {question.question_text}

            Options: {question.options}

            Background:
            {question.background_info}

            Resolution Criteria:
            {question.resolution_criteria}

            Fine Print:
            {question.fine_print}

            Research Summary:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Analyze:
            (a) Time until resolution.
            (b) Most likely option under status quo.
            (c) One plausible surprise outcome.

            Assign probabilities to ALL options. Avoid zero probabilities.
            Favor moderate uncertainty unless evidence is overwhelming.

            End with lines like:
            Option_X: YY%
            ...
            """
        )
        parsing_instructions = f"Valid options: {question.options}"
        reasoning = await forecaster.invoke(prompt)
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser"),
            additional_instructions=parsing_instructions,
        )
        return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        forecaster = self.get_llm("default")
        upper_msg, lower_msg = self._create_upper_and_lower_bound_messages(question)

        prompt = clean_indents(
            f"""
            You are a professional superforecaster.

            Question:
            {question.question_text}

            Units: {question.unit_of_measure or 'Infer from context'}

            Background:
            {question.background_info}

            Resolution Criteria:
            {question.resolution_criteria}

            Fine Print:
            {question.fine_print}

            Research Summary:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_msg}
            {upper_msg}

            Analyze:
            (a) Time until resolution.
            (b) Current best estimate / status quo.
            (c) Trend extrapolation.
            (d) Expert consensus.
            (e) Low-outcome surprise scenario.
            (f) High-outcome surprise scenario.

            Be humble—set wide confidence intervals.

            End with:
            Percentile 10: X
            Percentile 20: X
            Percentile 40: X
            Percentile 60: X
            Percentile 80: X
            Percentile 90: X
            """
        )
        reasoning = await forecaster.invoke(prompt)
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        # Same as original — just reused
        ub = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        lb = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound

        upper_msg = f"The question creator thinks the number is likely not higher than {ub}." if question.open_upper_bound else f"The outcome cannot be higher than {ub}."
        lower_msg = f"The question creator thinks the number is likely not lower than {lb}." if question.open_lower_bound else f"The outcome cannot be lower than {lb}."
        return upper_msg, lower_msg


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run custom forecasting bot (internal research only)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="test_questions",
        help="Run mode",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    # 🔑 KEY: Define LLMs using ONLY your specified OpenRouter models
    bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=3,  # Reduced for cost; increase if desired
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,  # Set to True when ready to submit
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        llms={
            "researcher": GeneralLlm(
                model="openrouter/openai/gpt-5",
                temperature=0.2,
                timeout=60,
                allowed_tries=2,
            ),
            "default": GeneralLlm(
                model="openrouter/anthropic/claude-sonnet-4.5",
                temperature=0.5,
                timeout=45,
                allowed_tries=2,
            ),
            "parser": GeneralLlm(
                model="openrouter/openai/gpt-4o",
                temperature=0.0,
                timeout=30,
                allowed_tries=2,
            ),
        },
    )

    if run_mode == "tournament":
        seasonal = asyncio.run(bot.forecast_on_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True))
        mini = asyncio.run(bot.forecast_on_tournament(MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True))
        reports = seasonal + mini
    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(bot.forecast_on_tournament(MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True))
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        bot.skip_previously_forecasted_questions = False
        questions = [MetaculusApi.get_question_by_url(url.strip()) for url in EXAMPLE_QUESTIONS]
        reports = asyncio.run(bot.forecast_questions(questions, return_exceptions=True))

    bot.log_report_summary(reports)
