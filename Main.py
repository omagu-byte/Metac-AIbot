import argparse
import asyncio
import logging
import os
from typing import List
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

load_dotenv()

logger = logging.getLogger("PreMortemOpenRouterBot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# -------------------------------------------------------------------
# OpenRouter-only LLM client
# -------------------------------------------------------------------
import aiohttp

class OpenRouterLlm:
    """
    Simple async client that routes ALL models through OpenRouter.
    """
    def __init__(self, model: str, api_key: str):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        self.model = model  # e.g. "openrouter/openai/gpt-4o"
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
    """
    Forecast bot that forces all LLM calls through OpenRouter
    while keeping the standard ForecastBot architecture.
    """
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        # Default model set (can be overridden from CLI)
        self.models = {
            "narrative_primary": "openrouter/gpt-o3",
            "narrative_secondary": "openrouter/gpt-5",
            "analytical_synthesis": "openrouter/openai/gpt-4o",
            "analytical_judgment": "openrouter/gpt-5",
        }

    def llm_factory(self, stage: str) -> OpenRouterLlm:
        """Return an OpenRouter LLM for a given forecasting stage."""
        return OpenRouterLlm(self.models[stage], self.api_key)

    async def forecast_question(self, question: MetaculusQuestion) -> str:
        """
        Example forecast routine using a single OpenRouter call.
        Extend with your research + synthesis logic as needed.
        """
        llm = self.llm_factory("analytical_judgment")
        prompt = f"Provide a reasoned forecast for:\n{question.title}\n{question.url}"
        answer = await llm.invoke(prompt, temperature=0.4)
        logger.info(f"[Forecast] {question.id} => {answer[:120]}...")
        return answer

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
        help=(
            "Metaculus tournament IDs "
            f"(default: 32813 and current minibench {MetaculusApi.CURRENT_MINIBENCH_ID})"
        ),
    )
    parser.add_argument(
        "--models",
        nargs=4,
        metavar=("NARR_PRIMARY", "NARR_SECONDARY", "SYNTHESIS", "JUDGMENT"),
        help="Override models for the four stages (OpenRouter format).",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY must be set in your environment.")

    bot = PreMortemOpenRouterBot(
        api_key=api_key,
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
    )

    if args.models:
        bot.models.update(
            {
                "narrative_primary": args.models[0],
                "narrative_secondary": args.models[1],
                "analytical_synthesis": args.models[2],
                "analytical_judgment": args.models[3],
            }
        )
        logger.info(f"Custom model config: {bot.models}")

    # Forecast all tournaments
    for tid in args.tournament_ids:
        logger.info(f"--- Forecasting tournament {tid} ---")
        await bot.forecast_on_tournament(tournament_id=int(tid), return_exceptions=True)

if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    asyncio.run(main())
