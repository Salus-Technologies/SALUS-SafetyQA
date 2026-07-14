#!/usr/bin/env python3
"""
Benchmark evaluation script that supports both API and direct LLM evaluation.

Features:
- API mode: Evaluates questions using your API endpoint with document retrieval
- Direct mode: Evaluates questions directly against LLM providers (OpenAI, Anthropic, Google)
- Automatic batching: API endpoint automatically handles both single and batch requests
- Concurrent processing: Both modes process questions in configurable batches for efficiency
- YAML configuration: Flexible configuration with command-line overrides
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

# Load environment variables from .env file
load_dotenv()


# Enums
class EvaluationMode(str, Enum):
    API = "api"
    DIRECT = "direct"


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    API = "api"


# Models that reject a custom `temperature` (they return HTTP 400 if one is sent).
# Extend these as new families ship.
#   - OpenAI GPT-5 family / reasoning models: any model name containing "gpt-5"
#     (gpt-5, gpt-5.4, gpt-5.6 and tier variants such as gpt-5.6-sol, ...)
#   - Anthropic Opus 4.7+, Sonnet 5, Fable 5, Mythos 5
_ANTHROPIC_NO_TEMPERATURE = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)


def rejects_custom_temperature(provider: ModelProvider, model: str) -> bool:
    """Whether a provider/model pair rejects a custom ``temperature`` setting."""
    if provider == ModelProvider.OPENAI and "gpt-5" in model:
        return True
    if provider == ModelProvider.ANTHROPIC and any(m in model for m in _ANTHROPIC_NO_TEMPERATURE):
        return True
    return False


# Data Models
class MCOption(BaseModel):
    """Multiple choice option"""

    label: str = Field(..., pattern="^[a-d]$")
    text: str
    is_correct: bool


class Question(BaseModel):
    """Benchmark question"""

    id: str | None = None
    mc_question: str
    mc_options: List[MCOption]
    mc_correct_answer: str = Field(..., pattern="^[a-d]$")
    question: Optional[str] = None
    expected_answer: Optional[str] = None
    question_type: Optional[str] = None
    source_type: Optional[str] = None
    country: Optional[str] = None
    province: Optional[str] = None

    @field_validator("mc_options")
    @classmethod
    def validate_options(cls, v):
        if len(v) != 4:
            raise ValueError("Must have exactly 4 options")
        labels = [opt.label for opt in v]
        if sorted(labels) != ["a", "b", "c", "d"]:
            raise ValueError("Options must have labels a, b, c, d")
        return v


class EvaluationResult(BaseModel):
    """Result from evaluating a single question"""

    question_id: str | None = None
    question: str
    correct_answer: str
    model_answer: Optional[str] = None
    is_correct: bool = False
    confidence: float = 0.0
    reasoning: Optional[str] = None
    error: Optional[str] = None
    evaluation_mode: EvaluationMode
    model_provider: str
    model_name: str
    retrieved_documents: int = 0
    context_used: bool = False


class APIResult(BaseModel):
    """Result from API evaluation"""

    selected_answer: Optional[str] = Field(None, pattern="^[a-d]$")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    retrieved_documents: int = 0
    context_used: bool = False
    error: Optional[str] = None


class APIResponse(BaseModel):
    """API response wrapper"""

    data: Dict[str, List[APIResult]]


# Configuration Models
class APIConfig(BaseModel):
    """API configuration"""

    base_url: str = ""
    endpoint_path: str = "/evaluate"
    api_key: Optional[str] = None
    bearer_token: Optional[str] = None


class EvaluationConfig(BaseModel):
    """Evaluation configuration"""

    search_limit: int = 50
    rerank_limit: int = 20
    top_k: Optional[int] = None
    include_reasoning: bool = True
    temperature: float = 0.0
    delay_between_calls: float = 0.5
    batch_size: int = 10
    output_dir: str = "benchmark_results"


class ModelConfig(BaseModel):
    """LLM model configuration"""

    name: str
    display_name: str


class ProviderConfig(BaseModel):
    """LLM provider configuration"""

    models: List[ModelConfig]


class Config(BaseModel):
    """Complete configuration"""

    system_prompt: str
    api: APIConfig
    llm_providers: Dict[str, ProviderConfig]
    evaluation: EvaluationConfig


class BenchmarkOutput(BaseModel):
    """Structured output for benchmark evaluation (for pydantic-ai)"""

    selected_answer: str = Field(..., description="The selected option label (a, b, c, d)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the answer (0-1)")
    reasoning: str = Field(..., description="Reasoning for the selected answer")


class UnifiedBenchmarkEvaluator:
    """Unified evaluator that supports both API and direct LLM evaluation."""

    def __init__(self, config_path: str):
        """Initialize evaluator with configuration file."""
        self.config = self._load_config(config_path)
        self.results: List[EvaluationResult] = []

    def _load_config(self, config_path: str) -> Config:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Override with environment variables if available
        if not config_dict["api"].get("api_key"):
            config_dict["api"]["api_key"] = os.getenv("API_KEY")

        if not config_dict["api"].get("bearer_token"):
            config_dict["api"]["bearer_token"] = os.getenv("BEARER_TOKEN")

        # Convert to Pydantic model
        return Config(**config_dict)

    def _build_result(
        self,
        question: Question,
        model_answer: Optional[str] = None,
        confidence: float = 0.0,
        reasoning: Optional[str] = None,
        error: Optional[str] = None,
        mode: EvaluationMode = EvaluationMode.DIRECT,
        provider: str = "unknown",
        model: str = "unknown",
        retrieved_documents: int = 0,
        context_used: bool = False,
    ) -> EvaluationResult:
        """Build a standardized result."""
        is_correct = (
            model_answer == question.mc_correct_answer if model_answer and not error else False
        )

        return EvaluationResult(
            question_id=question.id,
            question=question.mc_question,
            correct_answer=question.mc_correct_answer,
            model_answer=model_answer,
            is_correct=is_correct,
            confidence=confidence,
            reasoning=reasoning if self.config.evaluation.include_reasoning else None,
            error=error,
            evaluation_mode=mode,
            model_provider=provider,
            model_name=model,
            retrieved_documents=retrieved_documents,
            context_used=context_used,
        )

    def _print_progress(self, batch_start: int, batch_size: int, total: int) -> None:
        """Print batch progress and accuracy."""
        batch_end = min(batch_start + batch_size, total)
        batch_num = batch_start // batch_size + 1
        print(f"\nProcessing batch {batch_num} ({batch_start + 1}-{batch_end} of {total})")

        if self.results:
            correct = sum(1 for r in self.results if r.is_correct)
            print(
                f"Progress: {correct}/{len(self.results)} correct ({100 * correct / len(self.results):.1f}%)"
            )

    def _print_batch_results(self, batch_results: List[EvaluationResult], batch_start: int) -> None:
        """Print individual results for a batch."""
        for j, result in enumerate(batch_results):
            question_num = batch_start + j + 1
            if result.is_correct:
                print(f"  Question {question_num}: ✓ Correct (confidence: {result.confidence:.2f})")
            elif result.error:
                print(f"  Question {question_num}: ✗ Error: {result.error}")
            else:
                print(
                    f"  Question {question_num}: ✗ Incorrect - Selected: {result.model_answer}, "
                    f"Expected: {result.correct_answer} (confidence: {result.confidence:.2f})"
                )

    async def load_questions(self, questions_file: str) -> List[Question]:
        """Load questions from JSON file."""
        path = (
            questions_file
        )
        with open(path, "r") as f:
            questions_data = json.load(f)

        # Convert to Question objects
        questions = [Question(**q) for q in questions_data]
        print(f"Loaded {len(questions)} questions from {questions_file}")
        return questions

    # API Evaluation Methods
    async def evaluate_api_batch(
        self, session: aiohttp.ClientSession, questions: List[Question]
    ) -> List[EvaluationResult]:
        """Evaluate a batch of questions via the API."""
        # Construct API URL - endpoint path can be configured if needed
        api_endpoint = self.config.api.endpoint_path
        api_url = f"{self.config.api.base_url}{api_endpoint}"

        # Build headers based on auth type
        headers = {"Content-Type": "application/json"}
        if self.config.api.api_key:
            headers["x-api-key"] = self.config.api.api_key
        elif self.config.api.bearer_token:
            headers["Authorization"] = f"Bearer {self.config.api.bearer_token}"

        # Convert questions to dict format for API
        questions_dict = [q.model_dump() for q in questions]

        payload = {
            "questions": questions_dict,
            "include_reasoning": self.config.evaluation.include_reasoning,
            "search_limit": self.config.evaluation.search_limit,
            "rerank_limit": self.config.evaluation.rerank_limit,
            "system_prompt": self.config.system_prompt,
        }

        # Add optional parameters
        if self.config.evaluation.top_k is not None:
            payload["top_k"] = self.config.evaluation.top_k

        try:
            async with session.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                    api_response = APIResponse(**response_data)

                    results = []
                    for i, question in enumerate(questions):
                        if i < len(api_response.data["results"]):
                            result = api_response.data["results"][i]
                            results.append(
                                self._build_result(
                                    question,
                                    model_answer=result.selected_answer,
                                    confidence=result.confidence,
                                    reasoning=result.reasoning,
                                    error=result.error,
                                    mode=EvaluationMode.API,
                                    provider="api",
                                    model="api-endpoint",
                                    retrieved_documents=result.retrieved_documents,
                                    context_used=result.context_used,
                                )
                            )
                    return results
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
        except Exception as e:
            print(f"API error: {e}")
            # Return error results for all questions in the batch
            return [
                self._build_result(
                    q, error=str(e), mode=EvaluationMode.API, provider="api", model="api-endpoint"
                )
                for q in questions
            ]

    # Direct LLM Evaluation Methods
    def _create_direct_agent(self, provider: ModelProvider, model: str) -> Agent:
        """Create an agent for direct LLM evaluation."""
        # Model and provider configuration
        providers_config = {
            ModelProvider.GEMINI: (GoogleModel, GoogleProvider, "GEMINI_API_KEY"),
            ModelProvider.OPENAI: (OpenAIModel, OpenAIProvider, "OPENAI_API_KEY"),
            ModelProvider.ANTHROPIC: (AnthropicModel, AnthropicProvider, "ANTHROPIC_API_KEY"),
        }

        if provider not in providers_config:
            raise ValueError(f"Unsupported provider: {provider}")

        model_class, provider_class, env_key = providers_config[provider]
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(
                f"API key required for {provider.value}. Set {env_key} environment variable"
            )

        # Create model
        llm_model = model_class(
            model_name=model,
            provider=provider_class(api_key=api_key),
        )

        # Create agent with appropriate settings.
        # Some models reject a custom temperature and 400 if one is sent
        # (OpenAI GPT-5 family; Anthropic Opus 4.7+/Sonnet 5/Fable 5). Use provider
        # defaults for those; otherwise pin the configured temperature.
        if rejects_custom_temperature(provider, model):
            model_settings = ModelSettings()  # provider default (custom temperature unsupported)
        else:
            model_settings = ModelSettings(temperature=self.config.evaluation.temperature)

        agent = Agent(
            llm_model,
            name=f"benchmark-{provider.value}",
            output_type=BenchmarkOutput,
            model_settings=model_settings,
        )

        system_prompt = self.config.system_prompt

        @agent.system_prompt
        async def get_system_prompt(ctx: RunContext) -> str:
            return system_prompt

        return agent

    async def evaluate_direct_single(
        self, agent: Agent, question: Question, provider: ModelProvider, model: str
    ) -> EvaluationResult:
        """Evaluate a single question directly using an LLM."""
        # Create user prompt
        prompt = f"Question: {question.mc_question}\n\n"
        prompt += "Options:\n"
        for opt in question.mc_options:
            prompt += f"{opt.label}) {opt.text}\n"

        prompt += "\nSelect the best answer and provide your reasoning."

        try:
            result = await agent.run(prompt)
            output: BenchmarkOutput = result.output

            return self._build_result(
                question,
                model_answer=output.selected_answer,
                confidence=output.confidence,
                reasoning=output.reasoning,
                mode=EvaluationMode.DIRECT,
                provider=provider.value,
                model=model,
            )
        except Exception as e:
            return self._build_result(
                question,
                error=str(e),
                mode=EvaluationMode.DIRECT,
                provider=provider.value,
                model=model,
            )

    # Main evaluation methods
    async def run_api_evaluation(self, questions: List[Question]) -> None:
        """Run evaluation using the API."""
        # Check for authentication
        if not self.config.api.api_key and not self.config.api.bearer_token:
            raise ValueError(
                "Authentication required for API evaluation. "
                "Set either API_KEY env var or bearer_token in config"
            )

        print("\nRunning API evaluation...")
        async with aiohttp.ClientSession() as session:
            # Process in batches
            batch_size = self.config.evaluation.batch_size
            for i in range(0, len(questions), batch_size):
                batch = questions[i : i + batch_size]
                self._print_progress(i, batch_size, len(questions))
                batch_results = await self.evaluate_api_batch(session, batch)
                self.results.extend(batch_results)

    async def run_direct_evaluation(
        self, questions: List[Question], provider: ModelProvider, model: str
    ) -> None:
        """Run direct LLM evaluation with batch processing."""
        print(f"\nRunning direct evaluation with {provider.value} - {model}...")

        agent = self._create_direct_agent(provider, model)
        batch_size = self.config.evaluation.batch_size

        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            self._print_progress(i, batch_size, len(questions))

            # Evaluate batch concurrently
            batch_results = await asyncio.gather(
                *[self.evaluate_direct_single(agent, q, provider, model) for q in batch]
            )
            self.results.extend(batch_results)

            # Print batch results
            self._print_batch_results(batch_results, i)

            # Add delay between batches
            if i + batch_size < len(questions):
                delay = self.config.evaluation.delay_between_calls
                if delay > 0:
                    print(f"Waiting {delay}s before next batch...")
                    await asyncio.sleep(delay)

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        total = len(self.results)
        if not total:
            return self._empty_metrics()

        correct = sum(1 for r in self.results if r.is_correct)
        errors = sum(1 for r in self.results if r.error is not None)
        valid_results = [r for r in self.results if r.error is None]

        # Calculate confidence metrics
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        confidences = [r.confidence for r in valid_results]
        correct_confidences = [r.confidence for r in valid_results if r.is_correct]
        incorrect_confidences = [r.confidence for r in valid_results if not r.is_correct]

        # Get model info from first result
        first = self.results[0]

        metrics = {
            "evaluation_mode": first.evaluation_mode.value,
            "model_provider": first.model_provider,
            "model_name": first.model_name,
            "model_info": f"{first.model_provider} - {first.model_name}",
            "total_questions": total,
            "correct_answers": correct,
            "incorrect_answers": total - correct - errors,
            "errors": errors,
            "accuracy": correct / total,
            "average_confidence": avg(confidences),
            "correct_answer_confidence": avg(correct_confidences),
            "incorrect_answer_confidence": avg(incorrect_confidences),
        }

        # Add API-specific metrics
        if first.evaluation_mode == EvaluationMode.API:
            api_results = [r for r in valid_results if r.retrieved_documents > 0]
            if api_results:
                metrics["documents_retrieved_avg"] = avg(
                    [r.retrieved_documents for r in api_results]
                )
                metrics["context_used_percentage"] = sum(
                    1 for r in api_results if r.context_used
                ) / len(api_results)

        return metrics

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when no results."""
        return {
            "evaluation_mode": "unknown",
            "model_provider": "unknown",
            "model_name": "unknown",
            "model_info": "unknown",
            "total_questions": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
            "errors": 0,
            "accuracy": 0.0,
            "average_confidence": 0.0,
            "correct_answer_confidence": 0.0,
            "incorrect_answer_confidence": 0.0,
        }

    def save_results(self, output_dir: Optional[str] = None) -> None:
        """Save evaluation results to files."""
        output_dir = output_dir or self.config.evaluation.output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine suffix based on evaluation mode
        if self.results:
            first = self.results[0]
            suffix = (
                "api"
                if first.evaluation_mode == EvaluationMode.API
                else f"{first.model_provider}_{first.model_name.replace('/', '_')}"
            )
        else:
            suffix = "unknown"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save all outputs
        base_name = f"benchmark_{suffix}_{timestamp}"
        # Convert results to dict for JSON serialization
        results_dict = [r.model_dump() for r in self.results]
        self._save_json(output_path / f"{base_name}_results.json", results_dict, "Detailed results")

        # Save metrics with config
        metrics = self.calculate_metrics()
        metrics["timestamp"] = timestamp
        metrics["config_used"] = self._get_config_summary()
        self._save_json(output_path / f"{base_name}_metrics.json", metrics, "Metrics")

        # Save summary report
        self._save_summary_report(output_path, base_name, metrics)

    def _save_json(self, path: Path, data: Any, description: str) -> None:
        """Helper to save JSON files."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)  # default=str handles enums
        print(f"{description} saved to: {path}")

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for metrics."""
        config = {
            "system_prompt": self.config.system_prompt[:100] + "...",
            "search_limit": self.config.evaluation.search_limit,
            "rerank_limit": self.config.evaluation.rerank_limit,
            "temperature": self.config.evaluation.temperature,
        }
        if self.config.evaluation.top_k is not None:
            config["top_k"] = self.config.evaluation.top_k
        return config

    def _save_summary_report(
        self, output_path: Path, base_name: str, metrics: Dict[str, Any]
    ) -> None:
        """Save human-readable summary report."""
        summary_file = output_path / f"{base_name}_summary.txt"

        with open(summary_file, "w") as f:
            f.write("BENCHMARK EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluation Mode: {metrics['evaluation_mode']}\n")
            f.write(f"Model: {metrics['model_info']}\n")
            f.write(f"Total Questions: {metrics['total_questions']}\n\n")

            f.write("OVERALL RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Correct Answers: {metrics['correct_answers']} / {metrics['total_questions']}\n"
            )
            f.write(f"Accuracy: {metrics['accuracy']:.2%}\n")
            f.write(f"Errors: {metrics['errors']}\n\n")

            f.write("CONFIDENCE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Confidence: {metrics['average_confidence']:.3f}\n")
            f.write(f"Confidence on Correct: {metrics['correct_answer_confidence']:.3f}\n")
            f.write(f"Confidence on Incorrect: {metrics['incorrect_answer_confidence']:.3f}\n\n")

            if "documents_retrieved_avg" in metrics:
                f.write("RETRIEVAL STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Avg Documents Retrieved: {metrics['documents_retrieved_avg']:.1f}\n")
                f.write(f"Context Used: {metrics['context_used_percentage']:.1%}\n\n")

            # Add error details if any
            errors = [r for r in self.results if r.error is not None]
            if errors:
                f.write("ERRORS ENCOUNTERED\n")
                f.write("-" * 30 + "\n")
                for r in errors:
                    f.write(f"\nQuestion ID: {r.question_id}\n")
                    f.write(f"Error: {r.error}\n")
                f.write("\n")

            # Add examples of incorrect answers
            f.write("SAMPLE INCORRECT ANSWERS\n")
            f.write("-" * 30 + "\n")
            incorrect = [r for r in self.results if not r.is_correct and r.error is None]
            for r in incorrect[:5]:  # Show first 5
                f.write(f"\nQuestion ID: {r.question_id}\n")
                f.write(f"Question: {r.question[:100]}...\n")
                f.write(f"Correct Answer: {r.correct_answer}\n")
                f.write(f"Model Answer: {r.model_answer}\n")
                f.write(f"Confidence: {r.confidence:.3f}\n")
                if r.reasoning and self.config.evaluation.include_reasoning:
                    f.write(f"Reasoning: {r.reasoning[:200]}...\n")

        print(f"Summary report saved to: {summary_file}")

        # Print summary to console
        print("\n" + "=" * 50)
        print("EVALUATION COMPLETE")
        print("=" * 50)
        print(f"Mode: {metrics['evaluation_mode']}")
        print(f"Model: {metrics['model_info']}")
        print(
            f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct_answers']}/{metrics['total_questions']})"
        )
        print(f"Average Confidence: {metrics['average_confidence']:.3f}")
        print(f"Errors: {metrics['errors']}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified benchmark evaluation script supporting API and direct LLM modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # API evaluation
  %(prog)s --mode api --questions questions.json

  # Direct evaluation
  %(prog)s --mode direct --questions questions.json --provider gemini --model gemini-2.0-flash-latest

  # With custom config and overrides
  %(prog)s --config my_config.yaml --mode api --questions questions.json --top-k 20
        """,
    )

    # Core arguments
    parser.add_argument("--config", default="benchmark_config.yaml", help="Configuration YAML file")
    parser.add_argument("--mode", required=True, choices=["api", "direct"], help="Evaluation mode")
    parser.add_argument("--questions", required=True, help="Questions JSON file")

    # Direct mode arguments
    parser.add_argument(
        "--provider", choices=["gemini", "openai", "anthropic"], help="LLM provider for direct mode"
    )
    parser.add_argument("--model", help="Model name for direct mode")

    # Optional overrides
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--no-reasoning", action="store_true", help="Skip reasoning in responses")
    parser.add_argument("--batch-size", type=int, help="Batch size for concurrent requests")
    parser.add_argument("--delay", type=float, help="Delay between batches (direct mode)")
    parser.add_argument("--top-k", type=int, help="Documents for reranking")

    return parser


def apply_config_overrides(evaluator: UnifiedBenchmarkEvaluator, args: argparse.Namespace) -> None:
    """Apply command-line overrides to evaluator config."""
    if args.no_reasoning:
        evaluator.config.evaluation.include_reasoning = False
    if args.batch_size:
        evaluator.config.evaluation.batch_size = args.batch_size
    if args.delay:
        evaluator.config.evaluation.delay_between_calls = args.delay
    if args.top_k:
        evaluator.config.evaluation.top_k = args.top_k


async def main():
    parser = create_parser()

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "direct" and (not args.provider or not args.model):
        parser.error("--provider and --model are required for direct mode")

    # Initialize and configure evaluator
    evaluator = UnifiedBenchmarkEvaluator(args.config)
    apply_config_overrides(evaluator, args)

    # Load questions and run evaluation
    questions = await evaluator.load_questions(args.questions)

    if args.mode == "api":
        await evaluator.run_api_evaluation(questions)
    else:
        # Convert provider string to enum
        provider = ModelProvider(args.provider)
        await evaluator.run_direct_evaluation(questions, provider, args.model)

    # Save results
    evaluator.save_results(args.output_dir)


def cli_main():
    """CLI entry point for the evaluation script."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
