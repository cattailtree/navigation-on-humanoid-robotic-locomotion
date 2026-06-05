from __future__ import annotations

import argparse
import os

from llm.task_parser import HttpNavigationTaskParser, NavigationTaskParser, RuleBasedTaskParser, unload_ollama_model


TASK_PARSER_CHOICES = ("rule", "llm_http", "openai_compatible")
DEFAULT_TASK_ENDPOINT = "http://127.0.0.1:12182/parse_task"
DEFAULT_OPENAI_COMPATIBLE_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"


def add_task_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task-parser",
        choices=TASK_PARSER_CHOICES,
        default="rule",
        help="Task parser for natural-language navigation goals.",
    )
    parser.add_argument("--llm-endpoint", default=None, help="External LLM parser endpoint.")
    parser.add_argument("--llm-model", default=None, help="Model name for OpenAI-compatible chat-completion endpoints.")
    parser.add_argument(
        "--llm-api-key-env",
        default="SEMANTIC_NAV_LLM_API_KEY",
        help="Environment variable containing the API key for the LLM endpoint, if needed.",
    )
    parser.add_argument("--llm-timeout-s", type=float, default=20.0, help="Timeout for LLM task parsing requests.")
    parser.add_argument("--log-llm", action="store_true", help="Print raw LLM parser responses.")


def make_task_parser(
    kind: str,
    *,
    endpoint: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    timeout_s: float = 20.0,
    log_raw: bool = False,
) -> NavigationTaskParser:
    if kind == "rule":
        return RuleBasedTaskParser()
    if kind == "llm_http":
        return HttpNavigationTaskParser(
            endpoint or DEFAULT_TASK_ENDPOINT,
            mode="task_endpoint",
            model=model,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
            log_raw=log_raw,
        )
    if kind == "openai_compatible":
        return HttpNavigationTaskParser(
            _chat_completions_endpoint(endpoint or os.environ.get("OPENAI_BASE_URL")),
            mode="openai_chat",
            model=model or os.environ.get("SEMANTIC_NAV_LLM_MODEL"),
            api_key_env=api_key_env,
            timeout_s=timeout_s,
            log_raw=log_raw,
        )
    raise ValueError(f"Unknown task parser kind: {kind}")


def _chat_completions_endpoint(endpoint: str | None) -> str:
    if endpoint is None:
        return DEFAULT_OPENAI_COMPATIBLE_ENDPOINT
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def make_task_parser_from_args(args: argparse.Namespace) -> NavigationTaskParser:
    return make_task_parser(
        args.task_parser,
        endpoint=args.llm_endpoint,
        model=args.llm_model,
        api_key_env=args.llm_api_key_env,
        timeout_s=args.llm_timeout_s,
        log_raw=args.log_llm,
    )


def release_task_parser_resources_from_args(args: argparse.Namespace) -> None:
    if getattr(args, "task_parser", None) != "openai_compatible":
        return
    endpoint = _chat_completions_endpoint(getattr(args, "llm_endpoint", None) or os.environ.get("OPENAI_BASE_URL"))
    model = getattr(args, "llm_model", None) or os.environ.get("SEMANTIC_NAV_LLM_MODEL")
    unload_ollama_model(endpoint, model, timeout_s=2.0, log_raw=getattr(args, "log_llm", False))


def normalize_target_node_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() in {"auto", "llm", "none", "null"}:
        return None
    return text
