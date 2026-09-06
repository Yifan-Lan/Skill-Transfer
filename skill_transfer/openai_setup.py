"""
Configure the OpenAI client for either standard OpenAI or Azure OpenAI.

Azure mode is activated automatically when AZURE_OPENAI_ENDPOINT is set.
Call setup_openai() once at the start of each entry script that uses the
OpenAI Agents SDK (Runner.run / Runner.run_streamed).

Standard OpenAI (default):
    Set OPENAI_API_KEY and call setup_openai() — nothing else needed.

Azure OpenAI:
    Set:
        AZURE_OPENAI_ENDPOINT    e.g. https://<resource>.openai.azure.com
        AZURE_OPENAI_API_KEY     your Azure key
    Then pass the deployment name as --model / --weak-model / etc. in the
    shell scripts (the deployment name IS the model identifier in Azure mode).
"""

import os
import sys

_DIM    = "\033[90m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

_PRINTED: set[str] = set()   # avoid duplicate banners when a script imports multiple modules


def setup_openai(label: str = "") -> None:
    """Configure the OpenAI Agents SDK for Azure or standard OpenAI.

    Prints a one-line banner to stderr so the active backend is always visible.
    No-op (except the banner) when AZURE_OPENAI_ENDPOINT is not set.

    Args:
        label: Optional caller name shown in the banner, e.g. "skill-agent".
    """
    tag = f"[{label}] " if label else ""

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()

    if not endpoint:
        # Standard OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print(
                f"{_YELLOW}WARNING:{_RESET} {tag}OPENAI_API_KEY is not set — API calls will fail.",
                file=sys.stderr,
            )
        else:
            key_hint = api_key[:8] + "..." + api_key[-4:]
            banner_key = ("openai", label)
            if banner_key not in _PRINTED:
                _PRINTED.add(banner_key)
                print(
                    f"{_DIM}[openai]{_RESET} {tag}{_GREEN}Standard OpenAI{_RESET}"
                    f"  key={_DIM}{key_hint}{_RESET}",
                    file=sys.stderr,
                )
        return

    # Azure OpenAI
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            f"{tag}Azure mode requires AZURE_OPENAI_API_KEY (or OPENAI_API_KEY) to be set."
        )

    from openai import AsyncOpenAI
    from agents import set_default_openai_client, set_tracing_disabled

    base_url = endpoint.rstrip("/") + "/openai/v1/"
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    set_default_openai_client(client, use_for_tracing=False)
    set_tracing_disabled(True)

    key_hint = api_key[:8] + "..." + api_key[-4:]
    banner_key = ("azure", label)
    if banner_key not in _PRINTED:
        _PRINTED.add(banner_key)
        print(
            f"{_DIM}[openai]{_RESET} {tag}{_CYAN}{_BOLD}Azure OpenAI{_RESET}"
            f"  endpoint={_CYAN}{endpoint}{_RESET}"
            f"  key={_DIM}{key_hint}{_RESET}",
            file=sys.stderr,
        )
