"""
llm_client.py
-------------
Optional LLM integration. If ANTHROPIC_API_KEY (or OPENAI_API_KEY) is set
in the environment, this module can be used to turn already-computed
numeric answers into more natural, conversational sentences.

IMPORTANT: The LLM is NEVER used to calculate numbers. All figures must
be computed with pandas beforehand and passed into the prompt as facts;
the LLM's only job is to phrase them naturally. This keeps every answer
grounded in the real data even when AI text-generation is enabled.

The rest of the application works perfectly well with no API key at all
-- this module is purely an enhancement layer.
"""

from __future__ import annotations

import os
from typing import Optional


def is_llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) or bool(os.environ.get("OPENAI_API_KEY"))


def enhance_text_with_llm(computed_facts: str, instruction: str) -> Optional[str]:
    """
    Send already-computed facts to an LLM purely for natural-language
    phrasing. Returns None if no API key is configured or the call fails,
    in which case the caller should fall back to the plain computed text.
    """
    if not is_llm_available():
        return None

    prompt = (
        "You are a business analyst writing for an executive audience. "
        "Rephrase the following computed facts into 2-4 clear, natural sentences. "
        "Do NOT invent, adjust, or estimate any numbers beyond what is given.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Computed facts:\n{computed_facts}"
    )

    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in response.content if hasattr(block, "text"))

        if os.environ.get("OPENAI_API_KEY"):
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            return response.choices[0].message.content

    except Exception:
        # Any failure (missing package, network issue, bad key, etc.) simply
        # falls back to the plain, pandas-computed text -- never breaks the app.
        return None

    return None
