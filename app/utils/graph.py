"""This file contains the graph utilities for the application."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages import trim_messages as _trim_messages

from app.core.config import settings
from app.core.logging import logger
from app.schemas import Message


def content_blocks_to_plain_text(content: Any) -> str:
    """Flatten OpenAI Responses / GPT-5 style content (list of blocks) to user-visible text.

    `process_llm_response` used to only keep ``type=="text"`` blocks; models often return
    ``reasoning``-only turns or ``output_text`` / nested ``summary`` payloads, which produced
    empty strings and caused the API to drop assistant messages entirely.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    def from_dict(block: dict) -> str:
        if not isinstance(block, dict):
            return ""
        btype = block.get("type")
        if btype in ("text", "output_text") and block.get("text") is not None:
            return str(block["text"])
        if isinstance(block.get("text"), str) and block["text"].strip():
            return block["text"]
        if isinstance(block.get("content"), str) and block["content"].strip():
            return block["content"]
        if btype == "refusal" and block.get("refusal"):
            return str(block["refusal"])
        if btype == "reasoning":
            summary = block.get("summary")
            if isinstance(summary, list):
                return "\n".join(
                    content_blocks_to_plain_text(s) if isinstance(s, (list, dict)) else (str(s) if s else "")
                    for s in summary
                ).strip()
            if isinstance(summary, str) and summary.strip():
                return summary
            if isinstance(summary, dict):
                return from_dict(summary)
        return ""

    if isinstance(content, dict):
        return from_dict(content)

    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                parts.append(item)
        elif isinstance(item, dict):
            chunk = from_dict(item)
            if chunk:
                parts.append(chunk)
        elif item is not None:
            parts.append(str(item))

    return "\n".join(parts)


def _fallback_text_from_blocks(blocks: list) -> str:
    """Last resort: collect user-visible strings from nested OpenAI / Responses-style blocks."""
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        elif isinstance(x, dict):
            for key in ("text", "output_text", "content", "message", "refusal"):
                v = x.get(key)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
            for v in x.values():
                if isinstance(v, (list, dict)):
                    walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(blocks)
    # Dedupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return "\n".join(uniq)


def dump_messages(messages: list[Message]) -> list[dict]:
    """Dump the messages to a list of dictionaries.

    Args:
        messages (list[Message]): The messages to dump.

    Returns:
        list[dict]: The dumped messages.
    """
    return [message.model_dump() for message in messages]


def process_llm_response(response: BaseMessage) -> BaseMessage:
    """Process LLM response to handle structured content blocks (e.g., from GPT-5 models).

    GPT-5 models return content as a list of blocks like:
    [
        {'id': '...', 'summary': [], 'type': 'reasoning'},
        {'type': 'text', 'text': 'actual response'}
    ]

    This function extracts the actual text content from such structures.

    Args:
        response: The raw response from the LLM

    Returns:
        BaseMessage with processed content
    """
    if isinstance(response.content, list):
        raw_blocks = response.content
        for block in raw_blocks:
            if isinstance(block, dict) and block.get("type") == "reasoning":
                logger.debug(
                    "reasoning_block_received",
                    reasoning_id=block.get("id"),
                    has_summary=bool(block.get("summary")),
                )
        flat = content_blocks_to_plain_text(raw_blocks)
        if not flat.strip():
            flat = _fallback_text_from_blocks(raw_blocks)
        response.content = flat
        logger.debug(
            "processed_structured_content",
            block_count=len(raw_blocks),
            extracted_length=len(flat),
        )

    return response


def prepare_messages(messages: list[Message], llm: BaseChatModel, system_prompt: str) -> list[Message]:
    """Prepare the messages for the LLM.

    Args:
        messages (list[Message]): The messages to prepare.
        llm (BaseChatModel): The LLM to use.
        system_prompt (str): The system prompt to use.

    Returns:
        list[Message]: The prepared messages.
    """
    try:
        trimmed_messages = _trim_messages(
            dump_messages(messages),
            strategy="last",
            token_counter=llm,
            max_tokens=settings.CHAT_HISTORY_TRIM_MAX_TOKENS,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
    except ValueError as e:
        # Handle unrecognized content blocks (e.g., reasoning blocks from GPT-5)
        if "Unrecognized content block type" in str(e):
            logger.warning(
                "token_counting_failed_skipping_trim",
                error=str(e),
                message_count=len(messages),
            )
            # Skip trimming and return all messages
            trimmed_messages = messages
        else:
            raise

    return [Message(role="system", content=system_prompt)] + trimmed_messages
