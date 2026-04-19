"""LangGraph tools for enhanced language model capabilities.

This package contains custom tools that can be used with LangGraph to extend
the capabilities of language models. Currently includes tools for web search
and other external integrations.
"""

from langchain_core.tools.base import BaseTool

from .duckduckgo_search import duckduckgo_search_tool
from .log_search_tool import search_logs
from .rag_search import search_incident_knowledge

tools: list[BaseTool] = [search_logs, search_incident_knowledge, duckduckgo_search_tool]
