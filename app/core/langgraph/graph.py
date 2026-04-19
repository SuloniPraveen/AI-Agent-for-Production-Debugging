"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

import asyncio
import inspect
import json
import re
from typing import (
    Any,
    AsyncGenerator,
    List,
    Optional,
)
from uuid import UUID
from urllib.parse import quote_plus

from asgiref.sync import sync_to_async
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import (
    END,
    StateGraph,
)
from langgraph.graph.state import (
    Command,
    CompiledStateGraph,
)
from langgraph.types import (
    RunnableConfig,
    StateSnapshot,
)
from mem0 import AsyncMemory
from psycopg_pool import AsyncConnectionPool

from app.core.config import (
    Environment,
    settings,
)
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.metrics import llm_latency_seconds
from app.core.prompts import load_system_prompt
from app.schemas import (
    GraphState,
    Message,
)
from app.schemas.chat import LogCitation
from app.services.llm import llm_service
from app.services.log_batch_scope import resolve_log_search_batch_ids
from app.services.log_search import log_search_service
from app.services.log_search_scope import (
    reset_log_search_batch_ids,
    set_log_search_batch_ids,
)
from app.utils import (
    dump_messages,
    prepare_messages,
    process_llm_response,
)
from app.utils.graph import content_blocks_to_plain_text

# If the model skips search_logs, still populate API citations for incident-style questions (demo UX).
_LOG_QUERY_FALLBACK = re.compile(
    r"\b(503|502|504|500|error|errors|timeout|deploy|checkout|logs?\b|log lines|ingested|incident|"
    r"production|evidence|database|payment|redis|gateway|outage|fail(?:ed|ure)?|unhealthy)\b",
    re.I,
)

# One retrieval pass per user message (top chunks already returned in tool JSON).
_MAX_SEARCH_LOGS_PER_USER_MESSAGE = 1


async def _fallback_citations_from_user_query(user_text: str) -> List[LogCitation]:
    """Run log vector search once when the agent did not call search_logs."""
    out: List[LogCitation] = []
    q = user_text.strip()[:3000]
    if not q:
        return out
    try:
        raw = await log_search_service.search_json(q, top_k=settings.LOG_SEARCH_DEFAULT_TOP_K)
        data = json.loads(raw)
    except Exception as e:
        logger.warning("log_citation_fallback_failed", error=str(e))
        return out
    for c in data.get("citations") or []:
        if not isinstance(c, dict):
            continue
        try:
            out.append(LogCitation(**c))
        except Exception:
            logger.warning("log_citation_fallback_skip_malformed", exc_info=True)
    return out


def _messages_since_last_human(raw_messages: List[BaseMessage]) -> List[BaseMessage]:
    """Checkpoint `messages` is the full thread; only the tail is this user turn."""
    last = -1
    for i, m in enumerate(raw_messages):
        if isinstance(m, HumanMessage):
            last = i
    if last < 0:
        return raw_messages
    return raw_messages[last:]


def _dedupe_log_citations(citations: List[LogCitation]) -> List[LogCitation]:
    """Keep first occurrence per chunk_id (agent may call search_logs multiple times)."""
    seen: set[int] = set()
    out: List[LogCitation] = []
    for c in citations:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out


def extract_log_citations_from_messages(raw_messages: List[BaseMessage]) -> List[LogCitation]:
    """Parse search_logs ToolMessage payloads into structured citations."""
    out: List[LogCitation] = []
    for m in raw_messages:
        if not isinstance(m, ToolMessage):
            continue
        if (getattr(m, "name", None) or "") != "search_logs":
            continue
        content = m.content
        if not isinstance(content, str):
            content = str(content)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue
        for c in data.get("citations") or []:
            if not isinstance(c, dict):
                continue
            try:
                out.append(LogCitation(**c))
            except Exception:
                logger.warning("skip_malformed_citation", exc_info=True)
    return out


def _search_logs_tool_was_used(raw_messages: List[BaseMessage]) -> bool:
    """True if the agent called search_logs this turn (even when it returned zero rows)."""
    for m in raw_messages:
        if isinstance(m, ToolMessage) and (getattr(m, "name", None) or "") == "search_logs":
            return True
    return False


def _openai_content_to_text(content: Any) -> str:
    """Turn LangChain / OpenAI message content into plain text for API responses."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content_blocks_to_plain_text(content)
    if isinstance(content, dict):
        return content_blocks_to_plain_text(content)
    return str(content)


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use the LLM service with tools bound
        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        self.memory: Optional[AsyncMemory] = None
        logger.info(
            "langgraph_agent_initialized",
            model=settings.DEFAULT_LLM_MODEL,
            environment=settings.ENVIRONMENT.value,
        )

    async def _long_term_memory(self) -> AsyncMemory:
        """Initialize the long term memory."""
        if self.memory is None:
            created = AsyncMemory.from_config(
                config_dict={
                    "vector_store": {
                        "provider": "pgvector",
                        "config": {
                            "collection_name": settings.LONG_TERM_MEMORY_COLLECTION_NAME,
                            "dbname": settings.POSTGRES_DB,
                            "user": settings.POSTGRES_USER,
                            "password": settings.POSTGRES_PASSWORD,
                            "host": settings.POSTGRES_HOST,
                            "port": settings.POSTGRES_PORT,
                        },
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {"model": settings.LONG_TERM_MEMORY_MODEL},
                    },
                    "embedder": {"provider": "openai", "config": {"model": settings.LONG_TERM_MEMORY_EMBEDDER_MODEL}},
                    # "custom_fact_extraction_prompt": load_custom_fact_extraction_prompt(),
                }
            )
            self.memory = await created if inspect.isawaitable(created) else created
        return self.memory

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                connection_url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )

                self._connection_pool = AsyncConnectionPool(
                    connection_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we might want to degrade gracefully
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool

    async def _get_relevant_memory(self, user_id: str, query: str) -> str:
        """Get the relevant memory for the user and query.

        Args:
            user_id (str): The user ID.
            query (str): The query to search for.

        Returns:
            str: The relevant memory.
        """
        if not settings.LONG_TERM_MEMORY_ENABLED:
            return ""
        try:
            memory = await self._long_term_memory()
            results = await asyncio.wait_for(
                memory.search(user_id=str(user_id), query=query),
                timeout=settings.LONG_TERM_MEMORY_RETRIEVE_TIMEOUT_SECONDS,
            )
            rows = results.get("results", []) if isinstance(results, dict) else []
            return "\n".join(f"* {r.get('memory', r)}" for r in rows)
        except asyncio.TimeoutError:
            logger.warning(
                "long_term_memory_retrieve_timeout",
                user_id=user_id,
                timeout_sec=settings.LONG_TERM_MEMORY_RETRIEVE_TIMEOUT_SECONDS,
            )
            return ""
        except Exception as e:
            logger.error("failed_to_get_relevant_memory", error=str(e), user_id=user_id, query=query)
            return ""

    async def _update_long_term_memory(self, user_id: str, messages: list[dict], metadata: dict = None) -> None:
        """Update the long term memory.

        Args:
            user_id (str): The user ID.
            messages (list[dict]): The messages to update the long term memory with.
            metadata (dict): Optional metadata to include.
        """
        if not settings.LONG_TERM_MEMORY_ENABLED:
            return
        try:
            memory = await self._long_term_memory()
            await asyncio.wait_for(
                memory.add(messages, user_id=str(user_id), metadata=metadata),
                timeout=max(45.0, settings.LONG_TERM_MEMORY_RETRIEVE_TIMEOUT_SECONDS * 4),
            )
            logger.info("long_term_memory_updated_successfully", user_id=user_id)
        except asyncio.TimeoutError:
            logger.warning("long_term_memory_update_timeout", user_id=user_id)
        except Exception as e:
            logger.exception(
                "failed_to_update_long_term_memory",
                user_id=user_id,
                error=str(e),
            )

    async def _chat(self, state: GraphState, config: RunnableConfig) -> Command:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.

        Returns:
            Command: Command object with updated state and next node to execute.
        """
        # Get the current LLM instance for metrics
        current_llm = self.llm_service.get_llm()
        model_name = (
            current_llm.model_name
            if current_llm and hasattr(current_llm, "model_name")
            else settings.DEFAULT_LLM_MODEL
        )

        SYSTEM_PROMPT = load_system_prompt(long_term_memory=state.long_term_memory)

        # Prepare messages with system prompt
        messages = prepare_messages(state.messages, current_llm, SYSTEM_PROMPT)

        try:
            # Use LLM service with automatic retries and circular fallback
            with llm_latency_seconds.labels(model=model_name).time():
                response_message = await self.llm_service.call(dump_messages(messages))

            # Process response to handle structured content blocks
            response_message = process_llm_response(response_message)

            logger.info(
                "llm_response_generated",
                session_id=config["configurable"]["thread_id"],
                model=model_name,
                environment=settings.ENVIRONMENT.value,
            )

            # Determine next node based on whether there are tool calls
            if response_message.tool_calls:
                goto = "tool_call"
            else:
                goto = END

            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception as e:
            logger.error(
                "llm_call_failed_all_models",
                session_id=config["configurable"]["thread_id"],
                error=str(e),
                environment=settings.ENVIRONMENT.value,
            )
            raise Exception(f"failed to get llm response after trying all models: {str(e)}")

    # Define our tool node
    async def _tool_call(self, state: GraphState) -> Command:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Command: Command object with updated messages and routing back to chat.
        """
        turn_so_far = _messages_since_last_human(state.messages)
        prior_search_logs = sum(
            1
            for m in turn_so_far
            if isinstance(m, ToolMessage) and (getattr(m, "name", None) or "") == "search_logs"
        )
        search_logs_in_batch = 0
        outputs = []
        for tool_call in state.messages[-1].tool_calls:
            name = tool_call["name"]
            if name == "search_logs" and prior_search_logs + search_logs_in_batch >= _MAX_SEARCH_LOGS_PER_USER_MESSAGE:
                outputs.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "citations": [],
                                "context": (
                                    f"Maximum {_MAX_SEARCH_LOGS_PER_USER_MESSAGE} search_logs calls per user "
                                    "message. Answer using the results you already have."
                                ),
                                "error": "search_logs_limit",
                            }
                        ),
                        name="search_logs",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue
            if name == "search_logs":
                search_logs_in_batch += 1
            tool_result = await self.tools_by_name[name].ainvoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=tool_result,
                    name=name,
                    tool_call_id=tool_call["id"],
                )
            )
        return Command(update={"messages": outputs}, goto="chat")

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """
        if self._graph is None:
            try:
                graph_builder = StateGraph(GraphState)
                graph_builder.add_node("chat", self._chat, ends=["tool_call", END])
                graph_builder.add_node("tool_call", self._tool_call, ends=["chat"])
                graph_builder.set_entry_point("chat")
                graph_builder.set_finish_point("chat")

                # Get connection pool (may be None in production if DB unavailable)
                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                else:
                    # In production, proceed without checkpointer if needed
                    checkpointer = None
                    if settings.ENVIRONMENT != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
                )

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we don't want to crash the app
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

        return self._graph

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        focus_log_batch_id: Optional[UUID] = None,
    ) -> tuple[list[Message], list[LogCitation]]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for Langfuse tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.

        Returns:
            Tuple of **this-turn** messages for the API (latest user from the request + latest
            assistant text) and log citations. Full thread stays in the checkpoint; we do not
            re-send the entire history (avoids empty tool-only assistant turns and duplicate user lines).

        Raises:
            Exception: Re-raises after logging; callers must not receive None.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
            "recursion_limit": settings.LANGGRAPH_RECURSION_LIMIT,
        }
        relevant_memory = (
            await self._get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."
        bids = (
            resolve_log_search_batch_ids(int(user_id), focus_log_batch_id)
            if user_id is not None
            else []
        )
        scope_token = set_log_search_batch_ids(tuple(str(b) for b in bids))
        try:
            response = await self._graph.ainvoke(
                input={"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                config=config,
            )
            raw_messages = response.get("messages")
            if raw_messages is None:
                logger.error("graph_ainvoke_missing_messages", session_id=session_id)
                raise RuntimeError("Agent returned no messages")

            processed = self.__process_messages(raw_messages)
            turn_messages = _messages_since_last_human(raw_messages)
            citations = _dedupe_log_citations(extract_log_citations_from_messages(turn_messages))
            # Do not add unfiltered fallback citations when search_logs already ran: the model's
            # answer reflects the filtered tool result; extra chunks would contradict "no evidence".
            if (
                not citations
                and messages
                and messages[-1].role == "user"
                and _LOG_QUERY_FALLBACK.search(messages[-1].content or "")
                and not _search_logs_tool_was_used(turn_messages)
            ):
                citations = await _fallback_citations_from_user_query(messages[-1].content or "")

            # Expose one completion to clients: last user in this request + last assistant with text.
            # (Graph state includes every past user turn; tool-only assistant turns have empty content.)
            assistants = [m for m in processed if m.role == "assistant" and (m.content or "").strip()]
            latest_a = assistants[-1] if assistants else None
            if latest_a and messages and messages[-1].role == "user":
                processed = [messages[-1], latest_a]
            elif latest_a:
                processed = [latest_a]
            else:
                processed = []

            # Only update long-term memory after we successfully built the API response
            asyncio.create_task(
                self._update_long_term_memory(
                    user_id, convert_to_openai_messages(raw_messages), config["metadata"]
                )
            )
            return processed, citations
        except Exception as e:
            logger.exception("Error getting response", error=str(e), session_id=session_id)
            raise
        finally:
            reset_log_search_batch_ids(scope_token)

    async def get_stream_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        focus_log_batch_id: Optional[UUID] = None,
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response.
        """
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler(
                    environment=settings.ENVIRONMENT.value, debug=False, user_id=user_id, session_id=session_id
                )
            ],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
            "recursion_limit": settings.LANGGRAPH_RECURSION_LIMIT,
        }
        if self._graph is None:
            self._graph = await self.create_graph()

        relevant_memory = (
            await self._get_relevant_memory(user_id, messages[-1].content)
        ) or "No relevant memory found."

        _bids = (
            resolve_log_search_batch_ids(int(user_id), focus_log_batch_id)
            if user_id is not None
            else []
        )
        scope_token = set_log_search_batch_ids(tuple(str(b) for b in _bids))
        try:
            try:
                async for token, _ in self._graph.astream(
                    {"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                    config,
                    stream_mode="messages",
                ):
                    try:
                        raw = getattr(token, "content", None)
                        if raw:
                            yield _openai_content_to_text(raw)
                    except Exception as token_error:
                        logger.error("Error processing token", error=str(token_error), session_id=session_id)
                        # Continue with next token even if current one fails
                        continue

                # After streaming completes, get final state and update memory in background
                state: StateSnapshot = await sync_to_async(self._graph.get_state)(config=config)
                if state.values and "messages" in state.values:
                    asyncio.create_task(
                        self._update_long_term_memory(
                            user_id, convert_to_openai_messages(state.values["messages"]), config["metadata"]
                        )
                    )
            except Exception as stream_error:
                logger.error("Error in stream processing", error=str(stream_error), session_id=session_id)
                raise stream_error
        finally:
            reset_log_search_batch_ids(scope_token)

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        if self._graph is None:
            self._graph = await self.create_graph()

        state: StateSnapshot = await sync_to_async(self._graph.get_state)(
            config={"configurable": {"thread_id": session_id}}
        )
        if not state.values:
            return []
        raw = state.values.get("messages")
        if not raw:
            return []
        return self.__process_messages(raw)

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        out: list[Message] = []
        for message in openai_style_messages:
            role = message.get("role") if isinstance(message, dict) else None
            if role not in ("assistant", "user"):
                continue
            text = _openai_content_to_text(message.get("content") if isinstance(message, dict) else None).strip()
            if not text:
                continue
            try:
                out.append(Message(role=role, content=text))
            except Exception:
                logger.warning("skip_invalid_chat_message", role=role, exc_info=True)
        return out

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()

            # Use a new connection for this specific operation
            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
                        logger.info(f"Cleared {table} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error clearing {table}", error=str(e))
                        raise

        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e))
            raise
