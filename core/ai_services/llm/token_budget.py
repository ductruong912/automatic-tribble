import logging
from typing import List, Dict, Any, Tuple
from config.settings import Config

logger = logging.getLogger(__name__)

class TokenBudgetManager:
    """
    Centralized token budgeting system for LLM requests.
    
    WARNING: TOKENIZER APPROXIMATION
    This implementation uses tiktoken's 'cl100k_base' encoding as an ESTIMATOR only.
    - It is exact/appropriate for some OpenAI models (like GPT-4o-mini).
    - It is an APPROXIMATION for non-OpenAI models such as Qwen/Llama (which use different vocabularies).
    
    Because of this approximation, the budget calculates an `effective_context_window`
    by shrinking the raw CONTEXT_WINDOW by the CONTEXT_SAFETY_MARGIN (e.g. 10%).
    This ensures that even if local models consume slightly more tokens than tiktoken predicts,
    the request will not overflow the model's hard context limit (e.g., n_ctx=8192).
    """

    # Chat message structural overhead per message (e.g. role, name framing)
    MESSAGE_OVERHEAD = 4
    # Request-level overhead (e.g. base format framing)
    REQUEST_OVERHEAD = 3

    def __init__(self):
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.error("tiktoken is not installed. Token estimation will fail.")
            raise
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Estimate token count for a given text string."""
        if not text:
            return 0
        return len(self.encoding.encode(text, disallowed_special=()))

    def count_message_tokens(self, message: Dict[str, str]) -> int:
        """Estimate token count for a formatted chat message, including overhead."""
        count = self.MESSAGE_OVERHEAD
        for key, value in message.items():
            if isinstance(value, str):
                count += self.count_tokens(value)
        return count

    def get_effective_context_window(self) -> int:
        """Return the target server's absolute context window, reduced by the safety margin."""
        raw_window = Config.LLM.CONTEXT_WINDOW()
        margin = Config.LLM.CONTEXT_SAFETY_MARGIN()
        return int(raw_window * (1.0 - margin))

    def build_prompt_and_budget(
        self, 
        base_system_prompt: str, 
        query: str, 
        search_results: List[Dict[str, Any]], 
        history: List[Dict[str, str]],
        context_str: str = ""
    ) -> Tuple[str, List[Dict[str, str]], int]:
        """
        Allocate available tokens following this priority:
        1. System prompt (without context injected yet)
        2. Current user query
        3. RAG context (up to remaining budget)
        4. Recent conversation history (up to remaining budget)
        
        Returns:
            (formatted_rag_context: str, retained_history: List[Dict], safe_max_tokens: int)
        Raises:
            ValueError if system prompt + query + minimum output cannot fit.
        """
        effective_window = self.get_effective_context_window()
        min_output = Config.LLM.MIN_OUTPUT_TOKENS()

        # 1. & 2. System Prompt & Current Query (always required)
        system_msg = {"role": "system", "content": base_system_prompt}
        query_msg = {"role": "user", "content": query}
        
        base_tokens = self.REQUEST_OVERHEAD
        base_tokens += self.count_message_tokens(system_msg)
        base_tokens += self.count_message_tokens(query_msg)

        # Check if mandatory elements fit
        if base_tokens + min_output > effective_window:
            raise ValueError(
                f"Query is too long to process. Mandatory prompt ({base_tokens} tokens) + "
                f"minimum output ({min_output} tokens) exceeds effective context window ({effective_window} tokens)."
            )

        remaining_budget = effective_window - base_tokens - min_output

        # 3. RAG Context allocation
        rag_parts: List[str] = []
        rag_tokens = 0
        if search_results:
            # Prefer higher-scoring chunks
            ordered = sorted(search_results, key=lambda r: r.get("combined_score", 0.0), reverse=True)
            for result in ordered:
                block = f"[Chunk {result['index']}]\n{result['document']}\n"
                # +1 token roughly for the joining newline if parts exist
                cost = self.count_tokens(block) + (1 if rag_parts else 0)
                
                # We still respect MAX_CONTEXT_LENGTH as a secondary character cap for backward compatibility
                current_chars = sum(len(p) for p in rag_parts) + (len(rag_parts) if rag_parts else 0)
                if current_chars + len(block) > Config.LLM.MAX_CONTEXT_LENGTH():
                    continue
                    
                if rag_tokens + cost <= remaining_budget:
                    rag_parts.append(block)
                    rag_tokens += cost
                    
            formatted_rag_context = "\n".join(rag_parts)
            remaining_budget -= rag_tokens
        elif context_str:
            # Fallback if someone passed pre-built context instead of search results
            cost = self.count_tokens(context_str)
            if cost <= remaining_budget:
                formatted_rag_context = context_str
                rag_tokens = cost
            else:
                # Truncate string as best effort if it's too long
                # Roughly 4 chars per token
                allowed_chars = int(remaining_budget * 3.5)
                formatted_rag_context = context_str[:allowed_chars]
                rag_tokens = self.count_tokens(formatted_rag_context)
            remaining_budget -= rag_tokens
        else:
            formatted_rag_context = ""

        # 4. Conversation History allocation
        retained_history: List[Dict[str, str]] = []
        history_tokens = 0
        
        if history:
            # We enforce LLM_HISTORY_LENGTH for backward compat
            max_msgs = Config.LLM.LLM_HISTORY_LENGTH()
            history_to_eval = history[-max_msgs:] if len(history) > max_msgs else history
            
            # Iterate backwards to keep recent messages first
            for msg in reversed(history_to_eval):
                # We still enforce MAX_HISTORY_MESSAGE_LENGTH cap
                content = msg.get("content", "")
                if len(content) > Config.LLM.MAX_HISTORY_MESSAGE_LENGTH():
                    content = content[:Config.LLM.MAX_HISTORY_MESSAGE_LENGTH()]
                    msg = {"role": msg.get("role", "user"), "content": content}
                
                cost = self.count_message_tokens(msg)
                if history_tokens + cost <= remaining_budget:
                    retained_history.insert(0, msg)
                    history_tokens += cost
                else:
                    # Drop older messages when budget is exhausted
                    break

        # Calculate final prompt size and dynamically reduce max_tokens if needed
        # We don't just use `remaining_budget` because that doesn't include min_output which we reserved.
        # We recompute available output space:
        total_prompt_tokens = base_tokens + rag_tokens + history_tokens
        remaining_capacity = effective_window - total_prompt_tokens
        
        desired_output = Config.LLM.OPENAI_MAX_TOKENS()
        safe_max_tokens = min(desired_output, remaining_capacity)
        
        return formatted_rag_context, retained_history, safe_max_tokens
