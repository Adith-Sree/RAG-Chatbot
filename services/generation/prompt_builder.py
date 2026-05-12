"""
prompt_builder.py — Stage 9: Grounded Prompt Construction
===========================================================
WHY PROMPT CONSTRUCTION IS CRITICAL:
  The prompt is the interface between your retrieval system and the LLM.
  A well-constructed prompt:
    ✓ Forces the model to use ONLY retrieved context (prevents hallucination)
    ✓ Tells the model HOW to format citations
    ✓ Provides clear instruction on what to do when context is insufficient
    ✓ Controls response length and style

  A poorly constructed prompt:
    ✗ Allows the model to "fill in gaps" with hallucinated information
    ✗ Produces inconsistent citation formats
    ✗ Generates confident-sounding but unsupported claims

HALLUCINATION PREVENTION TECHNIQUES:
  1. EXPLICIT GROUNDING: "Use ONLY the provided context"
  2. REFUSAL INSTRUCTION: "If context is insufficient, say so explicitly"
  3. CITATION REQUIREMENT: Forces the model to anchor claims to sources
  4. LOW TEMPERATURE: Reduces creative divergence from facts
  5. CONTEXT ISOLATION: Separate context from instructions clearly

CONTEXT WINDOW MANAGEMENT:
  The prompt + context + question + expected answer must fit within the
  model's context window (128K for GPT-4o-mini).

  With top-n=5 chunks of ~800 tokens each:
  - Context: ~4,000 tokens
  - System prompt: ~200 tokens
  - Question: ~50 tokens
  - Instructions: ~150 tokens
  - Expected answer: up to 2,048 tokens
  Total: ~6,448 tokens — well within limits.

  DANGER: If chunks are not reranked and you send 20×800 = 16,000 tokens of
  context, you waste most of the context window on irrelevant information.

PRODUCTION CONSIDERATIONS:
  - Store prompts in .txt files (as we do) so non-engineers can tune them.
  - Version control your prompts — prompt changes are like code changes.
  - A/B test prompt variations to measure quality improvements.
  - Add conversation history for multi-turn chat support.
"""

from pathlib import Path

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class PromptBuilder:
    """
    Constructs grounded prompts for LLM generation.

    DESIGN:
      - Loads prompt templates from .txt files (easy for non-engineers to edit)
      - Formats context with source attribution for each chunk
      - Supports both one-shot and conversational prompts
      - Includes conversation history for multi-turn support

    USAGE:
        builder = PromptBuilder()
        system_prompt = builder.get_system_prompt()
        user_prompt = builder.build_answer_prompt(
            question="What is the proposed algorithm?",
            chunks=reranked_chunks,
        )
    """

    def __init__(self):
        prompts_dir = Path(settings.PROMPTS_DIR)
        self._system_prompt = self._load_template(prompts_dir / "system_prompt.txt")
        self._answer_template = self._load_template(prompts_dir / "answer_prompt.txt")
        self._citation_template = self._load_template(prompts_dir / "citation_prompt.txt")
        logger.info("PromptBuilder initialized with templates from disk.")

    @staticmethod
    def _load_template(path: Path) -> str:
        """Loads a prompt template from a .txt file."""
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def get_system_prompt(self) -> str:
        """Returns the system-level instruction prompt."""
        return self._system_prompt

    def build_answer_prompt(
        self,
        question: str,
        chunks: list[dict],
        conversation_history: list[dict] = None,
    ) -> str:
        """
        Builds the full answer prompt by injecting ranked chunks as context.

        CONTEXT FORMAT:
          Each chunk is labeled with its source and page number:
          [CHUNK 1 | Source: report.pdf | Page: 5]
          <chunk text here>

          This labeling helps the LLM form accurate citations like:
          "According to the report [Source: report.pdf, Page 5]..."

        WHY WE INCLUDE CHUNK INDEX?
          The LLM can reference multiple chunks and the citation extractor
          can validate that cited chunks actually exist in the context.

        Args:
            question: The user's question.
            chunks: Reranked chunks from the Reranker.
            conversation_history: Optional list of previous messages for multi-turn.

        Returns:
            Formatted prompt string ready for the LLM.
        """
        if not chunks:
            # Build a "no context" prompt — model should refuse to answer
            context_text = "[NO RELEVANT CONTEXT FOUND IN DOCUMENTS]"
            logger.warning("Building prompt with no retrieved chunks.")
        else:
            context_parts = []
            for i, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                source = meta.get("source", "Unknown")
                page = meta.get("page_number", "?")
                score = chunk.get("reranker_score", chunk.get("score", 0))

                # Format each chunk with clear source attribution
                chunk_header = f"[CHUNK {i+1} | Source: {source} | Page: {page} | Relevance Score: {score:.3f}]"
                context_parts.append(f"{chunk_header}\n{chunk['text']}")

            context_text = "\n\n---\n\n".join(context_parts)

        # Determine the first source name for the template placeholder
        source_name = chunks[0]["metadata"].get("source", "the document") if chunks else "the document"

        # Format the template with actual values
        prompt = self._answer_template.format(
            context=context_text,
            question=question,
            source_name=source_name,
        )

        # Prepend conversation history for multi-turn support
        if conversation_history:
            history_text = self._format_conversation_history(conversation_history)
            prompt = f"{history_text}\n\n---\n\nNew Question:\n{prompt}"

        return prompt

    def build_citation_extraction_prompt(self, answer_text: str) -> str:
        """
        Builds a prompt to extract structured citations from the LLM's answer.

        WHY SEPARATE CITATION EXTRACTION?
          We could ask the LLM to produce structured JSON directly.
          But combining answer generation + JSON formatting in one prompt
          often leads to less readable answers.

          Better: generate a natural-language answer first, then
          separately extract citations from it. Two focused tasks > one confused task.

        Args:
            answer_text: The LLM's generated answer text.

        Returns:
            Formatted citation extraction prompt.
        """
        return f"{answer_text}\n\n{self._citation_template}"

    @staticmethod
    def _format_conversation_history(history: list[dict]) -> str:
        """
        Formats conversation history for multi-turn prompts.

        Expected history format:
            [
                {"role": "user", "content": "What is the main topic?"},
                {"role": "assistant", "content": "The main topic is..."},
            ]

        WHY INCLUDE HISTORY?
          Follow-up questions like "Can you elaborate on that?" require
          the previous answer as context. Without history, the LLM has no
          idea what "that" refers to.

        Args:
            history: List of {"role": str, "content": str} dicts.

        Returns:
            Formatted conversation history string.
        """
        lines = ["=== PREVIOUS CONVERSATION ==="]
        for msg in history[-6:]:  # Limit to last 3 exchanges (6 messages) to save tokens
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("=== END OF CONVERSATION HISTORY ===")
        return "\n".join(lines)
