"""OCR text correction using OpenAI-compatible API (OpenAI or Gemini)."""

import os
import asyncio
from typing import Literal, Tuple

from openai import AsyncOpenAI


class OCRCorrector:
    """Corrects OCR errors in Hebrew text using OpenAI-compatible API.
    
    Supports both OpenAI and Gemini providers through OpenAI-compatible endpoints.
    Implements hybrid mode: splits text into Head (complex) and Body (easier) parts,
    processing them with different model strengths concurrently.
    
    Attributes:
        client: AsyncOpenAI client instance.
        provider: The provider being used ("openai" or "gemini").
        strong_model: Model name for complex text (head).
        weak_model: Model name for simpler text (body).
        system_prompt: System instruction for OCR correction.
    """
    
    # Prompt v1: existing detailed prompt (kept for fallback/A-B testing)
    SYSTEM_PROMPT = """You are an expert Hebrew Text Restoration Engine specializing in digitized newspaper archives (1980s-1990s).

TASK:
Reconstruct corrupted OCR text into clean, grammatically correct Hebrew while preserving the original journalistic style.

CORE REPAIR RULES:
1. **Fix Visual Scans**: Correct common OCR letter swaps (e.g., 'ו' vs 'ן', 'ם' vs 'ס', 'ב' vs 'כ', 'י' vs 'ו').
2. **Restore Logic**: Fix words that are visually similar to the correct word (e.g., "משוות" -> "מקורות", "הוסב" -> "חוסני" if context fits).
3. **Fix Garbled Numbers/Ranges**: A number followed by garbage and another number is usually a range.
   - Example: "10 דם2" -> "10-20"
   - Example: "1980 !99" -> "1980-1999"
4. **Remove Metadata**: Aggressively remove reporter names (bylines) and credits often found at the start/end of articles or embedded in the text.
   - Remove patterns like: "שמעון אלקבץ, אורן כהן", "(עמ' 3)", "מאת סופר עיתון".
5. **Preserve Content**: Do not summarize. Do not modernize. Keep the text length roughly the same.
. **Header/Body Consistency**: The first 2-3 sentences (headlines) are often the most corrupted. Use the context of the later text to infer names/places in the headline. (e.g., if the body mentions "Mubarak", fix "Hubarak" in the title).

### FEW-SHOT EXAMPLES (General Patterns):

Input:
"השר אמר ש1לום לכו לם. המדד עלה ב־0.5 אחוז."
Output:
"השר אמר שלום לכולם. המדד עלה ב-0.5 אחוז."

Input:
"הפגנה נערכה בכיכר. בין 50 דם00 איש הגיעו. דני ידעןגם המשטרה הגיעה."
Output:
"הפגנה נערכה בכיכר. בין 50 ל-100 איש הגיעו. גם המשטרה הגיעה."

Input:
"משוות במשרד החוץ מסרו כי הפגישה תתקיים."
Output:
"מקורות במשרד החוץ מסרו כי הפגישה תתקיים."

Output ONLY the corrected Hebrew text. No Markdown.
"""

    # Prompt v2: concise + stricter control rules
    SYSTEM_PROMPT_V2 = """You are a Hebrew OCR correction assistant for newspaper archives from the 1980s-1990s.

Task:
Correct OCR errors while preserving the original meaning and journalistic style.

Critical connection rule:
- The first part and second part are strongly connected.
- The first part usually has the worst OCR errors.
- Read the rest carefully first, then correct the first part using that context.
- Use later context to resolve names, places, entities, and ambiguous words in the opening lines.

Step order (always):
1) Remove metadata/bylines/credits.
2) Correct OCR character/word/spacing/punctuation errors.
3) Do a final Hebrew grammar and punctuation pass.

Hard rules:
- Do NOT summarize.
- Do NOT modernize style.
- Do NOT invent facts, names, places, or numbers.
- If uncertain, prefer conservative OCR repair over semantic rewriting.
- If two fixes are plausible, choose the one most consistent with repeated forms later in the text.
- Preserve paragraph structure whenever possible.

Metadata removal (aggressive, anywhere in text):
- Remove bylines/reporter credits and author lists: e.g., "אילן כפיר, פאר־לי שחר", "מאת ...", "כתבנו ...", "שליחנו ...", "מערכת", "צילום:", "איור:".
- Remove page markers: e.g., "עמ' 3", "(עמ׳ 2-3)".

Output:
Return only corrected Hebrew text.
No explanations, labels, markdown, bullets, or extra wrappers.
"""

    def __init__(
        self,
        provider: Literal["openai", "gemini"] = "openai",
        prompt_version: Literal["v1", "v2"] | None = None,
    ) -> None:
        """Initialize the OCR corrector.
        
        Args:
            provider: The provider to use ("openai" or "gemini").
            prompt_version: Prompt version to use ("v1" or "v2").
                           If None, reads OCR_PROMPT_VERSION env var (default: "v1").
            
        Raises:
            ValueError: If provider is invalid or API key is missing.
        """
        self.provider = provider.lower()
        selected_prompt_version = (
            (prompt_version or os.getenv("OCR_PROMPT_VERSION", "v1")).lower()
        )
        if selected_prompt_version not in {"v1", "v2"}:
            raise ValueError(
                f"Invalid prompt_version: {selected_prompt_version}. Must be 'v1' or 'v2'"
            )
        self.prompt_version: Literal["v1", "v2"] = selected_prompt_version
        self.system_prompt = (
            self.SYSTEM_PROMPT_V2 if self.prompt_version == "v2" else self.SYSTEM_PROMPT
        )
        
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set")
            
            self.client = AsyncOpenAI(api_key=api_key)
            self.strong_model = "gpt-4.1-mini"
            self.weak_model = "gpt-4o-mini"
            
        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set"
                )
            
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            self.strong_model = "gemini-2.5-flash"
            self.weak_model = "gemini-2.5-flash-lite"
            
        else:
            raise ValueError(f"Invalid provider: {provider}. Must be 'openai' or 'gemini'")
    
    def _split_head_body(self, text: str) -> Tuple[str, str]:
        """Split text into Head (complex start) and Body (easier text).
        
        Tries multiple strategies:
        1. Split by first 3 newlines
        2. Split by first 3 periods
        3. Fallback to first 400 characters
        
        Args:
            text: The text to split.
            
        Returns:
            Tuple of (head_text, body_text).
        """
        text = text.strip()
        
        if not text:
            return ("", "")
        
        # Strategy 1: Split by first 3 newlines
        lines = text.split("\n")
        if len(lines) >= 4:
            head = "\n".join(lines[:3])
            body = "\n".join(lines[3:])
            return (head, body)
        
        # Strategy 2: Split by first 3 periods
        periods = [i for i, char in enumerate(text) if char == "."]
        if len(periods) >= 3:
            split_idx = periods[2] + 1
            head = text[:split_idx].strip()
            body = text[split_idx:].strip()
            if head and body:
                return (head, body)
        
        # Strategy 3: Fallback to first 400 characters
        if len(text) > 400:
            head = text[:400].strip()
            body = text[400:].strip()
            return (head, body)
        
        # If text is too short, return all as head
        return (text, "")
    
    async def _correct_chunk(
        self,
        text: str,
        model: str,
        is_head: bool = False,
        full_text_context: str | None = None,
    ) -> str:
        """Correct a chunk of text using the specified model.
        
        Args:
            text: The text chunk to correct.
            model: The model to use for correction.
            is_head: Whether this is the head (complex) portion.
            
        Returns:
            Corrected text string.
            
        Raises:
            ValueError: If the API call fails.
        """
        if not text.strip():
            return ""
        
        section_name = "HEAD" if is_head else "BODY"
        context_block = ""
        if is_head and full_text_context:
            context_block = f"""
FULL ARTICLE CONTEXT (for disambiguation only):
{full_text_context}

IMPORTANT:
- Use the full context only to improve correction accuracy for names/places/terms.
- Output ONLY the corrected {section_name} section below.
- Do NOT output the full article.
- Do NOT add explanations, labels, or markdown.
"""

        user_prompt = f"""The following section contains OCR errors from a Hebrew newspaper scan.

REQUESTED SECTION ({section_name}) TO CORRECT:
{text}
{context_block}

Section-specific requirements:
- Correct only the REQUESTED SECTION text.
- If this is HEAD: use the full article context to resolve entities, but output only corrected HEAD.
- Remove bylines/reporter names/credits if they appear in this section.
- Remove page-number metadata if present.
- Do not summarize and do not add facts.
- Keep original meaning and structure.

Return only the corrected requested section text (no labels/markdown)."""
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            corrected_text = response.choices[0].message.content
            return corrected_text.strip() if corrected_text else ""
            
        except Exception as e:
            raise ValueError(f"Error calling {self.provider} API with model {model}: {e}")
    
    async def correct_text(
        self, 
        text: str, 
        use_hybrid: bool = True,
        use_strong_only: bool = False
    ) -> str:
        """Correct OCR errors in Hebrew text (async version).
        
        Args:
            text: The OCR-extracted text to correct.
            use_hybrid: If True, use hybrid mode (strong model for head, weak for body).
                       If False, use weak model for entire text.
            use_strong_only: If True, use only strong model for entire text (no splitting).
                           Overrides use_hybrid if True.
            
        Returns:
            Corrected text string.
            
        Raises:
            ValueError: If the API call fails.
        """
        if not text.strip():
            return ""
        
        if use_strong_only:
            # Use only strong model for entire text (no splitting)
            return await self._correct_chunk(
                text,
                self.strong_model,
                is_head=False,
                full_text_context=text,
            )
        
        if use_hybrid:
            head, body = self._split_head_body(text)
            
            # Process head and body concurrently
            if body:
                corrected_head, corrected_body = await asyncio.gather(
                    self._correct_chunk(
                        head,
                        self.strong_model,
                        is_head=True,
                        full_text_context=text,
                    ),
                    self._correct_chunk(body, self.weak_model, is_head=False)
                )
                return f"{corrected_head}\n{corrected_body}".strip()
            else:
                # Only head exists
                corrected_head = await self._correct_chunk(
                    head,
                    self.strong_model,
                    is_head=True,
                    full_text_context=text,
                )
                return corrected_head
        else:
            # Use weak model for entire text
            return await self._correct_chunk(text, self.weak_model)
    