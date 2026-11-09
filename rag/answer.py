"""Answers a question from the retrieved chunks, as a Pydantic object with citations."""
from pydantic import BaseModel, Field

from rag.chunking import Chunk

MODEL = "claude-haiku-5-5"

SYSTEM = """אתה עונה על שאלות של לקוחות חברת הרכב "אופק רכב".
ענה רק על סמך הקטעים שבתוך <chunk>. אל תשתמש בידע כללי.
לכל טענה בתשובה, הוסף ציטוט: מזהה הקטע וציטוט מדויק מתוכו, מילה במילה.
אם הקטעים לא מספיקים כדי לענות, החזר insufficient_context=true, בלי ציטוטים.
ענה בעברית, במשפטים קצרים."""


class Citation(BaseModel):
    chunk_id: str = Field(description="The chunk id, exactly as it appears in <chunk id=...>")
    quote: str = Field(description="An exact quote from that chunk, word for word")


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool


def build_context(chunks: list[Chunk]) -> str:
    import re
    def strip_html_comments(text: str) -> str:
        """Remove HTML comments (<!-- ... -->) from chunk text before sending to the model.

        This prevents prompt-injection attacks embedded as HTML comments in documents.
        The comment is silently dropped; the rest of the chunk text is unchanged.
        """
        return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()

    return "\n\n".join(
        f'<chunk id="{c.id}">\n{strip_html_comments(c.text)}\n</chunk>'
        for c in chunks
    )


def ask(question: str, chunks: list[Chunk], client=None) -> tuple[Answer, object]:
    """Returns the parsed Answer and the token usage. Needs ANTHROPIC_API_KEY in the environment."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"{build_context(chunks)}\n\nשאלה: {question}"}],
        output_format=Answer,
    )
    return response.parsed_output, response.usage


def _normalize(text: str) -> str:
    """Normalize text for substring matching.

    What we normalize:
    - All whitespace (spaces, tabs, newlines, NBSP \u00a0) → single space.
      Claude may quote across line breaks or reformat spacing.
    - Markdown table pipes "|" → space. A quote from a table row may include
      pipe characters that aren't part of the actual words.
    - Bold markers "**" → removed. Claude may include or drop bold markers.
    - Backticks "`" → removed. Same reason.
    - Leading/trailing whitespace stripped.

    What we deliberately do NOT normalize:
    - Hebrew letters, niqqud, geresh/gershayim (״ / ׳), punctuation inside words.
      A paraphrase or fabricated sentence must still fail.
    - Numeric values, units, model codes — exact match required.
    """
    import re
    text = text.replace("\u00a0", " ")          # NBSP → space
    text = text.replace("|", " ")               # table pipe → space
    text = text.replace("**", "")               # bold markers
    text = text.replace("`", "")                # backticks
    text = re.sub(r"\s+", " ", text)            # collapse all whitespace runs
    return text.strip()


def verify_citations(answer: Answer, chunks: list[Chunk]) -> list[Citation]:
    """Return the citations that are NOT backed by the chunks.

    A citation is bad if its chunk_id is not one of the chunks we sent,
    or if its quote does not appear in that chunk's text.
    An empty list means every citation checks out.
    """
    # Build a lookup from chunk id → normalized chunk text
    chunk_map = {c.id: _normalize(c.text) for c in chunks}

    bad: list[Citation] = []
    for citation in answer.citations:
        chunk_text = chunk_map.get(citation.chunk_id)
        if chunk_text is None:
            # chunk_id not among the chunks we sent → fabricated reference
            bad.append(citation)
        elif _normalize(citation.quote) not in chunk_text:
            # quote not found verbatim (after normalization) → bad citation
            bad.append(citation)
    return bad
