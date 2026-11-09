"""Unit tests for verify_citations. No API key needed. Run with: python tests_verify_citations.py"""
from rag.chunking import Chunk
from rag.answer import Answer, Citation, verify_citations


def make_chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, doc_id="test-doc", text=text, meta={})


def make_answer(citations: list[Citation]) -> Answer:
    return Answer(answer="test", citations=citations, insufficient_context=False)


# ── helpers ──────────────────────────────────────────────────────────────────

CHUNK_A = make_chunk("doc#0", "אופק סיטי הוא הדגם הקטן והחסכוני של החברה.")
CHUNK_TABLE = make_chunk("doc#1",
    "אופק סיטי (OF-210) › מנוע והנעה\n"
    "| מאפיין | נתון |\n"
    "|---|---|\n"
    "| צריכת דלק משולבת | 18.5 ק״מ לליטר |\n"
    "| נפח מיכל דלק | 40 ליטר |"
)
CHUNK_B = make_chunk("doc#2",
    "כל רכב חדש של החברה מקבל אחריות יצרן ל־7 שנים\n"
    "או ל־150,000 ק״מ, המוקדם מביניהם."
)

# ── tests ─────────────────────────────────────────────────────────────────────

def test_exact_match_passes():
    """Verbatim quote from a chunk → no bad citations."""
    c = Citation(chunk_id="doc#0", quote="אופק סיטי הוא הדגם הקטן והחסכוני של החברה.")
    result = verify_citations(make_answer([c]), [CHUNK_A])
    assert result == [], f"expected [], got {result}"
    print("PASS test_exact_match_passes")


def test_whitespace_normalization_passes():
    """Same quote with extra spaces / newlines → still passes."""
    c = Citation(chunk_id="doc#2",
                 quote="כל רכב חדש של החברה מקבל אחריות יצרן ל־7 שנים\n"
                       "או ל־150,000 ק״מ, המוקדם מביניהם.")
    result = verify_citations(make_answer([c]), [CHUNK_B])
    assert result == [], f"expected [], got {result}"
    print("PASS test_whitespace_normalization_passes")


def test_different_whitespace_same_words_passes():
    """Quote with collapsed whitespace (no newline) still passes."""
    c = Citation(chunk_id="doc#2",
                 quote="כל רכב חדש של החברה מקבל אחריות יצרן ל־7 שנים או ל־150,000 ק״מ, המוקדם מביניהם.")
    result = verify_citations(make_answer([c]), [CHUNK_B])
    assert result == [], f"expected [], got {result}"
    print("PASS test_different_whitespace_same_words_passes")


def test_table_row_with_pipes_passes():
    """Quote taken from inside a table row (includes pipe chars) → passes after normalization."""
    # Claude quotes the value from the table: the pipes are part of the raw chunk text
    c = Citation(chunk_id="doc#1", quote="צריכת דלק משולבת | 18.5 ק״מ לליטר")
    result = verify_citations(make_answer([c]), [CHUNK_TABLE])
    assert result == [], f"expected [], got {result}"
    print("PASS test_table_row_with_pipes_passes")


def test_partial_quote_passes():
    """A short verbatim substring of a chunk → passes."""
    c = Citation(chunk_id="doc#0", quote="הדגם הקטן והחסכוני")
    result = verify_citations(make_answer([c]), [CHUNK_A])
    assert result == [], f"expected [], got {result}"
    print("PASS test_partial_quote_passes")


def test_paraphrase_fails():
    """Paraphrase (different words, same meaning) → bad citation."""
    c = Citation(chunk_id="doc#0", quote="אופק סיטי הוא דגם קטן וחסכוני.")  # missing "הדגם", "של החברה"
    result = verify_citations(make_answer([c]), [CHUNK_A])
    assert len(result) == 1 and result[0] is c, f"expected 1 bad citation, got {result}"
    print("PASS test_paraphrase_fails")


def test_fabricated_sentence_fails():
    """Sentence that never appeared in the chunk → bad citation."""
    c = Citation(chunk_id="doc#0", quote="אופק סיטי כולל גג פנורמי ומערכת סאונד מתקדמת.")
    result = verify_citations(make_answer([c]), [CHUNK_A])
    assert len(result) == 1 and result[0] is c, f"expected 1 bad citation, got {result}"
    print("PASS test_fabricated_sentence_fails")


def test_unknown_chunk_id_fails():
    """chunk_id not among the sent chunks → bad citation."""
    c = Citation(chunk_id="totally-made-up#99", quote="כלשהו")
    result = verify_citations(make_answer([c]), [CHUNK_A, CHUNK_B])
    assert len(result) == 1 and result[0] is c, f"expected 1 bad citation, got {result}"
    print("PASS test_unknown_chunk_id_fails")


def test_empty_citations_returns_empty_list():
    """No citations → empty list."""
    result = verify_citations(make_answer([]), [CHUNK_A])
    assert result == [], f"expected [], got {result}"
    print("PASS test_empty_citations_returns_empty_list")


def test_mixed_good_and_bad():
    """One good citation, one fabricated → only the bad one is returned."""
    good = Citation(chunk_id="doc#0", quote="הדגם הקטן והחסכוני של החברה")
    bad = Citation(chunk_id="doc#0", quote="הדגם הכי טוב בעולם")
    result = verify_citations(make_answer([good, bad]), [CHUNK_A])
    assert result == [bad], f"expected [bad], got {result}"
    print("PASS test_mixed_good_and_bad")


def test_wrong_chunk_id_right_text():
    """Quote exists verbatim in doc#0 but citation claims doc#2 → bad (wrong chunk)."""
    c = Citation(chunk_id="doc#2", quote="אופק סיטי הוא הדגם הקטן והחסכוני של החברה.")
    result = verify_citations(make_answer([c]), [CHUNK_A, CHUNK_B])
    assert len(result) == 1 and result[0] is c, f"expected 1 bad citation, got {result}"
    print("PASS test_wrong_chunk_id_right_text")


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_exact_match_passes()
    test_whitespace_normalization_passes()
    test_different_whitespace_same_words_passes()
    test_table_row_with_pipes_passes()
    test_partial_quote_passes()
    test_paraphrase_fails()
    test_fabricated_sentence_fails()
    test_unknown_chunk_id_fails()
    test_empty_citations_returns_empty_list()
    test_mixed_good_and_bad()
    test_wrong_chunk_id_right_text()
    print("\nAll tests passed.")
