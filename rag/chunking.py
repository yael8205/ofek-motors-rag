"""Splits documents into chunks. Each chunk keeps its own text and a copy of the document's metadata."""
from dataclasses import dataclass

from rag.loader import Doc


@dataclass
class Chunk:
    id: str            # "<doc id>#<number>", e.g. "of-210-city#2"
    doc_id: str
    text: str          # the original text, exactly as it will be shown to the model
    meta: dict


def chunk_fixed(doc: Doc, size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Baseline: cut every `size` characters, with `overlap` characters repeated between neighbours."""
    chunks = []
    step = size - overlap
    for start in range(0, max(len(doc.text), 1), step):
        piece = doc.text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, piece, dict(doc.meta)))
        if start + size >= len(doc.text):
            break
    return chunks


def chunk_by_headings(doc: Doc) -> list[Chunk]:
    """One chunk per "## " section.

    - Every chunk starts with a heading path line: "<document title> › <section heading>".
      The text before the first "## " goes in a chunk whose path line is just the title.
    - Skip the "# " title line itself (the path line already carries the title).
    - Keep each section whole, so a table stays together with its header row.
    - Copy the document metadata and add meta["section"] = the section heading.
    - Ids: f"{doc.id}#{n}", numbered from 0.
    """
    chunks: list[Chunk] = []

    # Split the document body into lines
    lines = doc.text.splitlines()

    current_heading: str | None = None   # None = before first "## "
    current_body: list[str] = []

    def flush(heading: str | None, body: list[str]) -> None:
        """Build a Chunk from the accumulated lines and append it to chunks."""
        # Strip leading/trailing blank lines from the body
        body_text = "\n".join(body).strip()

        # Build the prefix line
        if heading is None:
            prefix = doc.title
        else:
            prefix = f"{doc.title} › {heading}"

        # Combine prefix + body; skip empty-body sections that add no value
        if heading is None and not body_text:
            return  # nothing before the first heading
        text = f"{prefix}\n{body_text}" if body_text else prefix

        meta = dict(doc.meta)
        meta["section"] = heading if heading is not None else ""

        n = len(chunks)
        chunks.append(Chunk(f"{doc.id}#{n}", doc.id, text, meta))

    for line in lines:
        if line.startswith("## "):
            # Save the previous section
            flush(current_heading, current_body)
            current_heading = line[3:].strip()   # heading text without "## "
            current_body = []
        elif line.startswith("# "):
            # Top-level title line — skip it (the prefix carries the title)
            pass
        else:
            current_body.append(line)

    # Flush the last section
    flush(current_heading, current_body)

    return chunks
