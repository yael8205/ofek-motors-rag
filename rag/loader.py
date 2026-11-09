"""Loads the Markdown documents in data/ together with their metadata (the block between the --- lines)."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Doc:
    id: str            # file name without .md, e.g. "of-210-city"
    title: str
    text: str          # the document body, without the metadata block
    meta: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    _, header, body = raw.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def load_docs(folder: str = "data") -> list[Doc]:
    docs = []
    for path in sorted(Path(folder).rglob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        docs.append(Doc(id=path.stem, title=meta.get("title", path.stem), text=body.strip(), meta=meta))
    return docs
