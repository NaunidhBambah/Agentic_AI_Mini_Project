"""Vector knowledge base (ChromaDB): course notes, TA-verified answers, graded assignments.

Three collections:
  course_notes  - chunks of the faculty's PDFs / notes (RAG source)
  ta_answers    - answers written by TAs; this is the system's long-term "learned" memory
  assignments   - graded questions; used by the academic-integrity guardrail
"""
import hashlib
import json
import re
import time
from functools import lru_cache
from pathlib import Path

import chromadb

from . import config
from .embeddings import get_embedding


# ---------------------------------------------------------------- loading
def _read_file(path: Path):
    """Yield (text, page_number) for .md/.txt/.pdf files."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        for i, page in enumerate(PdfReader(str(path)).pages, start=1):
            yield page.extract_text() or "", i
    else:
        yield path.read_text(encoding="utf-8", errors="ignore"), None


def _unit_label(path: Path) -> str:
    """'Unit 2' etc. The library catalogue wins; otherwise read 'unit2_' from the file name."""
    entry = load_catalog().get(path.name, {})
    if "unit" in entry:                                   # "" means the teacher chose Other
        unit = entry["unit"]
        return (f"Unit {unit}" if str(unit).isdigit() else str(unit)) if unit else "Other"
    m = re.search(r"unit[_\s-]*(\d+)", path.stem, re.I)
    return f"Unit {m.group(1)}" if m else "Other"


# ---------------------------------------------------------------- library catalogue
# One JSON file next to the documents: {file: {title, unit, kind, topics, summary, pages, ...}}
def _catalog_path() -> Path:
    return config.NOTES_DIR / "_catalog.json"


def load_catalog() -> dict:
    try:
        return json.loads(_catalog_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_catalog(cat: dict):
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    _catalog_path().write_text(json.dumps(cat, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- subjects
# A teacher (or a whole department) can keep several subjects in one DoubtLoop. Every document belongs
# to one subject; searches, TA memory, doubts and reports are kept separate per subject.
DEFAULT_SUBJECT = config.COURSE_NAME


def _subjects_path() -> Path:
    return config.NOTES_DIR / "_subjects.json"


def doc_subject(file: str, cat: dict = None) -> str:
    return ((cat if cat is not None else load_catalog()).get(file, {}).get("subject") or DEFAULT_SUBJECT)


def _load_subject_meta() -> dict:
    """{name: {"category": ..., "description": ...}} from _subjects.json (older files hold plain names)."""
    try:
        raw = json.loads(_subjects_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = []
    out = {}
    for item in raw:
        if isinstance(item, str):
            out[item] = {"category": "", "description": ""}
        elif isinstance(item, dict) and item.get("name"):
            out[item["name"]] = {"category": item.get("category", ""), "description": item.get("description", ""),
                                 "web": item.get("web", True)}
    return out


def _save_subject_meta(meta: dict):
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    _subjects_path().write_text(json.dumps([{"name": k, **v} for k, v in meta.items()], indent=1,
                                           ensure_ascii=False), encoding="utf-8")


def list_subjects() -> list:
    """Subjects created by the teacher plus any used by documents (default subject first)."""
    cat = load_catalog()
    used = [doc_subject(p.name, cat) for p in config.NOTES_DIR.glob("*") if p.suffix.lower() in ALLOWED]
    return list(dict.fromkeys([DEFAULT_SUBJECT, *_load_subject_meta(), *used]))


def subject_info(name: str) -> dict:
    return {"category": "", "description": "", "web": True, **_load_subject_meta().get(name, {})}


def add_subject(name: str, category: str = "", description: str = "") -> str:
    name = re.sub(r"\s+", " ", name).strip()[:60]
    if not name:
        raise UploadError("Type a class / subject name first.")
    if name.lower() in {s_.lower() for s_ in list_subjects()}:
        raise UploadError(f"'{name}' already exists.")
    meta = _load_subject_meta()
    meta[name] = {"category": category.strip()[:40], "description": description.strip()[:200]}
    _save_subject_meta(meta)
    return name


def update_subject(name: str, category: str = None, description: str = None, web: bool = None):
    meta = _load_subject_meta()
    entry = meta.setdefault(name, {"category": "", "description": ""})
    if category is not None:
        entry["category"] = category.strip()[:40]
    if description is not None:
        entry["description"] = description.strip()[:200]
    if web is not None:
        entry["web"] = bool(web)
    _save_subject_meta(meta)


def remove_subject(name: str):
    """Only empty subjects can be removed (documents must be moved or deleted first)."""
    if name == DEFAULT_SUBJECT:
        raise UploadError("The default subject (from .env) can't be removed.")
    if any(d["subject"] == name for d in list_documents()):
        raise UploadError("Move or remove this subject's documents first.")
    meta = _load_subject_meta()
    meta.pop(name, None)
    _save_subject_meta(meta)


def chunk_text(text: str, size: int, overlap: int):
    """Heading-aware chunking: split on markdown headings, then pack paragraphs up to `size` chars."""
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
                buf = []
            heading = line.lstrip("#").strip()
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))

    chunks = []
    for heading, body in sections:
        if not body:
            continue
        paras = []
        for p in (p.strip() for p in re.split(r"\n\s*\n", body)):
            if not p:
                continue
            if len(p) <= size:
                paras.append(p)
                continue
            piece = ""                      # PDFs often have no blank lines: split long text by sentences
            for sent in re.split(r"(?<=[.!?])\s+", p.replace("\n", " ")):
                if piece and len(piece) + len(sent) > size:
                    paras.append(piece.strip())
                    piece = ""
                piece += sent + " "
            if piece.strip():
                paras.append(piece.strip())
        current = ""
        for p in paras:
            if current and len(current) + len(p) > size:
                chunks.append((heading, current.strip()))
                current = current[-overlap:] + "\n" + p if overlap else p
            else:
                current = (current + "\n\n" + p) if current else p
        if current.strip():
            chunks.append((heading, current.strip()))
    return chunks


# ---------------------------------------------------------------- client
@lru_cache(maxsize=1)
def _client():
    _, backend = get_embedding()
    path = config.CHROMA_DIR / backend          # separate index per embedding backend
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


_COLLECTIONS = {}


def _collection(name: str):
    """Open each Chroma collection once and reuse it (re-opening on every search is slow)."""
    if name not in _COLLECTIONS:
        ef, _ = get_embedding()
        _COLLECTIONS[name] = _client().get_or_create_collection(name=name, embedding_function=ef,
                                                                metadata={"hnsw:space": "cosine"})
    return _COLLECTIONS[name]


def _drop(name: str):
    """Empty a collection. Its records are deleted instead of deleting the collection itself: on Windows,
    ChromaDB can fail with 'Error creating hnsw segment reader: Nothing found on disk' when a collection
    is deleted and recreated with the same name in a running app."""
    col = _collection(name)
    ids = col.get(include=[])["ids"]
    for i in range(0, len(ids), 500):
        col.delete(ids=ids[i:i + 500])


def _reset_client():
    """Forget cached handles so the next call re-opens the index from disk."""
    _COLLECTIONS.clear()
    _client.cache_clear()


def _notes_fingerprint() -> str:
    h = hashlib.md5()
    for p in sorted(config.NOTES_DIR.glob("*")):
        if p.suffix.lower() in {".md", ".txt", ".pdf"}:
            h.update(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}".encode())
    h.update(f"{config.CHUNK_SIZE}:{config.CHUNK_OVERLAP}:v{INDEX_VERSION}".encode())
    h.update(json.dumps({k: [v.get("unit"), v.get("subject")] for k, v in load_catalog().items()},
                        sort_keys=True).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- ingestion
INDEX_VERSION = "2"      # bump when the way notes are read or chunked changes (forces one re-index)
_PAGE_NO = re.compile(r"\d+")


def _page_title(text: str) -> str:
    """First line of a slide/page when it looks like a title ('PRINCE2 Principles')."""
    for line in text.splitlines():
        line = line.strip(" #:-•\t")
        if line:
            words = line.split()
            return line if len(line) <= 80 and len(words) <= 10 and not line.endswith((".", ",", ";")) else ""
    return ""


def clean_pages(path: Path):
    """Read a file page by page and prepare it for chunking.

    - removes lines repeated on many pages (institute name, course code, 'Page 3 of 31' footers),
      which otherwise appear in every chunk and blur the search
    - joins very short pages (a title slide) to the next page, so a title is searched with its content
    - remembers pages with no text at all (scanned images, which need OCR)

    Returns (pages, stats) where pages = [(title, text, page_number)].
    """
    raw = list(_read_file(path))
    removed = set()
    if len(raw) >= 3:
        count = {}
        for text, _ in raw:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for ln in set(lines[:3] + lines[-3:]):
                key = _PAGE_NO.sub("#", ln)
                count[key] = count.get(key, 0) + 1
        removed = {k for k, c in count.items() if c >= max(3, 0.4 * len(raw)) and len(k) <= 120}
    empty, out, carry, carry_page = [], [], "", None
    for text, page in raw:
        if len(text.strip()) < 20:
            empty.append(page)
        kept = "\n".join(ln for ln in text.splitlines() if _PAGE_NO.sub("#", ln.strip()) not in removed)
        kept = kept.strip()
        if path.suffix.lower() == ".pdf" and kept and len(kept) < 60:      # title-only slide: join with next page
            carry, carry_page = (carry + "\n" + kept).strip(), carry_page or page
            continue
        if carry:
            kept, page = (carry + "\n" + kept).strip(), carry_page
            carry, carry_page = "", None
        if kept:
            out.append((_page_title(kept), kept, page))
    if carry:
        out.append((_page_title(carry), carry, carry_page))
    stats = {"pages": len(raw) if path.suffix.lower() == ".pdf" else 0,
             "empty_pages": [p for p in empty if p],
             "removed_lines": sorted(k for k in removed)[:8]}
    return out, stats


def _stats_path() -> Path:
    _, backend = get_embedding()
    return config.CHROMA_DIR / backend / "index_stats.json"


def index_stats() -> dict:
    """How each file was read: pages, empty (scanned) pages, chunks, average chunk size, removed headers."""
    try:
        return json.loads(_stats_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_notes_index(force: bool = False) -> int:
    """(Re)index course notes when files change. Returns number of chunks."""
    _, backend = get_embedding()
    marker = config.CHROMA_DIR / backend / "notes_fingerprint.txt"
    fp = _notes_fingerprint()
    col = _collection("course_notes")
    if not force and marker.exists() and marker.read_text() == fp and col.count() > 0:
        return col.count()

    _drop("course_notes")
    col = _collection("course_notes")
    ids, docs, metas, stats = [], [], [], {}
    for path in sorted(config.NOTES_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        unit = _unit_label(path)
        subject = doc_subject(path.name)
        pages, st = clean_pages(path)
        n0 = len(docs)
        for title, text, page in pages:
            for j, (heading, chunk) in enumerate(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
                head = heading or title
                ids.append(f"{path.stem}-{page or 0}-{j}")
                # every chunk keeps its slide/section title, so later chunks of a page are still findable
                docs.append(chunk if not head or chunk.startswith(head) else f"{head}\n{chunk}")
                section = head or (f"page {page}" if page else "-")
                metas.append({"source": path.name, "unit": unit, "section": section[:80], "page": page or 0,
                              "subject": subject})
        sizes = [len(d) for d in docs[n0:]]
        stats[path.name] = {**st, "chunks": len(sizes), "avg_chars": int(sum(sizes) / len(sizes)) if sizes else 0}
    for i in range(0, len(ids), 64):
        col.add(ids=ids[i:i + 64], documents=docs[i:i + 64], metadatas=metas[i:i + 64])
    marker.write_text(fp)
    _stats_path().write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    notes_vocab.cache_clear()          # offline mode's vocabulary must include new documents
    return col.count()


ALLOWED = {".pdf", ".txt", ".md"}
MAX_UPLOAD_MB = 15
KINDS = ["Lecture notes", "Slides", "Lab manual", "Assignment", "Question paper", "Syllabus", "Reference", "Other"]


class UploadError(ValueError):
    pass


def _safe_name(name: str) -> str:
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:60] or "document"
    return stem + ext


def _extract(filename: str, data: bytes):
    """(full_text, pages) from uploaded bytes, before anything is saved."""
    if filename.lower().endswith(".pdf"):
        import io

        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [p.extract_text() or "" for p in reader.pages]
        except Exception as exc:
            raise UploadError(f"Couldn't read this PDF ({exc}).") from exc
        return "\n".join(pages), len(pages)
    return data.decode("utf-8", errors="ignore"), 1


def add_document(filename: str, data: bytes, unit: str = "auto", subject: str = "",
                 title: str = "", kind: str = "") -> dict:
    """Validate, catalogue (librarian agent), save and index an uploaded course document.

    unit:    'auto' (AI decides), '1'..'9', '' (Other / general) or any label ('Syllabus', 'Mid-sem')
    subject: which subject's library it goes into (default: the course in .env)
    title, kind: optional teacher overrides; otherwise the librarian agent fills them in
    """
    from . import agents                                  # local import avoids a cycle

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise UploadError(f"{ext or 'This file type'} isn't supported yet. Upload PDF, TXT or MD.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise UploadError(f"File is larger than {MAX_UPLOAD_MB} MB. Split it into smaller parts.")
    digest = hashlib.sha1(data).hexdigest()
    cat = load_catalog()
    subject = subject or DEFAULT_SUBJECT
    for f, meta in cat.items():
        if meta.get("sha1") == digest and (config.NOTES_DIR / f).exists() and doc_subject(f, cat) == subject:
            raise UploadError(f"This exact file is already in {subject} as {f}.")
    text, pages = _extract(filename, data)
    if len(text.strip()) < 50:
        raise UploadError("No readable text found. If it's a scanned PDF, it needs OCR first.")

    meta = agents.catalog_document(filename, text, pages, subject)  # title, unit, kind, topics, summary
    if unit != "auto":
        meta["unit"] = (unit or "").strip()
        meta["unit_by"] = "teacher"
    if title.strip():
        meta["title"] = title.strip()[:90]
    if kind.strip():
        meta["kind"] = kind.strip()[:40]
    name = _safe_name(filename)
    if str(meta.get("unit", "")).isdigit():            # replace any old unit prefix with the chosen one
        name = f"unit{meta['unit']}_" + (re.sub(r"^unit[_\s-]*\d+_?", "", name, flags=re.I) or name)
    stem, ext = Path(name).stem, Path(name).suffix
    n = 2
    while (config.NOTES_DIR / name).exists():          # never overwrite a different file with the same name
        name = f"{stem}_{n}{ext}"
        n += 1
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    (config.NOTES_DIR / name).write_bytes(data)
    cat[name] = {**meta, "subject": subject, "pages": pages, "sha1": digest, "original_name": filename,
                 "uploaded_at": time.strftime("%Y-%m-%d %H:%M"), "collection": meta.get("collection", "")}
    save_catalog(cat)
    total = build_notes_index()
    return {"file": name, "chunks": len(_collection("course_notes").get(where={"source": name})["ids"]),
            "total": total, **cat[name]}


def list_documents(subject: str = None) -> list:
    """Every document (of one subject, if given) with its catalogue entry, sorted by unit then title."""
    col = _collection("course_notes")
    counts = {}
    if col.count():
        for m in col.get(include=["metadatas"])["metadatas"]:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
    cat = load_catalog()
    out = []
    for p in config.NOTES_DIR.glob("*"):
        if p.suffix.lower() not in ALLOWED:
            continue
        meta = cat.get(p.name, {})
        subj = meta.get("subject") or DEFAULT_SUBJECT
        if subject and subj != subject:
            continue
        out.append({"file": p.name, "unit": _unit_label(p), "chunks": counts.get(p.name, 0), "subject": subj,
                    "kb": round(p.stat().st_size / 1024),
                    "title": meta.get("title") or p.stem.replace("_", " "),
                    "kind": meta.get("kind") or "Other", "topics": meta.get("topics") or [],
                    "summary": meta.get("summary") or "", "pages": meta.get("pages") or 0,
                    "collection": meta.get("collection") or "", "uploaded_at": meta.get("uploaded_at") or "",
                    "catalogued_by": meta.get("by", "")})

    def unit_key(d):
        m = re.search(r"\d+", d["unit"])
        return (int(m.group()) if m else 99, d["title"].lower())

    return sorted(out, key=unit_key)


def update_document(file: str, **fields) -> dict:
    """Teacher edits (or an approved AI arrangement): title, unit, kind, collection, topics."""
    cat = load_catalog()
    entry = cat.setdefault(file, {})
    for k, v in fields.items():
        if v is not None:
            entry[k] = v
    if fields.get("unit") is not None:
        entry["unit_by"] = "teacher"
    save_catalog(cat)
    build_notes_index()                      # unit changes re-label citations
    return entry


def document_chunks(scope: str = "", limit: int = 40, subject: str = None) -> list:
    """Chunks in reading order for a file, a unit ('unit:Unit 2') or everything (for summaries/preview)."""
    col = _collection("course_notes")
    if col.count() == 0:
        return []
    where = _scope_filter(scope, subject)
    res = col.get(where=where, include=["documents", "metadatas"]) if where else \
        col.get(include=["documents", "metadatas"])

    def order(item):
        _id, _, meta = item
        parts = _id.rsplit("-", 2)
        try:
            return (meta["source"], int(parts[-2]), int(parts[-1]))
        except (ValueError, IndexError):
            return (meta["source"], 0, 0)

    items = sorted(zip(res["ids"], res["documents"], res["metadatas"]), key=order)
    if len(items) > limit:                   # spread evenly across the material instead of the first pages only
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    return [{"id": i, "text": d, "meta": m, "sim": 1.0} for i, d, m in items]


def delete_document(name: str) -> int:
    (config.NOTES_DIR / Path(name).name).unlink(missing_ok=True)
    cat = load_catalog()
    if cat.pop(Path(name).name, None) is not None:
        save_catalog(cat)
    return build_notes_index()


def recatalog_document(file: str) -> dict:
    """Run the librarian agent on a file that was uploaded before the library existed."""
    from . import agents

    path = config.NOTES_DIR / Path(file).name
    text, pages = _extract(path.name, path.read_bytes())
    meta = agents.catalog_document(path.name, text, pages, doc_subject(path.name))
    cat = load_catalog()
    old = cat.get(path.name, {})
    m = re.search(r"^unit[_\s-]*(\d+)", path.stem, re.I)
    if old.get("unit_by") == "teacher":          # a unit the teacher set in Edit always wins
        meta["unit"], meta["unit_by"] = old.get("unit", ""), "teacher"
    elif m:                                      # the teacher chose a unit at upload time
        meta["unit"], meta["unit_by"] = m.group(1), "teacher"
    cat[path.name] = {**old, **meta, "pages": pages, "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
                      "uploaded_at": old.get("uploaded_at") or time.strftime("%Y-%m-%d %H:%M"),
                      "collection": old.get("collection", "")}
    save_catalog(cat)
    build_notes_index()
    return cat[path.name]


def read_document_bytes(name: str) -> bytes:
    return (config.NOTES_DIR / Path(name).name).read_bytes()


def library_topics(limit: int = 40, subject: str = None) -> list:
    """Topics the librarian agent found across the uploaded files (most common first)."""
    from collections import Counter

    c = Counter(t for d in list_documents(subject) for t in d["topics"])
    return [t for t, _ in c.most_common(limit)]


def library_overview(max_docs: int = 25, subject: str = None) -> str:
    """Short description of what the uploaded material covers (given to the router)."""
    lines = []
    for d in list_documents(subject)[:max_docs]:
        topics = ", ".join(d["topics"][:5])
        lines.append(f"- {d['unit']}: {d['title']} ({d['kind']}){': ' + topics if topics else ''}")
    return "\n".join(lines)


def notes_count() -> int:
    return _collection("course_notes").count()


def load_assignments() -> int:
    col = _collection("assignments")
    data = json.loads((config.DATA_DIR / "assignments.json").read_text(encoding="utf-8"))
    col.upsert(ids=[a["id"] for a in data], documents=[a["question"] for a in data],
               metadatas=[{"title": a["title"]} for a in data])
    return col.count()


@lru_cache(maxsize=1)
def notes_vocab() -> frozenset:
    """All words that appear anywhere in the course notes (used by offline mode)."""
    words = set()
    for path in config.NOTES_DIR.glob("*"):
        if path.suffix.lower() in {".md", ".txt", ".pdf"}:
            for text, _ in _read_file(path):
                words |= set(re.findall(r"[a-z0-9]+", text.lower()))
    return frozenset(words)


# ---------------------------------------------------------------- queries
def _hits(col, query: str, k: int, where: dict = None):
    if col.count() == 0:
        return []
    kwargs = {"where": where} if where else {}
    try:
        res = col.query(query_texts=[query], n_results=min(k, col.count()), **kwargs)
    except Exception as exc:           # stale index handle (see _drop): re-open from disk and retry once
        if "hnsw" not in str(exc).lower() and "nothing found on disk" not in str(exc).lower():
            raise
        name = col.name
        _reset_client()
        col = _collection(name)
        if col.count() == 0:
            return []
        res = col.query(query_texts=[query], n_results=min(k, col.count()), **kwargs)
    out = []
    for doc, meta, dist, _id in zip(res["documents"][0], res["metadatas"][0],
                                    res["distances"][0], res["ids"][0]):
        out.append({"id": _id, "text": doc, "meta": meta, "sim": round(1.0 - float(dist), 3)})
    return out


def _scope_filter(scope: str, subject: str = None):
    """'' = everything, 'unit:Unit 2' = one unit, otherwise one file; always inside one subject if given.
    Missing scopes fall back to the whole subject."""
    part = None
    if scope.startswith("unit:"):
        unit = scope[5:]
        if any(d["unit"] == unit for d in list_documents(subject)):
            part = {"unit": unit}
    elif scope and (config.NOTES_DIR / scope).exists():
        part = {"source": scope}
    subj = {"subject": subject} if subject else None
    if part and subj and "source" not in part:
        return {"$and": [subj, part]}
    return part or subj


def scope_label(scope: str) -> str:
    if not scope:
        return "All course materials"
    if scope.startswith("unit:"):
        return f"{scope[5:]} (all files)"
    meta = load_catalog().get(scope, {})
    return meta.get("title") or scope


_QSTOP = set("""the and for are was were with that this these those what why how when which who whom does did
can could should would will about into than then from your you our explain tell give difference between
list define describe mean means meaning example examples please notes note unit course using used use
also more most some many much very there their them they its it's have has had any all each other""".split())


def _terms(text: str) -> list:
    words = re.findall(r"[a-z0-9][a-z0-9-]{1,}", text.lower())
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in words if w not in _QSTOP and len(w) >= 3]


def _keyword_hits(col, query: str, where, n: int):
    """Keyword (BM25-style) search over the same chunks: catches exact terms such as PRINCE2, PERT, EVA
    that a small embedding model can miss."""
    q = set(_terms(query))
    if not q:
        return []
    res = col.get(where=where, include=["documents", "metadatas"]) if where else \
        col.get(include=["documents", "metadatas"])
    docs = [(i, d, m, set(_terms(d))) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
    if not docs:
        return []
    import math
    df = {t: sum(1 for *_, ts in docs if t in ts) for t in q}
    idf = {t: math.log(1 + len(docs) / (1 + df[t])) for t in q}
    scored = [(sum(idf[t] for t in q if t in ts), i, d, m) for i, d, m, ts in docs]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0] if scored else 1
    return [{"id": i, "text": d, "meta": m, "kw": sc / best} for sc, i, d, m in scored[:n]]


def search_notes(query: str, k: int = None, source: str = "", subject: str = None):
    """Top-k note chunks. `source` = the student's focus: a file, a unit ('unit:Unit 2') or ''.

    Hybrid retrieval: semantic search (MiniLM embeddings) + keyword search, merged and re-ranked
    (score = similarity + 0.25 x keyword match). 'sim' stays the pure embedding similarity, so the
    confidence thresholds keep their meaning."""
    k = k or config.TOP_K
    col, where = _collection("course_notes"), _scope_filter(source, subject)
    sem = _hits(col, query, max(12, k * 3), where)
    if not sem:
        return []
    kw = _keyword_hits(col, query, where, 6)
    by_id = {h["id"]: dict(h, kw=0.0) for h in sem}
    missing = [h for h in kw if h["id"] not in by_id]
    if missing:                                     # keyword-only hits: compute their embedding similarity
        ef, _ = get_embedding()
        vecs = ef([query] + [h["text"] for h in missing])
        qv = vecs[0]

        def cos(a, b):
            num = sum(x * y for x, y in zip(a, b))
            den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5) or 1.0
            return num / den
        for h, v in zip(missing, vecs[1:]):
            by_id[h["id"]] = {"id": h["id"], "text": h["text"], "meta": h["meta"],
                              "sim": round(float(cos(qv, v)), 3), "kw": 0.0}
    for h in kw:
        by_id[h["id"]]["kw"] = h["kw"]
    ranked = sorted(by_id.values(), key=lambda h: -(h["sim"] + 0.25 * h["kw"]))[:k]
    best = max(h["sim"] for h in ranked)
    # drop weak extras (much lower similarity and no keyword match), so sources stay relevant
    return [h for i, h in enumerate(ranked) if i == 0 or h["kw"] >= 0.5 or h["sim"] >= 0.5 * best]


def _spans(pages: list, gap: int = 4) -> list:
    """[1, 3, 4, 6, 18] -> [(1, 6), (18, 18)]: matches close together form one continuous span."""
    spans = []
    for p in sorted(set(pages)):
        if spans and p - spans[-1][1] <= gap:
            spans[-1] = (spans[-1][0], p)
        else:
            spans.append((p, p))
    return spans


def with_neighbours(hits: list, budget: int = 9000, max_chunks: int = 16) -> list:
    """Context expansion ("small-to-big" retrieval).

    Slides hold little text each, so the best-matching chunks are often fragments of one topic spread
    over several slides (a list of 7 steps over 5 pages). Matches that are close together are joined
    into a continuous span (every page between them is added), plus one page before and after. Spans
    containing the best match come first; everything is returned in reading order. Added chunks keep
    sim=0 so confidence still depends only on the real matches."""
    if not hits:
        return hits
    col = _collection("course_notes")
    picked, used = {h["id"]: h for h in hits}, sum(len(h["text"]) for h in hits)
    by_source = {}
    for rank, h in enumerate(hits):
        by_source.setdefault(h["meta"]["source"], []).append((rank, h["meta"].get("page") or 0))
    plans = []                                       # (priority, source, pages) - lower priority first
    for src, items in by_source.items():
        best = {p: r for r, p in sorted(items, reverse=True)}
        for lo, hi in _spans([p for _, p in items if p]):
            prio = min(best[p] for p in best if lo <= p <= hi)
            plans.append((prio, src, list(range(lo, hi + 1))))            # the span itself
            plans.append((prio + 0.5, src, [lo - 1, hi + 1]))              # then one page either side
    for _, src, pages in sorted(plans):
        if used >= budget or len(picked) >= max_chunks:
            break
        try:
            res = col.get(where={"$and": [{"source": src}, {"page": {"$in": [p for p in pages if p > 0]}}]},
                          include=["documents", "metadatas"])
        except Exception:
            continue
        rows = sorted(zip(res["ids"], res["documents"], res["metadatas"]), key=lambda r: r[2].get("page", 0))
        for _id, doc, meta in rows:
            if _id in picked or used + len(doc) > budget or len(picked) >= max_chunks:
                continue
            picked[_id] = {"id": _id, "text": doc, "meta": meta, "sim": 0.0, "kw": 0.0, "neighbour": True}
            used += len(doc)

    _fill_numbered_gaps(col, picked)

    def order(c):
        parts = c["id"].rsplit("-", 2)
        try:
            return (c["meta"]["source"], int(parts[-2]), int(parts[-1]))
        except (ValueError, IndexError):
            return (c["meta"]["source"], 0, 0)
    rank = {}
    for i, h in enumerate(hits):
        rank.setdefault(h["meta"]["source"], i)            # best file first, then reading order inside it
    return sorted(picked.values(), key=lambda c: (rank.get(c["meta"]["source"], 99), order(c)))


_NUMBERED = re.compile(r"\b(Step|Stage|Phase|Level|Principle|Rule|Method|Technique|Process)\s*[-–:#]?\s*(\d{1,2})\b")


def _fill_numbered_gaps(col, picked: dict, limit: int = 4):
    """Self-repair: if the context has 'Step 2' and 'Step 4' but no 'Step 3', fetch the chunk that has it.
    Works for any numbered sequence (steps, stages, phases, principles...)."""
    seen = {}
    for c in picked.values():
        for label, num in _NUMBERED.findall(c["text"]):
            seen.setdefault((c["meta"]["source"], label.title()), set()).add(int(num))
    added = 0
    for (src, label), nums in seen.items():
        for k in range(min(nums) + 1, max(nums)):
            if k in nums or added >= limit:
                continue
            for variant in (f"{label} {k}", f"{label}-{k}", f"{label} - {k}", f"{label.upper()} {k}"):
                try:
                    res = col.get(where={"source": src}, where_document={"$contains": variant},
                                  include=["documents", "metadatas"], limit=1)
                except Exception:
                    continue
                if res["ids"] and res["ids"][0] not in picked:
                    picked[res["ids"][0]] = {"id": res["ids"][0], "text": res["documents"][0],
                                             "meta": res["metadatas"][0], "sim": 0.0, "kw": 0.0, "neighbour": True}
                    added += 1
                    break


def evidence_lines(text: str, answer: str, question: str = "", max_lines: int = 6) -> str:
    """The lines of a passage that the answer actually used (for highlighting), not the whole slide.

    Each line is scored by how many of its content words appear in the answer sentences that cite it
    (plus the question). The best lines are kept, in their original order."""
    target = set(_terms(answer + " " + question))
    scored = []
    for i, line in enumerate(ln.strip(" •-*\t") for ln in re.split(r"[\r\n]+|(?<=[.;:])\s+(?=[A-Z•])", text)):
        words = _terms(line)
        if len(line) < 6 or not words:
            continue
        hit = sum(1 for w in words if w in target)
        scored.append((hit / len(words) + 0.05 * min(hit, 6), i, line, hit))
    good = [x for x in scored if x[3] >= 2 or (x[3] >= 1 and len(_terms(x[2])) <= 3)]
    good = sorted(good, reverse=True)[:max_lines] or sorted(scored, reverse=True)[:2]
    return "\n".join(line for _, _, line, _ in sorted(good, key=lambda x: x[1]))


def _highlight_rects(textpage, passage: str):
    """PDF rectangles of the lines of `passage` found on the page (searched line by line, because
    line breaks and spaces differ slightly between text extraction tools)."""
    rects = []
    for line in re.split(r"[\r\n]+", passage):
        line = re.sub(r"\s+", " ", line).strip(" •-*#")
        if len(line) < 12:
            continue
        probe = line[:50]
        try:
            searcher = textpage.search(probe, match_case=False)
            hit = searcher.get_next()
        except Exception:
            hit = None
        if not hit:
            continue
        index, _ = hit
        count = min(len(line), textpage.count_chars() - index)
        for k in range(textpage.count_rects(index, count)):
            rects.append(textpage.get_rect(k))
    return rects


def page_image(file: str, page: int, scale: float = 1.4, highlight: str = ""):
    """PNG bytes of one PDF page (for the source viewer), with the matched passage highlighted.
    Returns None if the page can't be rendered (not a PDF, or pypdfium2 missing)."""
    path = config.NOTES_DIR / file
    if not page or path.suffix.lower() != ".pdf" or not path.exists():
        return None
    try:
        import io
        import pypdfium2 as pdfium
        from PIL import ImageDraw

        pdf = pdfium.PdfDocument(str(path))
        if page > len(pdf):
            return None
        pg = pdf[page - 1]
        img = pg.render(scale=scale).to_pil().convert("RGBA")
        if highlight:
            _, height = pg.get_size()
            overlay = img.copy()
            overlay.putalpha(0)
            draw = ImageDraw.Draw(overlay)
            for left, bottom, right, top in _highlight_rects(pg.get_textpage(), highlight):
                box = (left * scale - 2, (height - top) * scale - 2, right * scale + 2, (height - bottom) * scale + 2)
                draw.rectangle(box, fill=(255, 214, 0, 95), outline=(240, 170, 0, 220), width=1)
            img = _composite(img, overlay)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:          # pypdfium2 not installed or the page can't be drawn
        return None


def _composite(base, overlay):
    from PIL import Image
    return Image.alpha_composite(base, overlay)


def page_text(file: str, page: int) -> str:
    path = config.NOTES_DIR / file
    if not path.exists():
        return ""
    for text, p in _read_file(path):
        if (p or 0) == (page or 0):
            return text
    return ""


def search_ta_memory(query: str, k: int = 1, subject: str = None):
    col = _collection("ta_answers")
    if subject and col.count() and not col.get(where={"subject": subject}, limit=1)["ids"]:
        return []
    return _hits(col, query, k, {"subject": subject} if subject else None)


def search_assignments(query: str, k: int = 1):
    return _hits(_collection("assignments"), query, k)


def add_ta_answer(question: str, answer: str, ta_name: str, topic: str, doubt_id: str = "",
                  subject: str = None) -> str:
    """Store a TA answer so future similar questions are answered instantly ("learning")."""
    _id = doubt_id or f"ta-{int(time.time() * 1000)}"
    _collection("ta_answers").upsert(
        ids=[_id], documents=[question],
        metadatas=[{"answer": answer, "ta": ta_name, "topic": topic, "subject": subject or DEFAULT_SUBJECT,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M")}])
    return _id


def list_ta_memory(subject: str = None) -> list:
    col = _collection("ta_answers")
    if col.count() == 0:
        return []
    res = col.get(where={"subject": subject}) if subject else col.get()
    items = [{"id": i, "question": d, **m} for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
    return sorted(items, key=lambda x: x.get("saved_at", ""), reverse=True)


def delete_ta_answer(answer_id: str):
    _collection("ta_answers").delete(ids=[answer_id])


def ta_memory_count(subject: str = None) -> int:
    col = _collection("ta_answers")
    if not subject or not col.count():
        return col.count()
    return len(col.get(where={"subject": subject})["ids"])


def reset_ta_memory():
    try:
        _drop("ta_answers")
    except Exception:
        pass