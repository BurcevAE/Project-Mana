"""
mana.graph_memory — layered, graph-structured long-term memory built on top
of MemoryManager.

What this adds, concretely, on top of the existing flat memory_items/events
tables (nothing here replaces them -- see the module docstring's last
section for why):

  1. **Distillation, not verbatim storage.** Every recorded turn goes
     through `distill_turn()` first: filler is stripped and only the
     highest-information sentence(s) are kept, hard-capped in length. The
     graph never stores "what was literally said" -- it stores the
     conclusion. (The existing `events` table still keeps the raw
     conversation for short-term continuity/audit -- this module is an
     additional, compressed long-term layer, not a replacement for it.)

  2. **A real graph, not a flat table.** Nodes are `memory_items` rows
     (reusing MemoryManager's existing table + embeddings, tagged with a
     new `item_type`); a new `memory_edges` table connects them:
     FOLLOWS (temporal chain within a session), MENTIONS (turn -> entity),
     PART_OF (turn -> the episode it got rolled up into).

  3. **Two layers.** Layer 1 = one distilled node per turn. Layer 2 =
     episode rollups: every `episode_every_n_turns` turns, the recent
     layer-1 nodes for a session are distilled *again* (a summary of
     summaries) into one layer-2 node, linked back via PART_OF. Querying
     can be answered from a single layer-2 node instead of walking dozens
     of layer-1 ones once a session gets long -- the point of having
     layers at all.

  4. **Context retrieval by graph traversal, not just top-k similarity.**
     `graph_context()` seeds from semantic_search, then walks MENTIONS/
     FOLLOWS/PART_OF edges outward with per-hop decay, so a node that
     shares an entity with the query but has low lexical/embedding
     overlap with it can still surface -- something flat top-k search
     structurally cannot do.

Nothing here needs a new database engine or a new process: it is a
handful of extra SQLite tables in the same file MemoryManager already
manages, and pure-Python traversal over rows fetched into memory only for
the (small) touched neighborhood -- consistent with the "must run on a
plain desktop PC" constraint the rest of the project already works under.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

#: Component version -- see mana/version.py for the bump conventions.
__version__ = "1.0"

# --- distillation --------------------------------------------------------

_FILLER_PATTERNS = [
    # Longer/more-specific phrases must come before their shorter prefixes
    # in this list, or the shorter pattern matches first and leaves a
    # dangling remainder (e.g. "в общем-то" -> "в общем" matches, leaving
    # a stray "-то").
    r"\bв общем-то\b", r"\bв общем\b",
    r"\bкороче говоря\b", r"\bкороче\b",
    r"\bсобственно говоря\b", r"\bсобственно\b",
    r"\bтипа того\b", r"\bтипа\b",
    r"\bкак говорится\b", r"\bтак сказать\b",
    r"\bну\b", r"\bкак бы\b", r"\bзначит\b", r"\bэто самое\b", r"\bпонимаешь\b",
    r"\bреально\b", r"\bвот смотри\b", r"\bсмотри\b", r"\bслушай\b",
    r"\bwell\b", r"\bactually\b", r"\bbasically\b", r"\byou know\b",
    r"\bkind of\b", r"\bsort of\b", r"\bi mean\b", r"\bi guess\b", r"\blike\b",
]
_FILLER_RE = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)


def strip_filler(text: str) -> str:
    """Remove common filler words/phrases and collapse the resulting
    whitespace/punctuation debris. Deterministic, no LLM needed -- this is
    the fallback path used whenever an LLM-based distiller isn't
    available. Imperfect by nature (it's regex, not language
    understanding) -- it removes the most blatant filler and caps length;
    it does not produce grammatically polished text. The LLM-based path in
    distill_turn() is what actually produces clean summaries when an LLM
    is available; this is only the degrade-gracefully fallback."""
    t = _FILLER_RE.sub(" ", text or "")
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s*,\s*,+", ",", t)          # filler removal often leaves "X, , Y"
    t = re.sub(r"(^|[.!?]\s*)\s*,\s*", r"\1", t)  # stray leading comma after a removed opener
    return t.strip(" ,.-\u2014")


def _sentence_density(sentence: str) -> float:
    """Cheap information-density proxy: digits and capitalized tokens are
    usually names/numbers/facts; longer sentences carry more content than
    short filler-only ones. No NLP model required."""
    digits = sum(ch.isdigit() for ch in sentence)
    caps = sum(1 for w in sentence.split() if w[:1].isupper())
    return digits * 2 + caps * 1.5 + len(sentence.split()) * 0.3


def extractive_distill(text: str, max_chars: int = 240) -> str:
    """Strip filler, then keep the most information-dense sentence(s) up to
    max_chars. This is the no-LLM fallback distiller."""
    cleaned = strip_filler(text).strip()
    if not cleaned:
        return ""
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not raw_sentences:
        return cleaned[:max_chars]
    by_density = sorted(enumerate(raw_sentences), key=lambda p: _sentence_density(p[1]), reverse=True)
    chosen: List[int] = []
    total = 0
    for idx, s in by_density:
        if chosen and total + len(s) + 1 > max_chars:
            continue
        chosen.append(idx)
        total += len(s) + 1
        if total >= max_chars:
            break
    chosen.sort()  # restore original reading order among the sentences we kept
    result = " ".join(raw_sentences[i] for i in chosen).strip()
    return result[:max_chars]


def distill_turn(user_text: str, assistant_text: str, llm_ask: Optional[Callable[[str], str]] = None,
                  max_chars: int = 240) -> str:
    """Turn one (user, assistant) exchange into a compact "conclusion"
    node. Prefers an LLM one-sentence summary when `llm_ask` is given and
    returns something usable; otherwise falls back to the deterministic
    extractive distiller -- same graceful-degradation pattern the rest of
    MANA already follows for every other optional capability."""
    if llm_ask is not None:
        try:
            prompt = (
                "Сожми это в ОДНО предложение — только вывод/факт, без вводных слов "
                f"и воды, максимум {max_chars} символов.\n"
                f"Вопрос: {user_text}\nОтвет: {assistant_text}"
            )
            summary = (llm_ask(prompt) or "").strip()
            if summary:
                return summary[:max_chars]
        except Exception:
            pass
    combined = f"{user_text.strip()} -> {assistant_text.strip()}" if user_text.strip() else assistant_text
    return extractive_distill(combined, max_chars=max_chars)


# --- lightweight entity extraction (connective tissue for MENTIONS edges) -

_STOPWORDS = {"это", "этот", "эта", "эти", "того", "который", "которая", "которые",
              "the", "this", "that", "these", "those", "and", "for", "with"}
_ENTITY_RE = re.compile(r"[A-ZА-Я][a-zа-яA-Za-z0-9_+#.-]{2,}")


def extract_entities(text: str, limit: int = 5) -> List[str]:
    """Heuristic entity/keyword extraction: capitalized tokens (proper
    nouns, tech terms, acronyms) minus stopwords, deduplicated,
    length-capped. Deliberately simple -- this only needs to be good
    enough to link related turns, not to do full NER."""
    seen: List[str] = []
    for m in _ENTITY_RE.finditer(text or ""):
        tok = m.group(0)
        low = tok.lower()
        if low in _STOPWORDS or low in seen:
            continue
        seen.append(low)
        if len(seen) >= limit:
            break
    return seen


# --- the graph store itself -----------------------------------------------

NODE_TURN = "graph_turn"
NODE_ENTITY = "graph_entity"
NODE_EPISODE = "graph_episode"

EDGE_FOLLOWS = "FOLLOWS"
EDGE_MENTIONS = "MENTIONS"
EDGE_PART_OF = "PART_OF"


@dataclass
class GraphTraversalTrace:
    seeds: List[int]
    visited: List[int]
    hops: int


class GraphMemoryStore:
    """Adds a graph layer on top of an existing MemoryManager instance,
    reusing its SQLite connection, lock, embedder and memory_items table
    (as graph nodes) rather than standing up parallel infrastructure."""

    def __init__(self, memory_manager: Any):
        self.mm = memory_manager
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.mm.lock, self.mm.con:
            self.mm.con.executescript("""
            CREATE TABLE IF NOT EXISTS memory_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                src_id INTEGER NOT NULL,
                dst_id INTEGER NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                UNIQUE(src_id, dst_id, edge_type)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_src ON memory_edges(src_id, edge_type);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON memory_edges(dst_id, edge_type);
            """)

    # -- node/edge primitives --

    def _add_edge(self, src_id: int, dst_id: int, edge_type: str, weight: float = 1.0) -> None:
        if not src_id or not dst_id or src_id == dst_id:
            return
        with self.mm.lock, self.mm.con:
            self.mm.con.execute(
                "INSERT OR IGNORE INTO memory_edges(src_id,dst_id,edge_type,weight,created_at) VALUES(?,?,?,?,?)",
                (src_id, dst_id, edge_type, float(weight), time.time()))

    def _get_or_create_entity(self, name: str, session_id: str) -> int:
        with self.mm.lock:
            row = self.mm.con.execute(
                "SELECT id FROM memory_items WHERE item_type=? AND text=? LIMIT 1",
                (NODE_ENTITY, name)).fetchone()
        if row:
            return int(row["id"])
        return self.mm.upsert_memory_item(NODE_ENTITY, name, session_id=session_id, importance=0.3,
                                           provenance={"kind": "entity"})

    def _last_turn_node(self, session_id: str) -> Optional[int]:
        with self.mm.lock:
            row = self.mm.con.execute(
                "SELECT id FROM memory_items WHERE item_type=? AND session_id=? "
                "ORDER BY created_at DESC LIMIT 1", (NODE_TURN, session_id)).fetchone()
        return int(row["id"]) if row else None

    # -- writing --

    def record_turn(self, session_id: str, user_text: str, assistant_text: str,
                     llm_ask: Optional[Callable[[str], str]] = None, importance: float = 0.5) -> int:
        """Distill one exchange and add it to the graph: a new layer-1
        node, a FOLLOWS edge from the previous turn in this session, and
        MENTIONS edges to whatever entities it references."""
        distilled = distill_turn(user_text, assistant_text, llm_ask=llm_ask)
        if not distilled:
            return 0
        prev_id = self._last_turn_node(session_id)
        node_id = self.mm.upsert_memory_item(
            NODE_TURN, distilled, session_id=session_id, importance=importance,
            provenance={"kind": "turn", "source_chars": len(user_text) + len(assistant_text),
                        "distilled_chars": len(distilled)})
        if prev_id:
            self._add_edge(prev_id, node_id, EDGE_FOLLOWS, weight=1.0)
        for entity in extract_entities(distilled):
            entity_id = self._get_or_create_entity(entity, session_id)
            self._add_edge(node_id, entity_id, EDGE_MENTIONS, weight=0.6)
        return node_id

    def maybe_rollup_episode(self, session_id: str, every_n_turns: int = 12,
                              llm_ask: Optional[Callable[[str], str]] = None) -> Optional[int]:
        """Every `every_n_turns` layer-1 turns since the last rollup,
        compress them into one layer-2 episode node (a distillation of
        distillations), linked back to every turn it summarizes."""
        with self.mm.lock:
            last_episode = self.mm.con.execute(
                "SELECT id, created_at FROM memory_items WHERE item_type=? AND session_id=? "
                "ORDER BY created_at DESC LIMIT 1", (NODE_EPISODE, session_id)).fetchone()
            since = float(last_episode["created_at"]) if last_episode else 0.0
            turns = self.mm.con.execute(
                "SELECT id, text FROM memory_items WHERE item_type=? AND session_id=? AND created_at>? "
                "ORDER BY created_at ASC", (NODE_TURN, session_id, since)).fetchall()
        if len(turns) < every_n_turns:
            return None
        combined = " ".join(r["text"] for r in turns)
        summary = distill_turn("", combined, llm_ask=llm_ask, max_chars=400)
        if not summary:
            return None
        episode_id = self.mm.upsert_memory_item(
            NODE_EPISODE, summary, session_id=session_id, importance=0.7,
            provenance={"kind": "episode_rollup", "turns_summarized": len(turns)})
        for r in turns:
            self._add_edge(int(r["id"]), episode_id, EDGE_PART_OF, weight=1.0)
        return episode_id

    # -- reading: graph traversal, not just top-k --

    def graph_context(self, session_id: str, query: str, depth: int = 2, limit: int = 8,
                       char_budget: int = 2000, recency_backbone: int = 2, seed_limit: int = 3
                       ) -> Tuple[str, Dict[str, Any]]:
        """Seed from semantic_search, then walk MENTIONS/FOLLOWS/PART_OF
        edges outward with per-hop decay so entity-linked-but-lexically-
        distant nodes can still surface, merge in a short recency
        backbone, rank, and budget-limit the result.

        `seed_limit` is deliberately independent of `limit`: it controls
        how many *direct* semantic matches seed the traversal, not how
        many results come out at the end. Coupling the two (asking for
        more output => seeding with more direct matches) would let enough
        weak direct matches into the seed set to outrank anything actually
        found by traversal purely by seed-bonus, defeating the point of
        walking the graph at all -- keep the seed set small and let hops
        earn their way into the ranked output instead.
        """
        seeds = self.mm.semantic_search(query, limit=max(1, seed_limit), session_id=session_id, cross_session=True)
        seed_ids = [int(r["id"]) for r in seeds if r.get("item_type") in (NODE_TURN, NODE_EPISODE)]

        scores: Dict[int, float] = {nid: 1.0 for nid in seed_ids}
        frontier: Set[int] = set(seed_ids)
        visited: Set[int] = set(seed_ids)
        decay = 0.55
        for hop in range(depth):
            if not frontier:
                break
            next_frontier: Set[int] = set()
            with self.mm.lock:
                placeholders = ",".join("?" * len(frontier))
                rows = self.mm.con.execute(
                    f"SELECT src_id, dst_id, weight FROM memory_edges "
                    f"WHERE src_id IN ({placeholders}) OR dst_id IN ({placeholders})",
                    tuple(frontier) * 2).fetchall()
            for r in rows:
                for a, b in ((r["src_id"], r["dst_id"]), (r["dst_id"], r["src_id"])):
                    if a in frontier and b not in visited:
                        gain = float(r["weight"]) * (decay ** hop)
                        scores[b] = max(scores.get(b, 0.0), scores.get(a, 0.5) * gain)
                        next_frontier.add(b)
            visited |= next_frontier
            frontier = next_frontier

        with self.mm.lock:
            recent = self.mm.con.execute(
                "SELECT id FROM memory_items WHERE item_type=? AND session_id=? "
                "ORDER BY created_at DESC LIMIT ?", (NODE_TURN, session_id, recency_backbone)).fetchall()
        for r in recent:
            scores[int(r["id"])] = max(scores.get(int(r["id"]), 0.0), 0.8)
            visited.add(int(r["id"]))

        if not visited:
            return "", {"seeds": [], "visited": [], "hops": depth}

        with self.mm.lock:
            placeholders = ",".join("?" * len(visited))
            rows = self.mm.con.execute(
                f"SELECT id, item_type, text, importance FROM memory_items "
                f"WHERE id IN ({placeholders}) AND item_type IN (?,?)",
                tuple(visited) + (NODE_TURN, NODE_EPISODE)).fetchall()
        ranked = sorted(
            ({"id": int(r["id"]), "text": r["text"],
              "score": scores.get(int(r["id"]), 0.0) * 0.7 + float(r["importance"]) * 0.3}
             for r in rows),
            key=lambda d: d["score"], reverse=True)

        out_lines: List[str] = []
        total = 0
        used_ids: List[int] = []
        for item in ranked:
            line = f"- {item['text']}"
            if total + len(line) > char_budget and out_lines:
                break
            out_lines.append(line)
            total += len(line) + 1
            used_ids.append(item["id"])
            if len(used_ids) >= limit:
                break

        trace = {"seeds": seed_ids, "visited": sorted(visited), "used": used_ids, "hops": depth}
        return "\n".join(out_lines), trace

    def stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        with self.mm.lock:
            def count(item_type: str) -> int:
                if session_id:
                    return int(self.mm.con.execute(
                        "SELECT COUNT(*) AS n FROM memory_items WHERE item_type=? AND session_id=?",
                        (item_type, session_id)).fetchone()["n"])
                return int(self.mm.con.execute(
                    "SELECT COUNT(*) AS n FROM memory_items WHERE item_type=?", (item_type,)).fetchone()["n"])
            edges = int(self.mm.con.execute("SELECT COUNT(*) AS n FROM memory_edges").fetchone()["n"])
        return {"turn_nodes": count(NODE_TURN), "entity_nodes": count(NODE_ENTITY),
                "episode_nodes": count(NODE_EPISODE), "edges": edges}
