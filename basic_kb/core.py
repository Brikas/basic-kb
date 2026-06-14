"""KnowledgeBase — orchestrates indexing and semantic search over DataSources."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from .chunkers import DEFAULT_CHUNK_SIZE, DEFAULT_MIN_CHUNK, DEFAULT_OVERLAP
from .embedders import EmbedderBase
from .models import SearchResult
from .rerankers import RerankerBase
from .sources import DataSourceBase

DEFAULT_N = 15


class KnowledgeBase:
    """
    Orchestrates indexing and semantic search across one or more DataSources.

    Each DataSource maps to its own ChromaDB collection (named by source_id).
    Searching multiple sources merges and re-ranks results across collections.
    """

    def __init__(
        self,
        embedder: EmbedderBase,
        chroma_dir: Path,
        reranker: Optional[RerankerBase] = None,
    ) -> None:
        self.embedder = embedder
        self.chroma_dir = chroma_dir
        self.reranker = reranker

    def _client(self):
        import chromadb
        return chromadb.PersistentClient(path=str(self.chroma_dir))

    def _get_collection(self, client, collection_name: str):
        """Get or create a named ChromaDB collection, clearing on EF conflict."""
        ef = self.embedder.as_chroma_ef()
        try:
            return client.get_or_create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError as exc:
            if "embedding function conflict" not in str(exc).lower():
                raise
            print(f"Note: embedding function changed for '{collection_name}'. Clearing old index.")
            client.delete_collection(collection_name)
            return client.create_collection(
                name=collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )

    def index(
        self,
        source: DataSourceBase,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        min_chunk: int = DEFAULT_MIN_CHUNK,
        force: bool = False,
    ) -> None:
        """Index all files from a DataSource into its own ChromaDB collection."""
        files = source.get_files()
        if not files:
            print(f"No files found for source '{source.label}'. Nothing to index.", file=sys.stderr)
            return

        client = self._client()
        if force:
            try:
                client.delete_collection(source.source_id)
                print(f"Cleared existing index for '{source.source_id}'.")
            except Exception:
                pass

        col = self._get_collection(client, source.source_id)
        existing_ids: set[str] = set(col.get(include=[])["ids"])

        chunker = source.make_chunker(chunk_size, overlap, min_chunk)
        print(
            f"Indexing {len(files)} {source.label} files  "
            f"(model={self.embedder.model_id}, chunker={chunker.name}, "
            f"chunk={chunk_size}, overlap={overlap})"
        )

        added = skipped = empty = 0

        for f in files:
            doc = source.parse_file(f)
            if doc is None:
                empty += 1
                continue

            chunks = chunker.chunk(doc)

            new_ids, new_docs, new_metas = [], [], []
            for chunk in chunks:
                chunk_id = f"{doc.id}::{chunk.id_suffix}"
                if chunk_id in existing_ids:
                    continue
                new_ids.append(chunk_id)
                new_docs.append(chunk.text)
                new_metas.append(chunk.metadata)

            if not new_ids:
                skipped += 1
                continue

            col.add(documents=new_docs, metadatas=new_metas, ids=new_ids)
            added += len(new_ids)
            print(f"  {f.name}  ({len(new_ids)} chunks)")

        total = col.count()
        print(
            f"\nDone [{source.source_id}]. "
            f"added={added} | skipped={skipped} already-indexed | "
            f"empty={empty} no-content | total={total}"
        )

    def search(
        self,
        sources: list[DataSourceBase],
        queries: list[str],
        n: int = DEFAULT_N,
        content_type_filter: Optional[str] = None,
        rerank_candidates: Optional[int] = None,
        strict_rerank: bool = False,
    ) -> list[SearchResult]:
        """
        Multi-source, multi-query semantic search.

        Each query is run against every source collection independently.
        Results are merged by score (highest wins per unique chunk ID),
        then optionally reranked with Jina.
        """
        if not self.chroma_dir.exists():
            print("No index found. Run: python -m basic_kb index")
            return []

        client = self._client()
        best: dict[str, SearchResult] = {}

        if self.reranker and rerank_candidates is None:
            rerank_candidates = min(n * 3, 50)
        fetch_n = (rerank_candidates or n) if self.reranker else n

        for source in sources:
            try:
                col = client.get_collection(
                    name=source.source_id,
                    embedding_function=self.embedder.as_chroma_ef(),
                )
            except Exception:
                print(
                    f"  Note: no index for '{source.source_id}' — "
                    f"run: python -m basic_kb index --source {source.source_id}",
                    file=sys.stderr,
                )
                continue

            where: Optional[dict] = None
            if content_type_filter:
                where = {"content_type": {"$eq": content_type_filter}}

            for q in queries:
                kwargs: dict = {
                    "query_embeddings": self.embedder.query_embed([q]),
                    "n_results": fetch_n,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    kwargs["where"] = where
                try:
                    results = col.query(**kwargs)
                except Exception as e:
                    print(f"  Query error on '{source.source_id}': {e}", file=sys.stderr)
                    continue

                for doc_id, doc, meta, dist in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    full_id = f"{source.source_id}::{doc_id}"
                    score = 1.0 - dist
                    if full_id not in best or score > best[full_id].score:
                        best[full_id] = SearchResult(doc=doc, metadata=meta, score=score)

        hits = sorted(best.values(), key=lambda r: r.score, reverse=True)

        if self.reranker and hits:
            candidates = hits[: rerank_candidates or n]
            try:
                hits = self.reranker.rerank(queries[0], candidates, top_n=n)
            except Exception as e:
                if strict_rerank:
                    raise
                print(f"Warning: reranking failed, falling back to cosine scores ({e})", file=sys.stderr)
                hits = hits[:n]
        else:
            hits = hits[:n]

        return hits

    def status(self, sources: list[DataSourceBase]) -> None:
        """Print index stats for one or more sources."""
        if not self.chroma_dir.exists():
            print("No index found. Run: python -m basic_kb index")
            return

        client = self._client()

        for source in sources:
            print(f"\n{'='*55}")
            print(f"Source  : {source.label}  ({source.source_id})")
            print(f"ChromaDB: {self.chroma_dir}")
            print(f"Model   : {self.embedder.model_id}")

            try:
                col = client.get_collection(
                    name=source.source_id,
                    embedding_function=self.embedder.as_chroma_ef(),
                )
            except Exception:
                print(f"  No index found. Run: python -m basic_kb index --source {source.source_id}")
                continue

            total_chunks = col.count()
            if total_chunks == 0:
                print(f"  Index empty. Run: python -m basic_kb index --source {source.source_id}")
                continue

            all_metas: list[dict] = col.get(include=["metadatas"])["metadatas"]
            indexed_files: set[str] = {m["file"] for m in all_metas}
            files_on_disk = len(source.get_files())
            unindexed = files_on_disk - len(indexed_files)

            print(f"Chunks  : {total_chunks:,}")
            print(f"Docs    : {len(indexed_files):,} indexed / {files_on_disk} on disk  ({unindexed} not indexed)")

            dates = sorted(
                m.get("date", "unknown") for m in all_metas
                if m.get("date", "unknown") != "unknown"
            )
            if dates:
                print(f"Dates   : {dates[0]} → {dates[-1]}")

            # Content-type breakdown — shown for any source whose chunks carry a
            # real content_type (e.g. markdown sources with frontmatter).
            has_content_type = any(
                str(m.get("content_type", "unknown")) != "unknown" for m in all_metas
            )
            if has_content_type:
                ct_counts: dict[str, int] = {}
                for m in all_metas:
                    ct = str(m.get("content_type", "unknown"))
                    ct_counts[ct] = ct_counts.get(ct, 0) + 1
                for ct, count in sorted(ct_counts.items()):
                    print(f"  {ct}: {count:,} chunks")

            oversized_count = sum(1 for m in all_metas if m.get("oversized"))
            if oversized_count:
                print(f"  ⚠  oversized chunks: {oversized_count}")
