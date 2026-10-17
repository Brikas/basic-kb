"""Transforms: a per-source stage between chunking and embedding (ADR 0006).

A transform sees each chunk after the chunker and before the embedder, and returns the chunk
(possibly rewritten) or None to drop it. Configured per source as a list, applied in order:

    transforms:
      - redact-secrets                        # a built-in, by name
      - type: redact-secrets                  # the same, with options
        extra_patterns: {"internal-key": "ik_[a-z0-9]{24}"}
      - type: mypackage.cleaning:StripProps    # code hook: an import path (ADR 0005)
        some_option: 3

Chunk ids derive from chunk text, so enabling or changing a transform changes the ids of
every chunk it touches and those chunks re-embed. That is intended: a stale chunk would keep
an unredacted vector alive.

Built-in transforms:
  redact-secrets  masks credentials as `[REDACTED:kind]` so the rest of the chunk stays searchable
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from .models import Chunk
from .plugins import is_import_path, load_import_path


class TransformBase(ABC):
    """Rewrites or drops one chunk. Options from config arrive as constructor kwargs."""

    name: str = ""
    # Bump whenever a code change alters this transform's output; the source's pipeline
    # fingerprint includes it, so the next index run re-chunks every file.
    version: str = "1"

    @abstractmethod
    def apply(self, chunk: Chunk) -> Optional[Chunk]:
        """Return the chunk to embed (the same object, or a rewritten copy), or None to drop it."""


def apply_transforms(transforms: Iterable[TransformBase], chunks: list[Chunk]) -> list[Chunk]:
    """Run every transform over every chunk, in order. A dropped chunk skips the rest."""
    transforms = tuple(transforms)
    if not transforms:
        return chunks
    out: list[Chunk] = []
    for chunk in chunks:
        current: Optional[Chunk] = chunk
        for t in transforms:
            current = t.apply(current)
            if current is None:
                break
        if current is not None:
            out.append(current)
    return out


# ---------------------------------------------------------------------------
# redact-secrets
# ---------------------------------------------------------------------------

# Provider rules: a value shape specific enough that a match is a secret wherever it appears.
# Ordered most specific first, since an earlier mask hides its span from later rules.
# The `(?=[^\s]*\d)` lookaheads require a digit, which keeps slugs such as
# "sk-learn-tutorial-for-beginners" from matching a key rule by accident.
_PROVIDER_RULES: list[tuple[str, str]] = [
    # A whole block when its END line is present; otherwise the BEGIN line and the base64
    # lines under it, so a sentence merely naming the header keeps the text after it.
    ("private-key", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----(?:[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
                    r"|(?:[ \t]*\n[ \t]*[A-Za-z0-9+/=]{16,})*)"),
    ("anthropic-key", r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    ("openai-key", r"\bsk-(?:proj-|svcacct-|admin-)?(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{20,}"),
    ("github-token", r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    ("gitlab-token", r"\bglpat-[A-Za-z0-9_-]{20,}"),
    ("aws-access-key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("slack-token", r"\bxox[abposr]-(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{10,}"),
    ("google-api-key", r"\bAIza[0-9A-Za-z_-]{35}"),
    ("stripe-key", r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{20,}"),
    ("hubspot-token", r"\bpat-(?:na|eu|ap)\d-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
]

# `Authorization: Bearer <token>` and friends. Only the token is masked.
_BEARER = re.compile(r"(?i)(\bbearer\s+)((?=[A-Za-z0-9._~+/=-]*\d)[A-Za-z0-9._~+/=-]{16,})")

# Credentials inside a URL, `scheme://user:password@host`. Only the password is masked.
_URL_CREDENTIALS = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s/@]{3,})(@)")

# Key-name rule: a value assigned to a name that says it is a secret, e.g.
# `DB_PASSWORD="…"`, `"apiKey": "…"`, `x-api-key: …`. Provider rules miss these, because
# the value has no recognisable shape; the name is the only evidence. Endings are specific
# (`api_key`, never bare `key`), so `primary_key = id` or `max_tokens: 512` never match.
_KEY_NAME = re.compile(
    r"""(?ix)
    (?<![A-Za-z0-9_])
    (?P<key>[A-Za-z0-9_.-]*?
      (?:api[_-]?key|access[_-]?key|secret[_-]?key|private[_-]?key|client[_-]?secret
        |secret|token|password|passwd|pwd))
    ["']?[ \t]*(?::|=|=>)[ \t]*
    (?P<quote>["'`]?)
    (?P<value>[^\s"'`,;()<>{}\[\]]{8,})
    """)

# Values a key-name match must not mask: references, placeholders and code, never secrets.
_PLACEHOLDER = re.compile(
    r"(?i)^(?:\$|%|@|/|\./|~|https?:)|redacted|example|your[_-]|changeme|placeholder|xxxx|\*\*\*|\.\.\."
    r"|^(?:none|null|true|false|undefined|required|optional|string|secret|password|token)$")


def _looks_like_secret(value: str) -> bool:
    """A key-name value is masked only when it is not an obvious placeholder and mixes
    character classes the way a generated credential does: letters plus a digit or a
    symbol. A word (`password: Something`), a number (`MAX_TOKEN=4096`) or an identifier
    (`token = self.token`, `apiKey: getApiKeyFromEnv`) stays, since `.`, `_` and `-` do
    not count as symbols."""
    if _PLACEHOLDER.search(value):
        return False
    letters = any(c.isalpha() for c in value)
    digit = any(c.isdigit() for c in value)
    symbol = any(not c.isalnum() and c not in "._-" for c in value)
    return letters and (digit or symbol)


class RedactSecrets(TransformBase):
    """Masks credentials in chunk text as `[REDACTED:kind]`.

    Masking rather than dropping keeps the chunk searchable for everything around the
    secret, and the typed placeholder keeps the sentence readable. The count of masks goes
    into the chunk's metadata as `redactions`, so a status pass can see what happened.

    Simple by design: provider shapes, Bearer tokens, URL credentials and a key-name rule.
    No entropy catch-all — it would mask hashes, UUIDs and base64 by the hundred.
    `extra_patterns` adds `{kind: regex}` provider rules from config.
    """

    name = "redact-secrets"

    def __init__(self, extra_patterns: Optional[dict] = None, key_names: bool = True) -> None:
        rules = list(_PROVIDER_RULES) + [(str(k), str(v)) for k, v in (extra_patterns or {}).items()]
        self._rules = [(kind, re.compile(pattern)) for kind, pattern in rules]
        self._key_names = bool(key_names)

    def redact(self, text: str) -> tuple[str, int]:
        """Return (masked text, number of masks). Idempotent: a placeholder never re-matches."""
        count = 0
        for kind, rx in self._rules:
            text, n = rx.subn(f"[REDACTED:{kind}]", text)
            count += n
        text, n = _BEARER.subn(lambda m: f"{m.group(1)}[REDACTED:bearer-token]", text)
        count += n
        text, n = _URL_CREDENTIALS.subn(lambda m: f"{m.group(1)}[REDACTED:url-password]{m.group(3)}", text)
        count += n
        if self._key_names:
            hits = 0

            def mask(m: re.Match) -> str:
                nonlocal hits
                if not _looks_like_secret(m.group("value")):
                    return m.group(0)
                hits += 1
                start = m.start("value") - m.start(0)
                return m.group(0)[:start] + "[REDACTED:assigned-secret]"

            text = _KEY_NAME.sub(mask, text)
            count += hits
        return text, count

    def apply(self, chunk: Chunk) -> Optional[Chunk]:
        text, count = self.redact(chunk.text)
        if not count:
            return chunk
        meta = {**chunk.metadata, "redactions": int(chunk.metadata.get("redactions", 0)) + count}
        return replace(chunk, text=text, metadata=meta)


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------

TRANSFORM_TYPES: dict[str, type[TransformBase]] = {
    cls.name: cls for cls in (RedactSecrets,)
}


def build_transforms(specs: list, base_dir: Optional[Path] = None) -> list[TransformBase]:
    """Build a source's `transforms:` list. Each item is a name, or a mapping whose `type`
    is a name or an import path and whose other keys are constructor options."""
    out: list[TransformBase] = []
    for spec in specs or []:
        if isinstance(spec, str):
            type_name, options = spec, {}
        elif isinstance(spec, dict) and spec.get("type"):
            type_name = str(spec["type"])
            options = {k: v for k, v in spec.items() if k != "type"}
        else:
            raise ValueError(f"Transform {spec!r}: give a name, or a mapping with a `type:`.")
        if is_import_path(type_name):
            cls = load_import_path(type_name, base_dir, what="Transform")
            if not (isinstance(cls, type) and issubclass(cls, TransformBase)):
                raise ValueError(f"Transform {type_name!r} is not a subclass of basic_kb TransformBase.")
        else:
            cls = TRANSFORM_TYPES.get(type_name)
            if cls is None:
                raise ValueError(
                    f"Unknown transform {type_name!r}. Options: {', '.join(sorted(TRANSFORM_TYPES))}, "
                    f"or an import path 'package.module:ClassName'.")
        try:
            out.append(cls(**options))
        except TypeError as e:
            raise ValueError(f"Transform {type_name!r}: {e}") from e
    return out
