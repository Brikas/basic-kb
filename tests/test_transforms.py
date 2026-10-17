"""The transform stage (ADR 0006) and its first transform, redact-secrets.

Secret-shaped test values are assembled at runtime, so this file never contains a string a
push-protection scanner would take for a real credential.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from basic_kb.models import Chunk
from basic_kb.sources import build_source
from basic_kb.transforms import RedactSecrets, apply_transforms, build_transforms

R = RedactSecrets()
ALNUM = "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0"


def redact(text: str) -> str:
    return R.redact(text)[0]


@pytest.mark.parametrize("text, kind", [
    ("key is " + "sk-" + "ant-api03-" + ALNUM, "anthropic-key"),
    ("export OPENAI_KEY=" + "sk-" + "proj-" + ALNUM, "openai-key"),
    ("token " + "gh" + "p_" + ALNUM[:36], "github-token"),
    ("token " + "github" + "_pat_" + ALNUM + "_" + ALNUM, "github-token"),
    ("token " + "gl" + "pat-" + ALNUM[:20], "gitlab-token"),
    ("aws " + "AK" + "IA" + "Q3EGRIOSFODNN7EX", "aws-access-key"),
    ("slack " + "xo" + "xb-" + "1234567890-abcdefghij", "slack-token"),
    ("maps " + "AI" + "za" + ALNUM[:35], "google-api-key"),
    ("stripe " + "sk" + "_live_" + ALNUM[:24], "stripe-key"),
    ("hubspot " + "pat-" + "eu1-" + "0a1b2c3d-1234-4abc-8def-0123456789ab", "hubspot-token"),
    ("jwt " + "ey" + "JhbGciOiJIUzI1NiJ9." + "ey" + "JzdWIiOiIxMjM0NTY3ODkwIn0." + "dozjgNryP4J3jVmNHl0w5N", "jwt"),
    ("-----BEGIN " + "RSA PRIVATE KEY-----\nMIIEow\nabc\n-----END RSA PRIVATE KEY-----", "private-key"),
    ("curl -H 'Authorization: Bearer " + "abc123DEF456ghi789JKL" + "'", "bearer-token"),
    ("postgres://app:" + "s3cretPass" + "@db.internal:5432/x", "url-password"),
    ('DB_PASSWORD="' + "Tr0ub4dor-and-3" + '"', "assigned-secret"),
    ('{"apiKey": "' + "f3b9c2d1e8a7" + '"}', "assigned-secret"),
    ("DEEPINFRA_API_KEY=" + "Zx81kQpLmN27vBcR", "assigned-secret"),
    ("-H 'x-api-key: " + "9f8e7d6c5b4a3f2e" + "'", "assigned-secret"),
])
def test_redacts_each_kind(text, kind):
    out = redact(text)
    assert f"[REDACTED:{kind}]" in out


@pytest.mark.parametrize("text", [
    "commit 3f2a9c1d8e7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f",                 # git sha
    "id 0a1b2c3d-1234-4abc-8def-0123456789ab",                         # uuid
    "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "max_tokens: 512",
    "password: str",
    "token = self.token",
    "api_key=api_key",
    "TOKEN_FILE=/srv/x/token.txt",
    'API_KEY="${API_KEY}"',
    "api_key=os.environ['X']",
    "DEEPINFRA_API_KEY=<your-key-here>",
    "password: correcthorsebatterystaple",
    "MAX_TOKEN=4096",
    "the bearer token is in the env file",
    "pip install scikit-learn; see sk-learn-tutorial-for-beginners",
    "git@github.com:Brikas/temple1.git",
    "https://example.com/path:with:colons@x",
    "primary_key = record_identifier_value",
    "Those use a different token:\n\nAssistant tool calls: Bash Run2 tests",   # never across a line
    "apiKey: getApiKeyFromEnvironment",
    "Password: Something",
])
def test_leaves_lookalikes_alone(text):
    assert redact(text) == text


def test_private_key_header_without_end_masks_only_the_key_lines():
    text = ("It starts with -----BEGIN " + "OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAE\n"
            "and then the rest of the explanation survives.")
    out = redact(text)
    assert out == "It starts with [REDACTED:private-key]\nand then the rest of the explanation survives."


def test_redaction_is_idempotent_and_keeps_surrounding_text():
    text = "use DB_PASSWORD=" + "Tr0ub4dor-and-3" + " for the staging db"
    once = redact(text)
    assert once == "use DB_PASSWORD=[REDACTED:assigned-secret] for the staging db"
    assert redact(once) == once


def test_apply_counts_masks_in_metadata():
    chunk = Chunk(text="a " + "sk-" + "ant-" + ALNUM + " and DB_PASSWORD=" + "Tr0ub4dor-and-3",
                  metadata={"source": "s"}, id_suffix="0")
    out = R.apply(chunk)
    assert out.metadata["redactions"] == 2 and "Tr0ub4dor" not in out.text
    clean = Chunk(text="nothing here", metadata={}, id_suffix="1")
    assert R.apply(clean) is clean


def test_extra_patterns_from_config():
    (t,) = build_transforms([{"type": "redact-secrets", "extra_patterns": {"internal": r"ik_[a-z0-9]{12}"}}])
    assert t.redact("x ik_abcdef123456 y")[0] == "x [REDACTED:internal] y"


def test_unknown_transform_and_bad_spec_fail_loudly():
    with pytest.raises(ValueError, match="Unknown transform 'nope'"):
        build_transforms(["nope"])
    with pytest.raises(ValueError, match="mapping with a `type:`"):
        build_transforms([{"extra_patterns": {}}])
    with pytest.raises(ValueError, match="colour"):
        build_transforms([{"type": "redact-secrets", "colour": "red"}])


DROPPER = textwrap.dedent("""\
    from basic_kb.transforms import TransformBase


    class DropShort(TransformBase):
        name = "drop-short"

        def __init__(self, min_words=3):
            self.min_words = min_words

        def apply(self, chunk):
            return chunk if len(chunk.text.split()) >= self.min_words else None
    """)


def test_code_hook_by_import_path_can_drop_chunks(tmp_path: Path):
    module = f"dropper_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{module}.py").write_text(DROPPER, encoding="utf-8")
    (t,) = build_transforms([{"type": f"{module}:DropShort", "min_words": 2}], tmp_path)
    chunks = [Chunk(text="one", metadata={}, id_suffix="0"), Chunk(text="two words", metadata={}, id_suffix="1")]
    assert [c.text for c in apply_transforms([t], chunks)] == ["two words"]


def test_index_and_preview_embed_only_redacted_text(kb, instance):
    secret = "Tr0ub4dor-and-3"
    (instance / "data" / "notes" / "creds.md").write_text(
        f"# Staging\n\nThe staging database uses DB_PASSWORD={secret} until rotation next week.\n",
        encoding="utf-8")
    src = build_source({"id": "notes", "type": "markdown", "path": "data/notes", "chunker": "recursive",
                        "transforms": ["redact-secrets"]}, instance)
    preview = {f.rel_path: f for f in kb.preview(src, chunk_size=400, overlap=40, min_chunk=20)}
    assert "[REDACTED:assigned-secret]" in preview["creds.md"].chunks[0].text
    kb.index(src, chunk_size=400, overlap=40, min_chunk=20)
    stored = [m for m in kb.store.all_metadata("notes") if m.get("rel_path") == "creds.md"]
    assert stored and stored[0]["redactions"] == 1
    hits = kb.search([src], ["staging database rotation"], n=3)
    assert hits and all(secret not in h.doc for h in hits)
    assert any("[REDACTED:assigned-secret]" in h.doc for h in hits)
