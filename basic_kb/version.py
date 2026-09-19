"""Single source of the package version (pyproject reads it too), and the wire
compatibility floors a mixed-version client/server pair is checked against.

Bump a floor only when a change actually breaks the HTTP contract — a renamed or removed
response field, a newly required request field, a changed route. Adding an optional field
breaks nothing and needs no bump. Say what broke in the commit message: these two numbers
are what turns a confusing failure at some call site into one clear message at connect time.
"""
__version__ = "0.3.0"

MIN_CLIENT_VERSION = "0.3.0"   # oldest client this server will answer
MIN_SERVER_VERSION = "0.3.0"   # oldest server this client will talk to


def parse_version(value: str) -> tuple:
    """`"0.3.1"` -> `(0, 3, 1)`. Trailing non-numeric parts are ignored, so a pre-release
    compares equal to its release; close enough for a compatibility floor."""
    parts = []
    for chunk in str(value).split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)
