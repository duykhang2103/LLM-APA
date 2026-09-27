"""Load raw challenge records and resolve source-code file references.

Expected future flow:

1. Read challenge JSON from a private path.
2. Validate required metadata and the ``code_file`` field.
3. Resolve the source path below the configured private data root.
4. Read C++ text into ``code`` without copying it into tracked files.
5. Return records matching :mod:`llm_grading.data.schema`.

Use ``examples/normalized_sample.json`` as a safe shape example. It is not
an instruction to commit or distribute real submissions.
"""

from pathlib import Path

from .schema import NormalizedSample


def load_samples(source: str | Path) -> list[NormalizedSample]:
    """Return normalized samples from a private JSON/data directory.

    The implementation should also report missing files, duplicate IDs, and
    malformed records in a data-audit report instead of silently dropping them.
    """
    # TODO: Read the official JSON files and resolve submissions/<problem>/<file>.
    raise NotImplementedError("Dataset loading is not implemented in the scaffold.")
