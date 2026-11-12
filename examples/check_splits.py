"""Check document-level splits using a synthetic manifest."""
from collections.abc import Iterable

PARTITIONS = {"train", "validation", "test"}


def assignments(rows: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Index a manifest by stable document ID and reject repeated IDs."""
    by_document: dict[str, str] = {}
    for document_id, partition in rows:
        if not document_id or partition not in PARTITIONS:
            raise ValueError(f"Invalid manifest row: {(document_id, partition)!r}")
        if document_id in by_document:
            previous = by_document[document_id]
            if previous != partition:
                raise ValueError(
                    f"{document_id!r} appears in both {previous!r} and {partition!r}"
                )
            raise ValueError(f"Duplicate manifest row for {document_id!r}")
        by_document[document_id] = partition
    return by_document


def check_shared_split(
    reference_rows: Iterable[tuple[str, str]],
    model_rows: Iterable[tuple[str, str]],
) -> None:
    """Require identical document coverage and partitions for one split seed."""
    reference = assignments(reference_rows)
    actual = assignments(model_rows)
    missing = reference.keys() - actual.keys()
    extra = actual.keys() - reference.keys()
    moved = {
        document_id
        for document_id in reference.keys() & actual.keys()
        if reference[document_id] != actual[document_id]
    }
    if missing or extra or moved:
        raise ValueError(
            f"Split mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}, reassigned={sorted(moved)}"
        )


if __name__ == "__main__":
    shared_manifest = [
        ("doc-001", "train"),
        ("doc-002", "validation"),
        ("doc-003", "test"),
    ]
    # Different data loaders may return the same documents in different orders.
    check_shared_split(shared_manifest, reversed(shared_manifest))
    print("Split check passed for all three synthetic documents.")

