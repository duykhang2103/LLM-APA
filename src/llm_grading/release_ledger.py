"""Durable inference results; ambiguous calls must never be silently regenerated."""

import hashlib
import json
import os
from pathlib import Path


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class ReleaseLedger:
    def __init__(self, root: Path, contract: dict, job_ids: list[str]):
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("Duplicate jobs")
        self.root, self.contract, self.jobs = root, fingerprint(contract), job_ids
        root.mkdir(parents=True, exist_ok=True)
        self.header = {"contract_sha256": self.contract, "job_ids": job_ids}
        header = root / "contract.json"
        if header.exists():
            if json.loads(header.read_text()) != self.header:
                raise ValueError("Recovery contract differs")
        else:
            if any(root.iterdir()):
                raise ValueError("Unbound existing ledger")
            atomic_json(header, self.header)
        self.rows = []
        for job in job_ids:
            path = self.path(job)
            if not path.exists():
                break
            item = json.loads(path.read_text())
            payload = item["payload"]
            if (
                item["sha256"] != fingerprint(payload)
                or payload["contract_sha256"] != self.contract
                or payload["job_id"] != job
            ):
                raise ValueError("Recovery result integrity differs")
            self.rows.append(payload)
        if any(self.path(j).exists() for j in job_ids[len(self.rows) :]):
            raise ValueError("Recovery ledger is not a complete prefix")
        active = root / "active-call.json"
        if active.exists():
            value = json.loads(active.read_text())
            if value is not None and value["job_id"] not in {
                r["job_id"] for r in self.rows
            }:
                raise ValueError(
                    "Ambiguous unfinished model call; stop, preserve evidence"
                )
        atomic_json(active, None)
        self.write_index()

    def path(self, job_id: str) -> Path:
        return self.root / (
            "call-" + hashlib.sha256(job_id.encode()).hexdigest() + ".json"
        )

    def start(self, job_id: str) -> None:
        if job_id != self.jobs[len(self.rows)]:
            raise ValueError("Unexpected or duplicate inference call")
        atomic_json(
            self.root / "active-call.json",
            {"job_id": job_id, "contract_sha256": self.contract},
        )

    def complete(self, job_id: str, row: dict) -> None:
        if job_id != self.jobs[len(self.rows)] or self.path(job_id).exists():
            raise ValueError("Refuse duplicate prediction overwrite")
        active = json.loads((self.root / "active-call.json").read_text())
        if active != {"job_id": job_id, "contract_sha256": self.contract}:
            raise ValueError("Result has no matching started call")
        if set(row) & {"job_id", "contract_sha256"}:
            raise ValueError("Reserved result fields")
        payload = {"job_id": job_id, "contract_sha256": self.contract, **row}
        atomic_json(
            self.path(job_id), {"payload": payload, "sha256": fingerprint(payload)}
        )
        self.rows.append(payload)
        self.write_index()
        atomic_json(self.root / "active-call.json", None)

    def write_index(self) -> None:
        atomic_json(
            self.root / "index.json",
            {
                "contract_sha256": self.contract,
                "completed": [
                    {"job_id": r["job_id"], "sha256": fingerprint(r)} for r in self.rows
                ],
            },
        )
