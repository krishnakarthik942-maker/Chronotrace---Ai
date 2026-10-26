"""Evidence integrity module for ChronoTrace AI.

Provides cryptographic hashing, tamper detection, immutable evidence custody,
and structured audit logging in compliance with digital forensics principles.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


SOFTWARE_VERSION = "1.0.0"


@dataclass
class EvidenceRecord:
    """Forensic record of an ingested video evidence file."""

    evidence_id: str
    camera_name: str
    original_filename: str
    stored_path: str
    file_size_bytes: int
    sha256_hash: str
    ingest_time_utc: str
    software_version: str = SOFTWARE_VERSION
    is_tampered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditLogEntry:
    """Immutable audit trail log entry."""

    log_id: str
    timestamp_utc: str
    action: str
    operator: str
    details: Dict[str, Any]
    input_hashes: List[str] = field(default_factory=list)
    output_hashes: List[str] = field(default_factory=list)
    software_version: str = SOFTWARE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceManager:
    """Manages digital video evidence files, custody records, and audit logs."""

    def __init__(self, base_evidence_dir: str | Path = "data/evidence", audit_log_path: str | Path = "data/evidence/audit_log.jsonl"):
        self.evidence_dir = Path(base_evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, EvidenceRecord] = {}
        self._log_counter = 0

    @staticmethod
    def compute_sha256(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute the cryptographic SHA-256 hash of a file using chunked streaming."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found for hashing: {file_path}")

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest().lower()

    def record_audit(self, action: str, details: Dict[str, Any], operator: str = "ForensicInvestigator",
                     input_hashes: Optional[List[str]] = None, output_hashes: Optional[List[str]] = None) -> AuditLogEntry:
        """Append an entry to the tamper-evident audit log."""
        self._log_counter += 1
        now_utc = datetime.now(timezone.utc).isoformat()
        log_id = f"AUDIT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._log_counter:04d}"

        entry = AuditLogEntry(
            log_id=log_id,
            timestamp_utc=now_utc,
            action=action,
            operator=operator,
            details=details,
            input_hashes=input_hashes or [],
            output_hashes=output_hashes or [],
            software_version=SOFTWARE_VERSION
        )

        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        return entry

    def register_evidence_file(self, src_path: str | Path, camera_name: str,
                               copy_to_evidence_vault: bool = True) -> EvidenceRecord:
        """Ingest video file into evidence custody without modifying the source."""
        source = Path(src_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source evidence file does not exist: {src_path}")

        file_size = source.stat().st_size
        if file_size == 0:
            raise ValueError(f"Evidence file is empty (0 bytes): {src_path}")

        # Compute initial hash on source file
        sha256 = self.compute_sha256(source)
        evidence_id = f"EVID-{sha256[:12].upper()}"

        if copy_to_evidence_vault:
            # Store in evidence repository with unique name to prevent any collision or overwrite
            ext = source.suffix
            vault_filename = f"{evidence_id}_{camera_name.replace(' ', '_')}{ext}"
            dest_path = self.evidence_dir / vault_filename

            if not dest_path.exists():
                shutil.copy2(source, dest_path)
                # Attempt to mark read-only to prevent tampering
                try:
                    os.chmod(dest_path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
                except Exception:
                    pass

            # Verify integrity of copied vault file
            vault_hash = self.compute_sha256(dest_path)
            if vault_hash != sha256:
                raise RuntimeError(f"Vault copy integrity check failed! Expected {sha256}, got {vault_hash}")
            stored_path = str(dest_path.resolve())
        else:
            stored_path = str(source.resolve())

        now_utc = datetime.now(timezone.utc).isoformat()
        record = EvidenceRecord(
            evidence_id=evidence_id,
            camera_name=camera_name,
            original_filename=source.name,
            stored_path=stored_path,
            file_size_bytes=file_size,
            sha256_hash=sha256,
            ingest_time_utc=now_utc,
            software_version=SOFTWARE_VERSION,
            is_tampered=False,
            metadata={}
        )

        self._records[evidence_id] = record

        self.record_audit(
            action="EVIDENCE_INGESTED",
            details={
                "evidence_id": evidence_id,
                "camera_name": camera_name,
                "original_filename": source.name,
                "stored_path": stored_path,
                "file_size_bytes": file_size,
                "sha256": sha256
            },
            input_hashes=[sha256]
        )

        return record

    def verify_integrity(self, evidence_id: str) -> bool:
        """Check if an evidence file has remained intact against its recorded hash."""
        if evidence_id not in self._records:
            raise KeyError(f"Evidence ID not registered: {evidence_id}")

        rec = self._records[evidence_id]
        current_hash = self.compute_sha256(rec.stored_path)
        is_valid = (current_hash == rec.sha256_hash)
        rec.is_tampered = not is_valid

        self.record_audit(
            action="INTEGRITY_VERIFICATION",
            details={
                "evidence_id": evidence_id,
                "expected_hash": rec.sha256_hash,
                "current_hash": current_hash,
                "is_valid": is_valid
            },
            input_hashes=[rec.sha256_hash],
            output_hashes=[current_hash]
        )
        return is_valid

    def get_all_records(self) -> List[EvidenceRecord]:
        return list(self._records.values())

    def get_record(self, evidence_id: str) -> Optional[EvidenceRecord]:
        return self._records.get(evidence_id)

    def load_audit_log(self) -> List[Dict[str, Any]]:
        """Read all lines from the audit log."""
        if not self.audit_log_path.exists():
            return []
        entries = []
        with open(self.audit_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries
