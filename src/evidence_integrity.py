"""Evidence integrity module for ChronoTrace AI.

Provides cryptographic hashing, tamper detection, immutable evidence custody,
and structured audit logging in compliance with digital forensics principles.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    """Immutable, cryptographically hash-chained audit trail log entry."""

    log_id: str
    timestamp_utc: str
    action: str
    operator: str
    details: Dict[str, Any]
    input_hashes: List[str] = field(default_factory=list)
    output_hashes: List[str] = field(default_factory=list)
    software_version: str = SOFTWARE_VERSION
    prev_entry_hash: str = ""
    entry_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceManager:
    """Manages digital video evidence files, custody records, and audit logs."""

    def __init__(
        self,
        base_evidence_dir: str | Path = "data/evidence",
        audit_log_path: str | Path = "data/evidence/audit_log.jsonl",
        temp_upload_dir: Optional[str | Path] = None
    ):
        self.evidence_dir = Path(base_evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_upload_dir = Path(temp_upload_dir) if temp_upload_dir else self.evidence_dir / "temp_uploads"
        self._records: Dict[str, EvidenceRecord] = {}
        self._log_counter = 0
        self._audit_lock = threading.Lock()

    @staticmethod
    def sanitize_path_component(name: str, default: str = "item") -> str:
        """Sanitize filenames and camera names against path traversal and illegal filesystem characters."""
        base = os.path.basename(str(name).strip())
        clean = re.sub(r'[\x00-\x1f\\/:*?"<>|]', '_', base)
        clean = re.sub(r'_+', '_', clean).strip(' ._')
        if not clean:
            clean = f"{default}_{uuid.uuid4().hex[:6]}"
        return clean

    @staticmethod
    def unlock_file(file_path: str | Path) -> None:
        """Clear read-only attribute on file to permit clean removal/teardown on Windows/POSIX."""
        path = Path(file_path)
        if path.exists():
            try:
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            except Exception:
                pass

    @staticmethod
    def set_read_only(file_path: str | Path) -> None:
        """Mark vaulted evidence file read-only to prevent accidental modification."""
        path = Path(file_path)
        if path.exists():
            try:
                os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            except Exception:
                pass

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

    @staticmethod
    def compute_entry_hash(
        log_id: str,
        timestamp_utc: str,
        action: str,
        operator: str,
        details: Dict[str, Any],
        input_hashes: List[str],
        output_hashes: List[str],
        software_version: str,
        prev_entry_hash: str
    ) -> str:
        """Compute deterministic SHA-256 hash of an audit log entry for cryptographic chaining."""
        payload = {
            "log_id": log_id,
            "timestamp_utc": timestamp_utc,
            "action": action,
            "operator": operator,
            "details": details,
            "input_hashes": sorted(input_hashes),
            "output_hashes": sorted(output_hashes),
            "software_version": software_version,
            "prev_entry_hash": prev_entry_hash
        }
        canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()

    def record_audit(self, action: str, details: Dict[str, Any], operator: str = "ForensicInvestigator",
                     input_hashes: Optional[List[str]] = None, output_hashes: Optional[List[str]] = None) -> AuditLogEntry:
        """Append an entry to the cryptographically hash-chained audit log with thread-safe locking."""
        with self._audit_lock:
            self._log_counter += 1
            now_utc = datetime.now(timezone.utc).isoformat()
            log_id = f"AUDIT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._log_counter:04d}"

            # Retrieve previous entry hash from last line of audit log on disk
            prev_hash = "0" * 64
            if self.audit_log_path.exists() and self.audit_log_path.stat().st_size > 0:
                try:
                    with open(self.audit_log_path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if lines:
                            last_entry = json.loads(lines[-1])
                            prev_hash = last_entry.get("entry_hash") or "0" * 64
                except Exception:
                    prev_hash = "0" * 64

            in_hashes = input_hashes or []
            out_hashes = output_hashes or []

            entry_hash = self.compute_entry_hash(
                log_id=log_id,
                timestamp_utc=now_utc,
                action=action,
                operator=operator,
                details=details,
                input_hashes=in_hashes,
                output_hashes=out_hashes,
                software_version=SOFTWARE_VERSION,
                prev_entry_hash=prev_hash
            )

            entry = AuditLogEntry(
                log_id=log_id,
                timestamp_utc=now_utc,
                action=action,
                operator=operator,
                details=details,
                input_hashes=in_hashes,
                output_hashes=out_hashes,
                software_version=SOFTWARE_VERSION,
                prev_entry_hash=prev_hash,
                entry_hash=entry_hash
            )

            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")

            return entry

    def verify_audit_log_integrity(
        self,
        expected_count: Optional[int] = None,
        expected_tail_hash: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Verify the cryptographic hash-chain integrity of the append-only audit trail."""
        if not self.audit_log_path.exists() or self.audit_log_path.stat().st_size == 0:
            if expected_count and expected_count > 0:
                return False, f"Audit log is empty but expected {expected_count} entries (log was deleted/truncated)."
            return True, "Audit log is empty (no entries to verify)."

        entries = []
        with open(self.audit_log_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError as ex:
                    return False, f"Audit log contains malformed, corrupted, or truncated JSON at line {line_no}: {ex}"

        if not entries:
            if expected_count and expected_count > 0:
                return False, f"Audit log contains 0 valid entries but expected {expected_count}."
            return True, "Audit log contains zero valid entries."

        if expected_count is not None and len(entries) != expected_count:
            return False, f"Audit log entry count mismatch: expected {expected_count}, got {len(entries)} (log has been truncated or appended)."

        expected_prev = "0" * 64
        for idx, entry_dict in enumerate(entries):
            actual_prev = entry_dict.get("prev_entry_hash", "")
            actual_hash = entry_dict.get("entry_hash", "")

            if not actual_hash:
                return False, f"Entry {idx} ({entry_dict.get('log_id')}) lacks cryptographic entry_hash."

            if actual_prev != expected_prev:
                return False, f"Hash-chain broken at entry {idx} ({entry_dict.get('log_id')}): expected prev_entry_hash '{expected_prev}', got '{actual_prev}'."

            recomputed_hash = self.compute_entry_hash(
                log_id=entry_dict["log_id"],
                timestamp_utc=entry_dict["timestamp_utc"],
                action=entry_dict["action"],
                operator=entry_dict["operator"],
                details=entry_dict["details"],
                input_hashes=entry_dict.get("input_hashes", []),
                output_hashes=entry_dict.get("output_hashes", []),
                software_version=entry_dict.get("software_version", SOFTWARE_VERSION),
                prev_entry_hash=actual_prev
            )

            if recomputed_hash != actual_hash:
                return False, f"Entry hash mismatch at entry {idx} ({entry_dict.get('log_id')}): entry content has been modified or tampered with."

            expected_prev = actual_hash

        if expected_tail_hash is not None:
            last_hash = entries[-1].get("entry_hash", "")
            if last_hash != expected_tail_hash:
                return False, f"Audit log tail hash mismatch: expected '{expected_tail_hash}', got '{last_hash}' (log has been truncated or altered)."

        return True, f"Cryptographic audit log hash-chain verified intact ({len(entries)} entries)."

    def register_evidence_file(self, src_path: str | Path, camera_name: str,
                               copy_to_evidence_vault: bool = True,
                               original_filename: Optional[str] = None,
                               operator: str = "ForensicInvestigator") -> EvidenceRecord:
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
        display_name = original_filename if original_filename else source.name
        safe_display = self.sanitize_path_component(display_name, "video")
        clean_cam = self.sanitize_path_component(camera_name, "cam")

        if copy_to_evidence_vault:
            # Store in evidence repository with unique name to prevent any collision or overwrite
            ext = Path(safe_display).suffix or source.suffix
            vault_filename = f"{evidence_id}_{clean_cam}{ext}"
            dest_path = self.evidence_dir / vault_filename

            # Detect and resolve collisions safely: never silently overwrite evidence
            if dest_path.exists():
                existing_hash = self.compute_sha256(dest_path)
                if existing_hash != sha256:
                    # Naming collision with different content! Disambiguate without overwriting
                    counter = 1
                    while dest_path.exists() and self.compute_sha256(dest_path) != sha256:
                        dest_path = self.evidence_dir / f"{evidence_id}_{clean_cam}_col{counter}{ext}"
                        counter += 1

            if not dest_path.exists():
                try:
                    shutil.copy2(source, dest_path)
                    # Verify integrity of copied vault file
                    vault_hash = self.compute_sha256(dest_path)
                    if vault_hash != sha256:
                        self.unlock_file(dest_path)
                        dest_path.unlink(missing_ok=True)
                        raise RuntimeError(f"Vault copy integrity check failed! Expected {sha256}, got {vault_hash}")
                    self.set_read_only(dest_path)
                except Exception:
                    # Guarantee partial file cleanup on failed copy
                    if dest_path.exists():
                        try:
                            self.unlock_file(dest_path)
                            dest_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    raise
            stored_path = str(dest_path.resolve())
        else:
            stored_path = str(source.resolve())

        # Collision check on in-memory evidence_id registry: ensure different cameras never overwrite each other
        candidate_id = evidence_id
        counter = 1
        while candidate_id in self._records and (
            self._records[candidate_id].sha256_hash != sha256 or
            self._records[candidate_id].camera_name != camera_name
        ):
            candidate_id = f"{evidence_id}_{counter}"
            counter += 1
        evidence_id = candidate_id

        now_utc = datetime.now(timezone.utc).isoformat()
        record = EvidenceRecord(
            evidence_id=evidence_id,
            camera_name=camera_name,
            original_filename=display_name,
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
                "original_filename": display_name,
                "stored_path": stored_path,
                "file_size_bytes": file_size,
                "sha256": sha256
            },
            operator=operator,
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

    @staticmethod
    def is_valid_video_container(data: bytes) -> bool:
        """Validate whether bytes correspond to a supported video container (MP4/MOV, AVI, MKV).

        Handles ISO BMFF and QuickTime files regardless of where moov/mdat atoms appear,
        including faststart files, non-faststart files where moov is at EOF, and QuickTime
        files starting with 'wide' or 'free' atoms.
        """
        if not data or len(data) < 8:
            return False

        # 1. AVI container (RIFF....AVI )
        if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"AVI ":
            return True

        # 2. MKV / WebM container (EBML Header \x1a\x45\xdf\xa3)
        if data.startswith(b"\x1a\x45\xdf\xa3"):
            return True

        # 3. ISO Base Media File Format (MP4) / QuickTime (MOV)
        known_atoms = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot"}

        # Quick check: does any essential atom appear in the leading bytes?
        if any(atom in data[:256] for atom in (b"ftyp", b"moov", b"mdat")):
            return True

        # Iteratively parse root-level box headers from start (up to first 64KB)
        offset = 0
        limit = min(len(data), 65536)
        has_iso_box = False

        while offset + 8 <= limit:
            box_size = int.from_bytes(data[offset:offset + 4], "big")
            box_type = data[offset + 4:offset + 8]

            if box_type in known_atoms:
                has_iso_box = True
                if box_type in (b"ftyp", b"moov", b"mdat"):
                    return True
                if box_size == 0 or box_size == 1:
                    # Extended 64-bit size or spans to EOF
                    break
                if box_size < 8:
                    break
                offset += box_size
            else:
                break

        # Check trailing bytes for moov atom (standard non-faststart MP4/MOV where moov is at EOF)
        tail_search_len = min(len(data), 262144)
        if b"moov" in data[-tail_search_len:] and (has_iso_box or b"mdat" in data):
            return True

        return False

    def safe_ingest_uploaded_stream(
        self,
        file_name: str,
        file_bytes: bytes,
        camera_name: str,
        max_bytes: int = 500 * 1024 * 1024,
        temp_dir: Optional[str | Path] = None,
        validate_decodable: bool = False,
        operator: str = "ForensicInvestigator"
    ) -> Tuple[Optional[EvidenceRecord], Optional[str]]:
        """Validate and safely ingest an in-memory uploaded video buffer.

        Validates non-empty content, enforces max size limits, checks container
        magic bytes, writes to an isolated temporary copy, optionally tests
        decodeability, registers into the evidence vault, and guarantees cleanup
        of temporary files even on error.

        Returns:
            (record, None) on success, or (None, error_message) on failure.
        """
        if hasattr(file_bytes, "getvalue"):
            raw_data = file_bytes.getvalue()
        elif hasattr(file_bytes, "read"):
            raw_data = file_bytes.read()
        elif isinstance(file_bytes, (bytes, bytearray)):
            raw_data = bytes(file_bytes)
        elif isinstance(file_bytes, str):
            raw_data = file_bytes.encode("utf-8")
        else:
            raw_data = b""

        if not raw_data or len(raw_data) == 0:
            return None, f"Uploaded file '{file_name}' is empty (0 bytes)."

        if len(raw_data) > max_bytes:
            mb = max_bytes / (1024 * 1024)
            return None, f"Uploaded file '{file_name}' ({len(raw_data)/(1024*1024):.1f} MB) exceeds maximum allowed size ({mb:.0f} MB)."

        # Validate video container magic bytes / signatures
        if not self.is_valid_video_container(raw_data):
            return None, f"Uploaded file '{file_name}' does not have a recognized video container signature (MP4, AVI, MOV, or MKV)."

        target_dir = Path(temp_dir) if temp_dir else self.temp_upload_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.sanitize_path_component(file_name, "upload")
        temp_file = target_dir / f"temp_{os.getpid()}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{safe_name}"

        try:
            temp_file.write_bytes(raw_data)

            # Optional deep validation: verify that video stream can actually be opened and decoded
            if validate_decodable:
                import cv2
                cap = cv2.VideoCapture(str(temp_file))
                opened = cap.isOpened()
                ret = False
                total_frames = 0
                if opened:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    ret, frame = cap.read()
                cap.release()
                if not opened or not ret or total_frames < 1:
                    return None, f"Uploaded file '{file_name}' contains a video container signature, but video stream cannot be opened or decoded (0 decodable frames)."

            record = self.register_evidence_file(
                temp_file,
                camera_name=camera_name,
                copy_to_evidence_vault=True,
                original_filename=Path(file_name).name,
                operator=operator
            )
            return record, None
        except Exception as ex:
            return None, f"Failed to ingest evidence '{file_name}': {str(ex)}"
        finally:
            if temp_file.exists():
                self.unlock_file(temp_file)
                try:
                    temp_file.unlink(missing_ok=True)
                except Exception:
                    pass
