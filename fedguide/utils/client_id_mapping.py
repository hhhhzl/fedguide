"""
Deterministic mapping from Flower's client IDs to 0, 1, 2, ..., num_clients-1.
Flower VCE uses long integer strings (e.g. '11638192436476818872') which cause
hash collisions when using int(cid)%N. This module uses file-based sequential
assignment to guarantee N distinct mappings for N distinct cids.
"""
from __future__ import annotations

import json
import os
import tempfile
import fcntl
import time
from typing import Optional


def get_mapped_client_id(cid: str, num_clients: int, mapping_file: Optional[str] = None) -> int:
    """
    Map Flower's cid to 0..num_clients-1 deterministically.
    Uses file locking for sequential assignment across Ray workers.
    """
    if mapping_file is None:
        mapping_file = os.path.join(tempfile.gettempdir(), "fedguide_cid_mapping.json")
    
    max_retries = 20
    for attempt in range(max_retries):
        try:
            with open(mapping_file, "a+") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    raw = f.read()
                    mapping = json.loads(raw) if raw.strip() else {}
                except (json.JSONDecodeError, ValueError):
                    mapping = {}
                
                if cid not in mapping:
                    mapping[cid] = len(mapping)
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(mapping))
                    f.flush()
                
                mapped = mapping[cid]
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return int(mapped) % num_clients
        except (BlockingIOError, OSError) as e:
            time.sleep(0.01 * (attempt + 1))
            continue
        except Exception:
            raise
    
    # Fallback: hash-based (may collide)
    import hashlib
    h = int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)
    return h % num_clients


def clear_mapping_file(mapping_file: Optional[str] = None) -> None:
    """Remove mapping file at start of a fresh run."""
    if mapping_file is None:
        mapping_file = os.path.join(tempfile.gettempdir(), "fedguide_cid_mapping.json")
    try:
        if os.path.exists(mapping_file):
            os.remove(mapping_file)
    except OSError:
        pass
