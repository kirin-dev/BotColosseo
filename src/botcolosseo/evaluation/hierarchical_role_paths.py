"""Resolve role populations without assuming union indices equal role indices."""

import hashlib


def role_paths(identity, paths):
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    if hashes != identity["strategies"] or len(set(hashes)) != len(hashes):
        raise ValueError("Population union hash/order mismatch")
    by_hash = dict(zip(hashes, paths, strict=True))
    return {
        role: [by_hash[key] for key in identity.get(f"{role}_strategies", hashes)]
        for role in ("host", "opponent")
    }
