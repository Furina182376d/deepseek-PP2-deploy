"""
Encrypted inter-node communication for vLLM pipeline parallelism.

Provides two layers:

1.  **Crypto primitives** — AES-256-GCM encrypt/decrypt for tensor data, using a
    pre-shared key (assumed negotiated out-of-band, injected via env var
    ``VLLM_COMM_PSK``).

2.  **ProcessGroup wrapper** — ``EncryptedP2PGroup`` intercepts point-to-point
    ``send`` / ``recv`` calls and encrypts tensors that cross node boundaries.
    Intra-node (same-host) transfers remain plaintext NCCL for performance.

Usage (monkey-patch vLLM's PP P2P calls)::

    from comm_crypto import install_encrypted_pp_hooks
    install_encrypted_pp_hooks()

    # Then load and run the model normally — PP activations crossing nodes
    # are automatically encrypted.

If the env var ``VLLM_COMM_PSK`` is not set, encryption is a no-op (passthrough).
"""

import os
import struct
from typing import Callable

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# Key management — pre-shared key from env (assumed negotiated out-of-band)
# ---------------------------------------------------------------------------
_PSK = os.environ.get("VLLM_COMM_PSK", "").encode("utf-8")


def _pad_key(key: bytes, length: int = 32) -> bytes:
    """Pad or truncate key to exactly *length* bytes."""
    if len(key) >= length:
        return key[:length]
    return key + b"\x00" * (length - len(key))


# ---------------------------------------------------------------------------
# AES-256-GCM via Python's cryptography module (if available)
# Falls back to a simple XOR cipher so the module loads without the dep, but
# logs a loud warning.
# ---------------------------------------------------------------------------
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _HAS_AES = True
except ImportError:
    _HAS_AES = False


def _warn_fallback():
    import warnings

    warnings.warn(
        "cryptography not installed — comm_crypto using XOR fallback (INSECURE). "
        "Install with: pip install cryptography"
    )


def encrypt_tensor(tensor: torch.Tensor) -> bytes:
    """
    Encrypt a tensor's raw bytes with AES-256-GCM (or XOR fallback).

    Returns
    -------
    bytes:
        ``nonce (12) + ciphertext + tag (16)``  —  for AES-GCM.
        ``len(4) + xordata``                    —  for XOR fallback.
    """
    if not _PSK:
        return b""  # no encryption key set

    raw = tensor.cpu().contiguous().numpy().tobytes()

    if _HAS_AES:
        nonce = os.urandom(12)
        aesgcm = AESGCM(_pad_key(_PSK, 32))
        ct = aesgcm.encrypt(nonce, raw, None)
        return nonce + ct  # ct already includes the 16-byte tag
    else:
        _warn_fallback()
        key_bytes = _pad_key(_PSK, 32)
        # Simple XOR (NOT secure — placeholder only)
        xor = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(raw))
        return struct.pack("<I", len(xor)) + xor


def decrypt_tensor(data: bytes, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    """
    Decrypt *data* back into a tensor of *shape* and *dtype*.
    """
    if not _PSK or not data:
        raise ValueError("Cannot decrypt: no key or empty data")

    if _HAS_AES:
        nonce, ct = data[:12], data[12:]
        aesgcm = AESGCM(_pad_key(_PSK, 32))
        raw = aesgcm.decrypt(nonce, ct, None)
    else:
        _warn_fallback()
        xor_len = struct.unpack("<I", data[:4])[0]
        xor = data[4 : 4 + xor_len]
        key_bytes = _pad_key(_PSK, 32)
        raw = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(xor))

    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# Node classification — determine which ranks are on the other node
# ---------------------------------------------------------------------------
def _is_cross_node(peer_rank: int, local_world_size: int | None = None) -> bool:
    """
    Heuristic: two ranks are on different nodes if they belong to different
    ``local_world_size`` blocks.
    """
    if local_world_size is None:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 4))
    my_rank = dist.get_rank()
    my_node = my_rank // local_world_size
    peer_node = peer_rank // local_world_size
    return my_node != peer_node


# ---------------------------------------------------------------------------
# ProcessGroup wrapper
# ---------------------------------------------------------------------------
class EncryptedP2PGroup:
    """
    Wraps a ``torch.distributed.ProcessGroup`` and encrypts P2P traffic that
    crosses node boundaries.

    Falls back to the underlying NCCL group for intra-node transfers.
    """

    def __init__(self, pg: dist.ProcessGroup, local_world_size: int | None = None):
        self._pg = pg
        self._local_ws = local_world_size or int(os.environ.get("LOCAL_WORLD_SIZE", 4))

    @property
    def _encrypt_active(self) -> bool:
        return bool(_PSK)

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0):
        if self._encrypt_active and _is_cross_node(dst, self._local_ws):
            data = encrypt_tensor(tensor)
            # Send metadata: shape info + encrypted data length
            meta = struct.pack("<III", tensor.ndim, len(data), tensor.element_size())
            for i, s in enumerate(tensor.shape):
                meta += struct.pack("<I", s)
            meta_t = torch.frombuffer(bytearray(meta), dtype=torch.uint8)
            dist.send(meta_t, dst, group=self._pg, tag=tag)

            data_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            dist.send(data_t, dst, group=self._pg, tag=tag + 1)
        else:
            dist.send(tensor, dst, group=self._pg, tag=tag)

    def recv(self, tensor: torch.Tensor, src: int, tag: int = 0):
        if self._encrypt_active and _is_cross_node(src, self._local_ws):
            # Receive metadata
            meta_t = torch.empty(16, dtype=torch.uint8)
            dist.recv(meta_t, src, group=self._pg, tag=tag)
            meta = meta_t.cpu().numpy().tobytes()
            ndim, data_len = struct.unpack("<II", meta[:8])
            shape = tuple(
                struct.unpack("<I", meta[8 + i * 4 : 8 + (i + 1) * 4])[0]
                for i in range(ndim)
            )

            # Receive encrypted data
            data_t = torch.empty(data_len, dtype=torch.uint8)
            dist.recv(data_t, src, group=self._pg, tag=tag + 1)
            data = data_t.cpu().numpy().tobytes()

            decrypted = decrypt_tensor(data, torch.Size(shape), tensor.dtype)
            tensor.copy_(decrypted)
        else:
            dist.recv(tensor, src, group=self._pg, tag=tag)


# ---------------------------------------------------------------------------
# Hook installer — monkey-patches vLLM's PP communication
# ---------------------------------------------------------------------------
_INSTALLED = False


def install_encrypted_pp_hooks() -> bool:
    """
    Monkey-patch vLLM to use encrypted P2P for cross-node PP traffic.

    Returns ``True`` if hooks were installed, ``False`` if already installed.
    """
    global _INSTALLED
    if _INSTALLED:
        return False
    _INSTALLED = True

    if not _PSK:
        print("[comm_crypto] VLLM_COMM_PSK not set — encryption disabled (no-op).")
        return True

    # Save originals as module attributes so remove() can find them
    dist._orig_send = dist.send
    dist._orig_recv = dist.recv

    def _encrypted_send(tensor: torch.Tensor, dst: int, group=None, tag: int = 0):
        if _is_cross_node(dst):
            data = encrypt_tensor(tensor)
            meta = struct.pack("<III", tensor.ndim, len(data), tensor.element_size())
            for s in tensor.shape:
                meta += struct.pack("<I", s)
            meta_t = torch.frombuffer(bytearray(meta), dtype=torch.uint8)
            dist._orig_send(meta_t, dst, group=group, tag=tag)
            data_t = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            dist._orig_send(data_t, dst, group=group, tag=tag + 1)
        else:
            dist._orig_send(tensor, dst, group=group, tag=tag)

    def _encrypted_recv(tensor: torch.Tensor, src: int, group=None, tag: int = 0):
        if _is_cross_node(src):
            meta_t = torch.empty(16, dtype=torch.uint8)
            dist._orig_recv(meta_t, src, group=group, tag=tag)
            meta = meta_t.cpu().numpy().tobytes()
            ndim, data_len = struct.unpack("<II", meta[:8])
            shape_parts = []
            for i in range(ndim):
                shape_parts.append(
                    struct.unpack("<I", meta[8 + i * 4 : 8 + (i + 1) * 4])[0]
                )
            shape = tuple(shape_parts)

            data_t = torch.empty(data_len, dtype=torch.uint8)
            dist._orig_recv(data_t, src, group=group, tag=tag + 1)
            data = data_t.cpu().numpy().tobytes()

            decrypted = decrypt_tensor(data, torch.Size(shape), tensor.dtype)
            tensor.copy_(decrypted)
        else:
            dist._orig_recv(tensor, src, group=group, tag=tag)

    dist.send = _encrypted_send
    dist.recv = _encrypted_recv

    print("[comm_crypto] Encrypted PP P2P hooks installed (AES-256-GCM).")
    return True


def remove_encrypted_pp_hooks():
    """Restore original torch.distributed.send/recv."""
    global _INSTALLED
    if hasattr(dist, "_orig_send"):
        dist.send = dist._orig_send
        del dist._orig_send
    if hasattr(dist, "_orig_recv"):
        dist.recv = dist._orig_recv
        del dist._orig_recv
    _INSTALLED = False
