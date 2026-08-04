#!/bin/bash
# ---------------------------------------------------------------------------
# Set up IPsec transport-mode encryption between two GPU nodes.
#
# This is an ALTERNATIVE to comm_crypto.py — zero vLLM code changes required.
# All IP traffic between the two nodes on the eth0 interface is encrypted
# transparently at the kernel level.
#
# PREREQUISITES:
#   - Run as root on BOTH nodes.
#   - Kernel IPsec (CONFIG_XFRM) enabled.
#   - iproute2 / iproute2-xfrm installed.
#
# USAGE:
#   Node 0 (192.168.0.63):  sudo ./setup_ipsec.sh 0
#   Node 1 (192.168.0.65):  sudo ./setup_ipsec.sh 1
#
# KEY GENERATION (do once, share between nodes):
#   dd if=/dev/urandom bs=32 count=1 2>/dev/null | xxd -p -c64
# ---------------------------------------------------------------------------
set -euo pipefail

NODE="${1:?Usage: $0 <node_index (0 or 1)>}"

MY_IPS=("192.168.0.63" "192.168.0.65")
PEER_IPS=("192.168.0.65" "192.168.0.63")

MY_IP="${MY_IPS[$NODE]}"
PEER_IP="${PEER_IPS[$NODE]}"

# ---- Pre-shared keys (REPLACE with generated values!) ----
# AES-128 key (16 bytes hex):  openssl rand -hex 16
AES_KEY="00112233445566778899aabbccddeeff"
# HMAC-SHA256 key (32 bytes hex):  openssl rand -hex 32
AUTH_KEY="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

SPI_OUT=0x1000
SPI_IN=0x1001

echo "=== IPsec transport mode ==="
echo "  Node ${NODE}: ${MY_IP}"
echo "  Peer:      ${PEER_IP}"
echo ""

# ---- Flush any existing SAs / policies for this peer pair ----
ip xfrm state delete src "${MY_IP}" dst "${PEER_IP}" proto esp spi "${SPI_OUT}" 2>/dev/null || true
ip xfrm state delete src "${PEER_IP}" dst "${MY_IP}" proto esp spi "${SPI_IN}"  2>/dev/null || true
ip xfrm policy delete src "${MY_IP}/32" dst "${PEER_IP}/32" dir out 2>/dev/null || true
ip xfrm policy delete src "${PEER_IP}/32" dst "${MY_IP}/32" dir in  2>/dev/null || true

# ---- Security Associations ----
# Outbound: MY_IP → PEER_IP
ip xfrm state add \
    src "${MY_IP}" dst "${PEER_IP}" \
    proto esp spi "${SPI_OUT}" reqid 1 mode transport \
    auth sha256 "${AUTH_KEY}" \
    enc aes "${AES_KEY}"

# Inbound: PEER_IP → MY_IP
ip xfrm state add \
    src "${PEER_IP}" dst "${MY_IP}" \
    proto esp spi "${SPI_IN}" reqid 2 mode transport \
    auth sha256 "${AUTH_KEY}" \
    enc aes "${AES_KEY}"

# ---- Policies ----
ip xfrm policy add \
    src "${MY_IP}/32" dst "${PEER_IP}/32" dir out \
    tmpl src "${MY_IP}" dst "${PEER_IP}" proto esp reqid 1 mode transport

ip xfrm policy add \
    src "${PEER_IP}/32" dst "${MY_IP}/32" dir in \
    tmpl src "${PEER_IP}" dst "${MY_IP}" proto esp reqid 2 mode transport

echo "IPsec configured."
echo ""
echo "Verify:  ip xfrm state; ip xfrm policy"
echo "Test:    tcpdump -i eth0 host ${PEER_IP} -c 5 -X  (should show ESP, not plaintext)"
echo "Teardown: sudo ip xfrm state flush; sudo ip xfrm policy flush"
