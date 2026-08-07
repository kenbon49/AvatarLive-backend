"""MuseTalk media packet parsing and timeline remapping."""

from __future__ import annotations

import struct


PACKET_MAGIC = b"MSTK"
PACKET_VERSION = 1
PACKET_PCM16 = 2
PACKET_HEADER = struct.Struct("<4sBBHIIQ")


def remap_media_packet(
    packet: bytes,
    *,
    sequence: int,
    pts_offset_us: int,
) -> tuple[bytes, int, int]:
    """Return a packet with a conversation-wide sequence and PTS."""

    if len(packet) < PACKET_HEADER.size:
        raise ValueError("media packet is shorter than its header")
    magic, version, packet_type, flags, _sequence, size, pts_us = PACKET_HEADER.unpack_from(
        packet
    )
    payload = packet[PACKET_HEADER.size :]
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise ValueError("unsupported media packet")
    if len(payload) != size:
        raise ValueError("media packet payload length differs from header")
    outgoing_pts = int(pts_offset_us) + int(pts_us)
    header = PACKET_HEADER.pack(
        magic,
        version,
        packet_type,
        flags,
        int(sequence),
        size,
        outgoing_pts,
    )
    return header + payload, packet_type, outgoing_pts
