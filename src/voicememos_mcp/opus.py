"""Silencing a time range inside an Ogg Opus file without decoding it (used by ``redact.py``).

A recorder's file is a chain of Ogg pages holding Opus packets of 2.5 to 120 ms. To forget a stretch of a
recording, every packet that overlaps the stretch is replaced with a packet of the same duration that
decodes to digital silence. Nothing else is touched: the packets outside the range keep their bytes, the
granule positions keep their values, so the recording is exactly as long as before and every time stamp
after the range still points at the same sound. The file gets smaller, not shorter.

Why not decode, zero the samples and encode again: that would re-encode two hours of speech to remove five
minutes (a lossy generation for everything that is kept), needs an encoder on the host, and the result's
length depends on the encoder's padding.

What a listener hears at the two edges: the decoder needs about 100 ms to settle after the silence, so the
first tenth of a second after the range may sound clipped. A packet that only partly overlaps the range is
silenced whole (at most 20 ms beyond each edge with the usual packet size): the range's own sound never
stays, at the price of a sliver of its neighbours.

``silence`` writes a new file and never the source; ``check`` re-reads a result against its source
(structure: checksums, packet count, durations, bytes outside the range); ``decoded_check`` decodes both
with ffmpeg, when the host has one, and measures what is heard.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path


class OpusError(Exception):
    """The file is not something this module can edit safely; nothing was written."""


_REVERSED = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))
# Frame length in ms by the TOC byte's configuration number (RFC 6716, table 2).
_FRAME_MS = [10, 20, 40, 60] * 3 + [10, 20] * 2 + [2.5, 5, 10, 20] * 4
_CELT_MS = [2.5, 5, 10, 20]
_SILENT_FRAME = b"\xff\xfe"  # a CELT frame whose first symbol is the silence flag
RATE = 48_000  # granule positions count 48 kHz samples whatever the input rate was
FFMPEG_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")


def _crc(data: bytes) -> int:
    """Ogg's checksum (CRC-32, polynomial 0x04c11db7, not reflected, start and end value 0), through zlib's
    reflected one: the same polynomial read from the other end, so the bytes and the result are bit-reversed."""
    value = zlib.crc32(data.translate(_REVERSED), 0xFFFFFFFF) ^ 0xFFFFFFFF
    return int(f"{value:032b}"[::-1], 2)


def _frames(packet: bytes) -> tuple[float, int]:
    """(frame length in ms, number of frames) of an Opus packet."""
    if not packet:
        raise OpusError("an empty Opus packet")
    code = packet[0] & 3
    if code == 0:
        count = 1
    elif code in (1, 2):
        count = 2
    else:
        if len(packet) < 2:
            raise OpusError("an Opus packet with a frame count but no count byte")
        count = packet[1] & 63
        if not count:
            raise OpusError("an Opus packet of zero frames")
    return _FRAME_MS[packet[0] >> 3], count


def samples_of(packet: bytes) -> int:
    ms, count = _frames(packet)
    return int(ms * RATE / 1000) * count


def silent_packet(packet: bytes) -> bytes:
    """A packet as long as ``packet`` that decodes to silence: CELT frames of the same length (20 ms ones for
    the 40 and 60 ms frames CELT does not have), each two bytes."""
    ms, count = _frames(packet)
    if ms > 20:
        count *= int(ms // 20)
        ms = 20
    toc = ((28 + _CELT_MS.index(ms)) << 3) | (packet[0] & 4)  # fullband CELT, the original's channel bit
    if count == 1:
        return bytes([toc]) + _SILENT_FRAME
    if count == 2:
        return bytes([toc | 1]) + _SILENT_FRAME * 2
    return bytes([toc | 3, count]) + _SILENT_FRAME * count


def is_silent(packet: bytes) -> bool:
    return packet == silent_packet(packet)


@dataclass
class _Entry:
    """One packet, or one page's share of a packet that runs over a page boundary."""

    data: bytes
    lacing: bytes
    packet: int | None  # the packet's number when it ends in this entry


@dataclass
class _Page:
    flags: int
    granule: int
    serial: int
    seq: int
    entries: list[_Entry]
    raw: bytes = b""  # the page as it is in the file
    foreign: bool = False  # a page of another stream in the same file: written back byte for byte


def _read_pages(data: bytes) -> list[_Page]:
    pages: list[_Page] = []
    at = 0
    while at < len(data):
        if data[at:at + 4] != b"OggS" or at + 27 > len(data):
            raise OpusError(f"not an Ogg page at byte {at}")
        version, flags, granule, serial, seq, crc, count = struct.unpack_from("<BBqIIIB", data, at + 4)
        lacing = data[at + 27:at + 27 + count]
        size = sum(lacing)
        end = at + 27 + count + size
        if version != 0 or end > len(data):
            raise OpusError(f"a broken Ogg page at byte {at}")
        raw = data[at:end]
        if _crc(raw[:22] + b"\0\0\0\0" + raw[26:]) != crc:
            raise OpusError(f"a page with a wrong checksum at byte {at}: the file is damaged, nothing is changed")
        entries: list[_Entry] = []
        body, i, o = raw[27 + count:], 0, 0
        while i < count:
            j, length = i, 0
            while j < count and lacing[j] == 255:
                length += 255
                j += 1
            ends = j < count
            if ends:
                length += lacing[j]
                j += 1
            entries.append(_Entry(body[o:o + length], bytes(lacing[i:j]), -1 if ends else None))
            o, i = o + length, j
        pages.append(_Page(flags, granule, serial, seq, entries, raw))
        at = end
    if not pages:
        raise OpusError("an empty file")
    return _own_stream(pages)


def _own_stream(pages: list[_Page]) -> list[_Page]:
    """Mark the pages that are not the Opus stream's. Some recorders write a second stream beside the audio
    (a Plaud recorder: one page of zeros for each page of sound). Such pages are never changed; a second
    stream that holds data is refused, because what it holds of the range is not known."""
    starts = [p for p in pages if p.flags & 2]
    opus_starts = [p for p in starts if p.entries and p.entries[0].data.startswith(b"OpusHead")]
    if len(opus_starts) != 1 or len({p.serial for p in starts}) != len(starts):
        raise OpusError("not one Opus stream in the file (a chained Ogg, or none): not edited")
    own = opus_starts[0].serial
    for page in pages:
        if page.serial == own:
            continue
        page.foreign = True
        if not page.flags & 2 and any(e.data.strip(b"\0") for e in page.entries):
            raise OpusError("a second stream in the file holds data: not edited")
    return pages


def _packets(pages: list[_Page]) -> list[tuple[bytes, list[tuple[int, int]]]]:
    """Every packet with the (page, entry) places its bytes sit in; numbers the entries where packets end."""
    out: list[tuple[bytes, list[tuple[int, int]]]] = []
    parts: list[tuple[int, int]] = []
    for p, page in enumerate(pages):
        if page.foreign:
            continue
        for e, entry in enumerate(page.entries):
            parts.append((p, e))
            if entry.packet is not None:
                entry.packet = len(out)
                out.append((b"".join(pages[a].entries[b].data for a, b in parts), parts))
                parts = []
    if parts:
        raise OpusError("the file ends in the middle of a packet")
    return out


def _lacing(length: int) -> bytes:
    return b"\xff" * (length // 255) + bytes([length % 255])


def _write_pages(pages: list[_Page]) -> bytes:
    out = []
    seq = -1
    for page in pages:
        if page.foreign:
            out.append(page.raw)
            continue
        seq += 1
        lacing = b"".join(e.lacing for e in page.entries)
        if len(lacing) > 255:
            raise OpusError("a page would hold more than 255 segments")
        body = b"".join(e.data for e in page.entries)
        head = b"OggS" + struct.pack("<BBqIIIB", 0, page.flags, page.granule, page.serial, seq, 0, len(lacing)) + lacing
        out.append(head[:22] + struct.pack("<I", _crc(head + body)) + head[26:] + body)
    return b"".join(out)


def _open(path: Path) -> tuple[list[_Page], list[tuple[bytes, list[tuple[int, int]]]], int]:
    pages = _read_pages(Path(path).read_bytes())
    packets = _packets(pages)
    if len(packets) < 2 or not packets[0][0].startswith(b"OpusHead") or not packets[1][0].startswith(b"OpusTags"):
        raise OpusError("not an Ogg Opus file (no OpusHead and OpusTags at its start)")
    head = packets[0][0]
    if len(head) < 19 or head[8] > 15:
        raise OpusError("an OpusHead this module does not know")
    return pages, packets, struct.unpack_from("<H", head, 10)[0]  # pre-skip: samples the decoder drops at the start


def is_opus(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            start = fh.read(64)
    except OSError:
        return False
    return start[:4] == b"OggS" and b"OpusHead" in start


def info(path: Path) -> dict:
    """Length and packet counts, read from the packets themselves (no decoder)."""
    pages, packets, pre_skip = _open(path)
    total = sum(samples_of(p) for p, _ in packets[2:])
    last = max((p.granule for p in pages if p.granule >= 0 and not p.foreign), default=0)
    return {"pages": len(pages), "other_stream_pages": sum(1 for p in pages if p.foreign), "packets": len(packets) - 2, "pre_skip": pre_skip, "bytes": Path(path).stat().st_size,
            "duration_ms": round(max(0, (last or total) - pre_skip) * 1000 / RATE),
            "silent_packets": sum(1 for p, _ in packets[2:] if is_silent(p))}


def _spans(packets: list[tuple[bytes, list]], pre_skip: int) -> list[tuple[float, float]]:
    """Each audio packet's (start, end) in ms of the decoded recording."""
    out, at = [], -pre_skip
    for packet, _ in packets[2:]:
        n = samples_of(packet)
        out.append((at * 1000 / RATE, (at + n) * 1000 / RATE))
        at += n
    return out


def silence(source: Path, target: Path, start_ms: float, end_ms: float) -> dict:
    """Write ``target``: ``source`` with every packet that overlaps [start_ms, end_ms) made silent. The source
    is only read. Doing it again to the result changes nothing (a silent packet's replacement is itself)."""
    if not end_ms > start_ms >= 0:
        raise OpusError("a range needs a start before its end")
    pages, packets, pre_skip = _open(source)
    spans = _spans(packets, pre_skip)
    replaced = already = 0
    first = last = None
    for number, ((begin, end), (packet, parts)) in enumerate(zip(spans, packets[2:]), start=2):
        if end <= start_ms or begin >= end_ms:
            continue
        first = begin if first is None else first
        last = end
        new = silent_packet(packet)
        if new == packet:
            already += 1
            continue
        replaced += 1
        for a, b in parts[:-1]:  # the packet's shares on earlier pages go
            pages[a].entries[b] = _Entry(b"", b"", None)
        a, b = parts[-1]
        pages[a].entries[b] = _Entry(new, _lacing(len(new)), number)
        if len(parts) > 1 and b == 0:
            pages[a].flags &= ~1  # the page no longer starts with the rest of a packet
    for page in pages:
        page.entries = [e for e in page.entries if e.lacing]
    kept = [p for p in pages if p.foreign or p.entries or p.flags & 6]  # never drop the first or the last page of the stream
    Path(target).write_bytes(_write_pages(kept))
    return {"packets": len(packets) - 2, "silenced": replaced, "already_silent": already,
            "audio_from_ms": None if first is None else round(first), "audio_to_ms": None if last is None else round(last)}


def unsilent(path: Path, start_ms: float, end_ms: float) -> int:
    """How many packets that overlap the range are not silent (0 after ``silence``). Reads only."""
    _, packets, pre_skip = _open(path)
    return sum(1 for (begin, end), (packet, _) in zip(_spans(packets, pre_skip), packets[2:])
               if not (end <= start_ms or begin >= end_ms) and not is_silent(packet))


def check(source: Path, target: Path, start_ms: float, end_ms: float) -> dict:
    """Read ``target`` back against ``source``: the same number of packets, each as long as before, every
    packet in the range silent and every other one byte for byte the same; the same length. Raises on any
    difference; returns the numbers."""
    _, before, skip_before = _open(source)
    pages, after, skip_after = _open(target)
    if len(before) != len(after) or skip_before != skip_after or before[0][0] != after[0][0] or before[1][0] != after[1][0]:
        raise OpusError("the silenced file does not have the source's packets")
    if [p.raw for p in _read_pages(Path(source).read_bytes()) if p.foreign] != [p.raw for p in pages if p.foreign]:
        raise OpusError("a page of the file's other stream changed")
    spans = _spans(before, skip_before)
    inside = same = 0
    for (begin, end), (old, _), (new, _) in zip(spans, before[2:], after[2:]):
        if samples_of(old) != samples_of(new):
            raise OpusError("a packet changed its length")
        if end <= start_ms or begin >= end_ms:
            if old != new:
                raise OpusError("a packet outside the range changed")
            same += 1
        else:
            if not is_silent(new):
                raise OpusError("a packet inside the range is not silent")
            inside += 1
    a, b = info(source), info(target)
    if abs(a["duration_ms"] - b["duration_ms"]) >= 50:
        raise OpusError(f"the length changed: {a['duration_ms']} ms -> {b['duration_ms']} ms")
    return {"duration_ms_before": a["duration_ms"], "duration_ms_after": b["duration_ms"], "packets_in_range": inside,
            "packets_outside_unchanged": same, "bytes_before": a["bytes"], "bytes_after": b["bytes"], "pages": len(pages)}


def ffmpeg() -> str | None:
    for folder in FFMPEG_DIRS:
        if os.access(os.path.join(folder, "ffmpeg"), os.X_OK):
            return os.path.join(folder, "ffmpeg")
    return shutil.which("ffmpeg")


EDGE_MS = 300  # the decoder settles within about 100 ms of a change; what is heard is judged this far from the edges
QUIET_RMS = 8.0  # of 32768: digital silence decodes to 0, a settling decoder to a few units


def decoded_check(source: Path, target: Path, start_ms: float, end_ms: float, rate: int = 16_000) -> dict:
    """Decode both files with ffmpeg and measure: the same number of samples, the range silent, the rest
    sounding as before. ``{"decoded": False, ...}`` when the host has no ffmpeg (the structural check still
    stands). Both decodes are streamed and nothing is written to disk. Raises when what is heard is wrong."""
    try:
        import numpy as np
    except ImportError:  # an optional dependency of this package
        return {"decoded": False, "why": "numpy is not installed"}

    tool = ffmpeg()
    if tool is None:
        return {"decoded": False, "why": "no ffmpeg on this host"}

    def stream(path: Path):
        return subprocess.Popen([tool, "-nostdin", "-loglevel", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    a, b = stream(source), stream(target)
    lo, hi = int(start_ms * rate / 1000), int(end_ms * rate / 1000)
    edge = int(EDGE_MS * rate / 1000)
    block = rate * 30
    at = count_a = count_b = 0
    inside = np.zeros(2)  # sum of squares, samples: the target's sound well inside the range
    outside_before, outside_after, differs = np.zeros(2), np.zeros(2), 0
    try:
        while True:
            raw_a, raw_b = a.stdout.read(block * 2), b.stdout.read(block * 2)
            if not raw_a and not raw_b:
                break
            x = np.frombuffer(raw_a[:len(raw_a) // 2 * 2], dtype="<i2").astype(np.float64)
            y = np.frombuffer(raw_b[:len(raw_b) // 2 * 2], dtype="<i2").astype(np.float64)
            count_a, count_b = count_a + len(x), count_b + len(y)
            n = min(len(x), len(y))
            where = np.arange(at, at + n)
            quiet = (where >= lo + edge) & (where < hi - edge)
            away = (where < lo - edge) | (where >= hi + edge)
            inside += (float((y[:n][quiet] ** 2).sum()), int(quiet.sum()))
            outside_before += (float((x[:n][away] ** 2).sum()), int(away.sum()))
            outside_after += (float((y[:n][away] ** 2).sum()), int(away.sum()))
            if away.any():
                differs = max(differs, int(np.abs(x[:n][away] - y[:n][away]).max()))
            at += n
    finally:
        for p in (a, b):
            p.stdout.close()
            p.wait()
    if a.returncode or not count_a:  # the original itself is not something this ffmpeg reads: nothing to compare
        return {"decoded": False, "why": "ffmpeg could not decode the original"}
    if b.returncode:
        raise OpusError("ffmpeg decodes the original but not the silenced file")

    def rms(acc) -> float | None:
        return round(float((acc[0] / acc[1]) ** 0.5), 2) if acc[1] else None

    out = {"decoded": True, "samples_before": count_a, "samples_after": count_b, "rate": rate,
           "rms_in_range": rms(inside), "rms_outside_before": rms(outside_before), "rms_outside_after": rms(outside_after),
           "max_difference_outside": differs}
    if abs(count_a - count_b) * 1000 / rate >= 50:
        raise OpusError(f"the decoded length changed: {count_a} -> {count_b} samples")
    if out["rms_in_range"] is not None and out["rms_in_range"] > QUIET_RMS:
        raise OpusError(f"the range is not silent when decoded (rms {out['rms_in_range']})")
    if differs > 64:
        raise OpusError(f"the sound outside the range changed (largest difference {differs})")
    return out
