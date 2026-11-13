#!/usr/bin/env python3
"""Generate the legacy CJK lookup tables used by `openjiuwen-sysop`.

The Rust rewrite decodes GB18030/GBK, Big5 and Shift-JIS without depending on an
external crate, so the code point mappings have to be committed as data. This
script produces that data from Python's own built-in codecs, which keeps the
Rust tables byte-for-byte compatible with what the Python reference decodes.

Usage (from the crate root or anywhere else)::

    python3 crates/openjiuwen-sysop/scripts/gen_cjk_tables.py           # write
    python3 crates/openjiuwen-sysop/scripts/gen_cjk_tables.py --check   # verify

The script also verifies its own output:

* every byte pair in the modelled space is compared against ``bytes.decode``;
* every GB18030 four-byte code is compared against ``bytes.decode``;
* random byte streams are decoded with a faithful model of the Rust decoder
  (including its replacement/resynchronisation rules) and compared with the
  Python codecs driven with ``errors="replace"``.

Only authoring-time tooling: builds and tests never invoke Python.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parent.parent
OUT_DIR = CRATE / "src" / "encoding" / "tables"

# --------------------------------------------------------------------------
# Code space geometry (mirrored by `src/encoding/cjk.rs`)
# --------------------------------------------------------------------------

GB_LEAD = range(0x81, 0xFF)
GB_TRAILS = list(range(0x40, 0x7F)) + list(range(0x80, 0xFF))
GB_ROWS = len(GB_LEAD)  # 126
GB_COLS = len(GB_TRAILS)  # 190

BIG5_LEAD = range(0xA1, 0xFA)
BIG5_TRAILS = list(range(0x40, 0x7F)) + list(range(0xA1, 0xFF))
BIG5_ROWS = len(BIG5_LEAD)  # 89
BIG5_COLS = len(BIG5_TRAILS)  # 157

SJIS_LEADS = list(range(0x81, 0xA0)) + list(range(0xE0, 0xFD))  # 60
SJIS_TRAILS = list(range(0x40, 0x7F)) + list(range(0x80, 0xFD))
SJIS_COLS = len(SJIS_TRAILS)  # 188

SJIS_HALFWIDTH_START = 0xA1
SJIS_HALFWIDTH_END = 0xDF
SJIS_HALFWIDTH_UNICODE = 0xFF61


def gb_col(trail: int) -> int | None:
    if 0x40 <= trail <= 0x7E:
        return trail - 0x40
    if 0x80 <= trail <= 0xFE:
        return trail - 0x41
    return None


def big5_col(trail: int) -> int | None:
    if 0x40 <= trail <= 0x7E:
        return trail - 0x40
    if 0xA1 <= trail <= 0xFE:
        return trail - 0x62
    return None


def sjis_col(trail: int) -> int | None:
    if 0x40 <= trail <= 0x7E:
        return trail - 0x40
    if 0x80 <= trail <= 0xFC:
        return trail - 0x80 + 63
    return None


def sjis_row(lead: int) -> int | None:
    if 0x81 <= lead <= 0x9F:
        return lead - 0x81
    if 0xE0 <= lead <= 0xFC:
        return lead - 0xE0 + 31
    return None


def gb_four_byte_index(b1: int, b2: int, b3: int, b4: int) -> int:
    return ((b1 - 0x81) * 10 + (b2 - 0x30)) * 1260 + (b3 - 0x81) * 10 + (b4 - 0x30)


# --------------------------------------------------------------------------
# Probing Python's codecs
# --------------------------------------------------------------------------


def decode_one(data: bytes, encoding: str) -> int | None:
    """Return the single code point `data` decodes to, or None when invalid."""
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        return None
    if len(text) != 1:
        raise SystemExit(f"{encoding}: {data.hex()} decoded to {text!r}")
    return ord(text)


def probe_two_byte(encoding: str, leads, cols_of, trails, rows: int, columns: int):
    """Probe a two-byte code space; returns a row-major list with 0 = unmapped."""
    table = [0] * (rows * columns)
    mapped = 0
    for row, lead in enumerate(leads):
        for trail in trails:
            col = cols_of(trail)
            assert col is not None
            codepoint = decode_one(bytes([lead, trail]), encoding)
            if codepoint is None:
                continue
            if codepoint == 0:
                raise SystemExit(f"{encoding}: {lead:02X}{trail:02X} maps to NUL")
            if codepoint > 0xFFFF:
                raise SystemExit(f"{encoding}: {lead:02X}{trail:02X} maps to {codepoint:X}")
            table[row * columns + col] = codepoint
            mapped += 1
    return table, mapped


def probe_gb18030_two_byte():
    return probe_two_byte(
        "gb18030", list(GB_LEAD), gb_col, GB_TRAILS, GB_ROWS, GB_COLS
    )


def probe_big5():
    return probe_two_byte(
        "big5", list(BIG5_LEAD), big5_col, BIG5_TRAILS, BIG5_ROWS, BIG5_COLS
    )


def probe_shift_jis():
    return probe_two_byte(
        "shift_jis", SJIS_LEADS, sjis_col, SJIS_TRAILS, len(SJIS_LEADS), SJIS_COLS
    )


def four_byte_bytes(index: int) -> bytes:
    b1 = 0x81 + index // 12600
    b2 = 0x30 + (index // 1260) % 10
    b3 = 0x81 + (index // 10) % 126
    b4 = 0x30 + index % 10
    return bytes([b1, b2, b3, b4])


def probe_gb18030_four_byte():
    """Probe every GB18030 four-byte index that can be mapped.

    GB18030 maps two disjoint index regions: `0..=39419` onto the BMP (skipping the
    surrogate block and `U+FFFE`), and `189000..=1237575` onto `U+10000..=U+10FFFF`.
    The region in between is unmapped and must be rejected by the decoder.
    """
    mapping: dict[int, int] = {}
    unmapped: list[int] = []
    # BMP region plus the first ~11k indices of the unmapped middle band.
    for b1 in range(0x81, 0x85):
        for b2 in range(0x30, 0x3A):
            for b3 in range(0x81, 0xFF):
                for b4 in range(0x30, 0x3A):
                    index = gb_four_byte_index(b1, b2, b3, b4)
                    codepoint = decode_one(bytes([b1, b2, b3, b4]), "gb18030")
                    if codepoint is None:
                        unmapped.append(index)
                    else:
                        mapping[index] = codepoint
    bmp_max = max(mapping)
    # The astral region is a single linear run; decode it in one chunk (every
    # sequence is self-delimiting) and check the element count.
    astral_start = 189000
    astral_end = astral_start + (0x110000 - 0x10000) - 1
    blob = b"".join(four_byte_bytes(index) for index in range(astral_start, astral_end + 1))
    text = blob.decode("gb18030")
    if len(text) != astral_end - astral_start + 1:
        raise SystemExit("gb18030: astral four-byte region is not a linear run")
    for offset, character in enumerate(text):
        mapping[astral_start + offset] = ord(character)
    # Boundaries around both regions must be rejected.
    for index in (bmp_max + 1, bmp_max + 2, astral_start - 1, astral_end + 1):
        if decode_one(four_byte_bytes(index), "gb18030") is not None:
            raise SystemExit(f"gb18030: index {index} is unexpectedly mapped")
    unmapped = [index for index in unmapped if index < bmp_max]
    return mapping, sorted(unmapped), bmp_max


def compress_runs(mapping: dict[int, int]):
    """Compress index->code point into (start, code_point, length) runs."""
    runs = []
    for index in sorted(mapping):
        codepoint = mapping[index]
        if runs:
            start, first, length = runs[-1]
            if index == start + length and codepoint == first + length:
                runs[-1] = (start, first, length + 1)
                continue
        runs.append((index, codepoint, 1))
    return runs


# --------------------------------------------------------------------------
# A faithful model of the Rust decoder (see src/encoding/cjk.rs)
# --------------------------------------------------------------------------


class ReferenceDecoder:
    """Byte-at-a-time model of `openjiuwen_sysop::encoding::cjk::CjkDecoder`."""

    def __init__(self, encoding: str, two_byte, four_byte_runs):
        self.encoding = encoding
        self.two_byte = two_byte
        self.four_byte_runs = four_byte_runs
        self.out: list[str] = []
        self.invalid = False
        self.pending = b""

    def reset(self) -> None:
        self.out = []
        self.invalid = False
        self.pending = b""

    # -- table lookups ---------------------------------------------------
    def _pair(self, lead: int, trail: int):
        if self.encoding == "gb18030":
            row = lead - 0x81 if 0x81 <= lead <= 0xFE else None
            col = gb_col(trail)
            columns = GB_COLS
        elif self.encoding == "big5":
            row = lead - 0xA1 if 0xA1 <= lead <= 0xF9 else None
            col = big5_col(trail)
            columns = BIG5_COLS
        else:
            row = sjis_row(lead)
            col = sjis_col(trail)
            columns = SJIS_COLS
        if row is None or col is None:
            return None
        value = self.two_byte[row * columns + col]
        return value or None

    def _four_byte(self, index: int):
        runs = self.four_byte_runs
        lo, hi = 0, len(runs)
        while lo < hi:
            mid = (lo + hi) // 2
            if runs[mid][0] <= index:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            return None
        start, codepoint, length = runs[lo - 1]
        if index < start + length:
            return codepoint + (index - start)
        return None

    # -- decoding --------------------------------------------------------
    def push(self, chunk: bytes) -> str:
        data = self.pending + chunk
        self.pending = b""
        out: list[str] = []
        i = 0
        size = len(data)
        while i < size:
            byte = data[i]
            if byte < 0x80:
                out.append(chr(byte))
                i += 1
                continue
            if (
                self.encoding == "shift_jis"
                and SJIS_HALFWIDTH_START <= byte <= SJIS_HALFWIDTH_END
            ):
                out.append(chr(SJIS_HALFWIDTH_UNICODE + byte - SJIS_HALFWIDTH_START))
                i += 1
                continue
            if self.encoding == "gb18030" and byte >= 0x80 and i + 1 < size:
                # Any high byte followed by a digit starts a four-byte candidate; the
                # 0x80/0xFF leads Python still accepts here are simply never mapped.
                if 0x30 <= data[i + 1] <= 0x39:
                    if i + 3 >= size:
                        # Incomplete four-byte sequence: buffer the tail.
                        self.pending = data[i:]
                        break
                    index = None
                    if (
                        0x81 <= byte <= 0xFE
                        and 0x81 <= data[i + 2] <= 0xFE
                        and 0x30 <= data[i + 3] <= 0x39
                    ):
                        index = gb_four_byte_index(byte, data[i + 1], data[i + 2], data[i + 3])
                    codepoint = None if index is None else self._four_byte(index)
                    if codepoint is not None:
                        out.append(chr(codepoint))
                        i += 4
                        continue
                    self.invalid = True
                    out.append("\ufffd")
                    i += 1
                    continue
            if i + 1 >= size:
                # Incomplete two-byte sequence: buffer the tail.
                self.pending = data[i:]
                break
            codepoint = self._pair(byte, data[i + 1])
            if codepoint is not None:
                out.append(chr(codepoint))
                i += 2
                continue
            self.invalid = True
            out.append("\ufffd")
            i += 1
        text = "".join(out)
        self.out.append(text)
        return text

    def finish(self) -> str:
        text = ""
        if self.pending:
            # Python raises `incomplete multibyte sequence` and replaces the whole
            # remaining tail with one U+FFFD.
            self.invalid = True
            text = "\ufffd"
            self.out.append(text)
            self.pending = b""
        return text

    def result(self) -> str:
        return "".join(self.out)


def reference_decode(data: bytes, encoding: str, two_byte, four_byte_runs) -> str:
    decoder = ReferenceDecoder(encoding, two_byte, four_byte_runs)
    decoder.push(data)
    decoder.finish()
    return decoder.result()


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_two_byte(encoding, leads, cols_of, trails, table, columns):
    """Re-derive every pair from the table and compare with Python's codec."""
    checked = 0
    for row, lead in enumerate(leads):
        for trail in trails:
            expected = decode_one(bytes([lead, trail]), encoding)
            col = cols_of(trail)
            assert col is not None
            actual = table[row * columns + col] or None
            if expected != actual:
                raise SystemExit(
                    f"{encoding}: {lead:02X}{trail:02X} table={actual} python={expected}"
                )
            checked += 1
    return checked


def verify_pair_space_against_decoder(two_byte, encoding):
    """Every byte pair must decode through the pair lookup like Python does."""
    model = ReferenceDecoder(encoding, two_byte, [])
    checked = 0
    for lead in range(0x20, 0x100):
        for trail in range(0x20, 0x100):
            data = bytes([lead, trail])
            expected = data.decode(encoding, "replace")
            model.reset()
            actual = model.push(data) + model.finish()
            if actual != expected:
                raise SystemExit(
                    f"{encoding}: {data.hex()} model={actual!r} python={expected!r}"
                )
            checked += 1
    return checked


def verify_four_byte_space(runs):
    probed = 0
    for b1 in range(0x81, 0xFF):
        for b2 in range(0x30, 0x3A):
            for b3 in range(0x81, 0xFF):
                for b4 in range(0x30, 0x3A):
                    data = bytes([b1, b2, b3, b4])
                    expected = data.decode("gb18030", "replace")
                    actual = reference_decode(data, "gb18030", [], runs)
                    if actual != expected:
                        raise SystemExit(
                            f"gb18030: {data.hex()} model={actual!r} python={expected!r}"
                        )
                    probed += 1
    return probed


def verify_fuzz(models, rounds=600, max_len=24, seed=20261113):
    """Compare the decoder model with Python's replace-mode output."""
    rng = random.Random(seed)
    probed = 0
    for _ in range(rounds):
        length = rng.randint(0, max_len)
        if rng.random() < 0.5:
            # Bias towards CJK-shaped bytes so real sequences dominate.
            data = bytes(
                rng.choice([rng.choice(range(0x100)), rng.randrange(0x80, 0x100)])
                for _ in range(length)
            )
        else:
            data = bytes(rng.choice(range(0x100)) for _ in range(length))
        for encoding, (two_byte, runs) in models.items():
            expected = data.decode(encoding, "replace")
            actual = reference_decode(data, encoding, two_byte, runs)
            if actual != expected:
                raise SystemExit(
                    f"{encoding}: fuzz {data.hex()} model={actual!r} python={expected!r}"
                )
            probed += 1
    return probed


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

ALLOW_HEADER = "#![allow(clippy::unreadable_literal)]\n\n"

HEADER = """//! Generated lookup tables for the {name} decoder — DO NOT EDIT.
//!
//! Emitted by `crates/openjiuwen-sysop/scripts/gen_cjk_tables.py` from Python's
//! built-in `{codec}` codec. Regenerate with
//! `python3 crates/openjiuwen-sysop/scripts/gen_cjk_tables.py` instead of patching
//! by hand; adding `--check` verifies the committed tables.
//!
{notes}
"""


def fmt_u16_array(name: str, values: list[int], doc: str) -> str:
    lines = [
        f"/// {doc}",
        "#[rustfmt::skip]",
        f"pub(crate) static {name}: [u16; {len(values)}] = [",
    ]
    for index in range(0, len(values), 16):
        chunk = ", ".join(f"0x{value:04X}" for value in values[index : index + 16])
        lines.append(f"    {chunk},")
    lines.append("];")
    return "\n".join(lines)


def fmt_runs(name: str, runs, doc: str) -> str:
    lines = [
        f"/// {doc}",
        "#[rustfmt::skip]",
        f"pub(crate) static {name}: [(u32, u32, u32); {len(runs)}] = [",
    ]
    for start, codepoint, length in runs:
        lines.append(f"    (0x{start:06X}, 0x{codepoint:04X}, {length}),")
    lines.append("];")
    return "\n".join(lines)


def write(path: Path, contents: str, check: bool) -> bool:
    if check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != contents:
            print(f"stale: {path.relative_to(CRATE)}")
            return False
        print(f"ok: {path.relative_to(CRATE)}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    print(f"wrote {path.relative_to(CRATE)} ({len(contents)} bytes)")
    return True


def build_gb18030(check: bool) -> bool:
    table, mapped = probe_gb18030_two_byte()
    four_byte, unmapped, bmp_max = probe_gb18030_four_byte()
    runs = compress_runs(four_byte)
    modelled = verify_two_byte(
        "gb18030", list(GB_LEAD), gb_col, GB_TRAILS, table, GB_COLS
    )
    four_probed = verify_four_byte_space(runs)
    pair_probed = verify_pair_space_against_decoder(table, "gb18030")
    slots = GB_ROWS * GB_COLS
    unmapped_slots = slots - mapped
    dense_runs = len(compress_runs(dict(enumerate(table))))
    # The BMP region skips the surrogate block by jumping the code point, not the index.
    jump = next(
        index + 1 for index, codepoint in four_byte.items() if codepoint == 0xD7FF
    )
    unmapped_bmp = (
        f"{len(unmapped)} unmapped positions"
        if unmapped
        else f"no unmapped position (the surrogate block `0xD800..=0xDFFF` is skipped by a "
        f"code-point jump at index `0x{jump:04X}`)"
    )
    notes = f"""//! Two-byte area: lead `0x81..=0xFE` × trail (`0x40..=0x7E`, `0x80..=0xFE`), stored
//! row-major with {GB_COLS} columns. {mapped} of {slots} slots are mapped and {unmapped_slots} are
//! unmapped: the area is dense (unassigned positions decode to private-use code points,
//! exactly like Python), so this flat table is smaller than the {dense_runs}-run range
//! compression it was compared against.
//!
//! Four-byte area: index `((b1-0x81)*10 + (b2-0x30))*1260 + (b3-0x81)*10 + (b4-0x30)`
//! with `b2, b4 ∈ 0x30..=0x39`, compressed into {len(runs)} runs of
//! `(index_start, code_point_start, run_len)`.
//! Index region `0..=0x{bmp_max:04X}` maps to the BMP with {unmapped_bmp}; region
//! `189000..=1237575` maps linearly onto `U+10000..=U+10FFFF`; every other index is
//! unmapped and rejected by the decoder.
//!
//! Verified by the generator: {four_probed} four-byte codes, {modelled} modelled two-byte
//! codes and {pair_probed} two-byte combinations reproduce `bytes.decode("gb18030")`
//! exactly."""
    contents = ALLOW_HEADER + HEADER.format(name="GB18030/GBK", codec="gb18030", notes=notes)
    contents += (
        "\n/// Trail-column count of the two-byte table.\n"
        f"pub(crate) const TWO_BYTE_COLUMNS: usize = {GB_COLS};\n\n"
    )
    contents += fmt_u16_array(
        "TWO_BYTE",
        table,
        "Row-major two-byte map; `0` marks an unmapped pair (none in GB18030).",
    )
    contents += "\n\n"
    contents += (
        "/// Highest GB18030 four-byte index the codec maps.\n"
        f"pub(crate) const FOUR_BYTE_MAX_INDEX: u32 = 0x{max(four_byte):04X};\n\n"
    )
    contents += fmt_runs(
        "FOUR_BYTE_RUNS",
        runs,
        "Four-byte ranges: `(index_start, code_point_start, run_len)`, sorted by index.",
    )
    contents += "\n"
    return write(OUT_DIR / "gb18030.rs", contents, check)


def build_big5(check: bool) -> bool:
    table, mapped = probe_big5()
    modelled = verify_two_byte(
        "big5", list(BIG5_LEAD), big5_col, BIG5_TRAILS, table, BIG5_COLS
    )
    pair_probed = verify_pair_space_against_decoder(table, "big5")
    slots = BIG5_ROWS * BIG5_COLS
    unmapped = slots - mapped
    notes = f"""//! Two-byte area: lead `0xA1..=0xF9` × trail (`0x40..=0x7E`, `0xA1..=0xFE`), stored
//! row-major with {BIG5_COLS} columns. {mapped} of {slots} slots are mapped and {unmapped} are
//! unmapped (they decode as U+FFFD). The area is {100 * mapped // slots}% dense, so a flat
//! table beats run-length encoding.
//!
//! Python's `big5` codec only covers this lead range; the Big5-HKSCS extension
//! (leads `0x87..=0xA0`) is out of scope and decodes as U+FFFD instead of guessing.
//!
//! Verified by the generator: every one of the {modelled} modelled pairs decodes like
//! Python's `big5` codec, and {pair_probed} two-byte combinations match
//! `bytes.decode("big5")` exactly."""
    contents = ALLOW_HEADER + HEADER.format(name="Big5", codec="big5", notes=notes)
    contents += (
        "\n/// Trail-column count of the two-byte table.\n"
        f"pub(crate) const TWO_BYTE_COLUMNS: usize = {BIG5_COLS};\n\n"
    )
    contents += fmt_u16_array(
        "TWO_BYTE",
        table,
        "Row-major two-byte map; `0` marks an unmapped pair.",
    )
    contents += "\n"
    return write(OUT_DIR / "big5.rs", contents, check)


def build_shift_jis(check: bool) -> bool:
    table, mapped = probe_shift_jis()
    modelled = verify_two_byte(
        "shift_jis", SJIS_LEADS, sjis_col, SJIS_TRAILS, table, SJIS_COLS
    )
    pair_probed = verify_pair_space_against_decoder(table, "shift_jis")
    slots = len(SJIS_LEADS) * SJIS_COLS
    unmapped = slots - mapped
    for byte in range(SJIS_HALFWIDTH_START, SJIS_HALFWIDTH_END + 1):
        expected = SJIS_HALFWIDTH_UNICODE + byte - SJIS_HALFWIDTH_START
        actual = decode_one(bytes([byte]), "shift_jis")
        if actual != expected:
            raise SystemExit(f"shift_jis: {byte:02X} single-byte table mismatch")
    notes = f"""//! Two-byte area: leads `0x81..=0x9F` and `0xE0..=0xFC` × trail (`0x40..=0x7E`,
//! `0x80..=0xFC`), stored row-major with {SJIS_COLS} columns and {len(SJIS_LEADS)} lead rows (rows
//! `0..=30` are leads `0x81..=0x9F`, rows `31..=59` are leads `0xE0..=0xFC`).
//! {mapped} of {slots} slots are mapped and {unmapped} are unmapped.
//!
//! Halfwidth katakana (`0xA1..=0xDF` -> `U+FF61..=U+FF9F`) is positional and needs no
//! table; the generator checked all 63 bytes against `bytes.decode("shift_jis")`.
//! This is JIS X 0208 (`shift_jis`), not the CP932 vendor extension.
//!
//! Verified by the generator: every one of the {modelled} modelled pairs decodes like
//! Python's `shift_jis` codec, and {pair_probed} two-byte combinations match
//! `bytes.decode("shift_jis")` exactly."""
    contents = ALLOW_HEADER + HEADER.format(name="Shift-JIS", codec="shift_jis", notes=notes)
    contents += (
        "\n/// Trail-column count of the two-byte table.\n"
        f"pub(crate) const TWO_BYTE_COLUMNS: usize = {SJIS_COLS};\n\n"
    )
    contents += fmt_u16_array(
        "TWO_BYTE",
        table,
        "Row-major two-byte map; `0` marks an unmapped pair.",
    )
    contents += "\n"
    return write(OUT_DIR / "shift_jis.rs", contents, check)


def main(argv: list[str]) -> int:
    unknown = [argument for argument in argv[1:] if argument != "--check"]
    if unknown:
        print(f"unknown arguments: {' '.join(unknown)}", file=sys.stderr)
        return 2
    check = "--check" in argv[1:]
    ok = build_gb18030(check)
    ok &= build_big5(check)
    ok &= build_shift_jis(check)

    # Cross-encoding fuzz of the decoder model against Python's error handling.
    gb_table, _ = probe_gb18030_two_byte()
    b5_table, _ = probe_big5()
    sj_table, _ = probe_shift_jis()
    gb_runs = compress_runs(probe_gb18030_four_byte()[0])
    fuzzed = verify_fuzz(
        {
            "gb18030": (gb_table, gb_runs),
            "big5": (b5_table, []),
            "shift_jis": (sj_table, []),
        }
    )
    print(f"verified {fuzzed} fuzzed streams against Python's errors='replace' output")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
