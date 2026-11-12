"""
How many alignment columns each reference position gets, and the padding that
puts every isolate in them.

`make_aligned_msa.py` builds one CELL per reference position. A cell normally
holds that position's single base. An insertion puts the whole ALT allele in the
cell at its start and blanks the rest of the REF span with the dummy character
`X`, which is stripped at write time. So a cell can hold 0, 1 or many characters,
and the alignment's width at a position is whatever the widest isolate needs.

The Jul-2025 upstream extractor got this wrong by overwriting downstream bases
(fixed 2026-10-22, see [msa-insertion-bug]). The 2026-10-22 fix then derived the
width from BOOKKEEPING -- `insertion_dict[idx] = max(len(ALT) - len(REF))` --
and padded each cell to `1 + insertion_dict[idx]`. That holds only while REF
spans do not overlap. Where two variants' REF spans DO overlap, both ALTs land
in cells inside one span, the bookkeeping counts neither against the other, and
the isolate comes out LONGER than the padded reference:

    embB SAMN08892416  3425 vs 3423      ethA SAMEA2535140  1738 vs 1737
    rrs  SAMN07766293  1830 vs 1813

which is an `assert len(seq) == len(new_ref_seq)` failure that killed the
embB / ethA / rrs re-extractions after they had written part of a FASTA.

So take the width from the CELLS THEMSELVES: per position, the longest cell any
isolate realizes (at least 1, since the reference always has its own base). Then
every row is that width by construction, no isolate can overflow it, and nothing
is dropped. For the ordinary case -- a single-base REF -- the width is
`1 + max insertion`, exactly what the bookkeeping gave, so alignments that
already built cleanly are unchanged (`eis` reproduces byte for byte).
"""


def clean(cell):
    """A cell's real characters: the X placeholders are not alignment content."""
    return cell.replace("X", "")


def realized_widths(cells_per_isolate, n_positions):
    """-> [columns per reference position], from an iterable of cell lists."""
    widths = [1] * n_positions
    for cells in cells_per_isolate:
        if len(cells) != n_positions:
            raise ValueError(f"{len(cells)} cells, expected {n_positions}")
        for i, cell in enumerate(cells):
            n = len(clean(cell))
            if n > widths[i]:
                widths[i] = n
    return widths


def pad_cells(cells, widths):
    """One isolate's cells -> its alignment row, gap-padded to `widths`."""
    out = []
    for cell, w in zip(cells, widths):
        c = clean(cell)
        if len(c) > w:
            raise ValueError(f"cell {c!r} wider than its {w} column(s)")
        out.append(c + "-" * (w - len(c)))
    return "".join(out)


def pad_reference(ref_bases, widths):
    """The H37Rv row: its own base, then a gap per extra column."""
    return "".join(b + "-" * (w - 1) for b, w in zip(ref_bases, widths))
