"""Convert sheet1 of an .xlsx to TSV with stdlib only (no openpyxl)."""
import sys, zipfile, re
from xml.etree import ElementTree as ET
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

def col_index(ref):
    m = re.match(r"([A-Z]+)", ref or "")
    if not m:
        return None
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1

def convert(src, dst, skip=1):
    z = zipfile.ZipFile(src)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    sheet = sorted(n for n in z.namelist()
                   if re.match(r"xl/worksheets/sheet\d+\.xml$", n))[0]
    n_out = 0
    with open(dst, "w") as fh:
        for i, (_, el) in enumerate(
                e for e in ET.iterparse(z.open(sheet), events=("end",))
                if e[1].tag == NS + "row"):
            if i < skip:
                el.clear(); continue
            cells = {}
            for c in el.findall(NS + "c"):
                v = c.find(NS + "v")
                txt = "" if v is None else (v.text or "")
                if c.get("t") == "s" and txt:
                    txt = shared[int(txt)]
                if c.get("t") == "inlineStr":
                    txt = "".join(t.text or "" for t in c.iter(NS + "t"))
                ci = col_index(c.get("r"))
                if ci is not None:
                    cells[ci] = txt.strip().replace("\t", " ").replace("\n", " ")
            if cells:
                width = max(cells) + 1
                fh.write("\t".join(cells.get(j, "") for j in range(width)) + "\n")
                n_out += 1
            el.clear()
    return n_out

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    print(f"{dst}: {convert(src, dst, skip)} rows")
