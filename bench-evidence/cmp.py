import re
def parse(p):
    rows, cur = {}, None
    for line in open(p, errors="ignore"):
        m = re.search(r"===== CASE \d+/\d+ (\S+.*\S) =====", line)
        if m: cur = m.group(1); rows[cur] = {}
        elif cur:
            kv = re.match(r"(status|generated_tokens|expected): (.+)", line.strip())
            if kv: rows[cur][kv.group(1)] = kv.group(2)
    return rows
a = parse("/home/ubuntu/eval-patched-full.trace")
b = parse("/home/ubuntu/eval-baseline-full.trace")
pa = sum(1 for r in a.values() if r.get("status")=="PASSED")
pb = sum(1 for r in b.values() if r.get("status")=="PASSED")
for qid in sorted(set(a) | set(b)):
    ra, rb = a.get(qid, {}), b.get(qid, {})
    sa, sb = ra.get("status","?"), rb.get("status","?")
    flag = "" if sa == sb else "   <-- FLIP"
    print("%-45s patched=%s(%s tok)  baseline=%s(%s tok)%s" % (
        qid, sa, ra.get("generated_tokens","?"), sb, rb.get("generated_tokens","?"), flag))
print("TOTALS: patched %d/%d passed, baseline %d/%d passed" % (pa, len(a), pb, len(b)))
