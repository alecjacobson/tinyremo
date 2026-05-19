#!/usr/bin/env python3
"""
Usage: ./compare.py snapshots/before.txt snapshots/after.txt

Joins on (scenario, N), prints speedup = t_before / t_after.
Lines with speedup > 1.10 are marked ◄ (faster after).
Lines with speedup < 0.90 are marked ▼ (slower after — regression).
"""
import sys

def parse(fname):
    rows = {}
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                scenario = parts[0]
                N        = int(parts[1])
                t        = float(parts[4])
                rows[(scenario, N)] = t
            except (ValueError, IndexError):
                continue
    return rows

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

before = parse(sys.argv[1])
after  = parse(sys.argv[2])

keys = sorted(set(before) & set(after))
if not keys:
    print("No matching rows found.")
    sys.exit(1)

print(f"{'scenario':<32} {'N':>7}  {'before':>12}  {'after':>12}  {'speedup':>8}")
print("-" * 78)

last_prefix = None
for scenario, N in keys:
    prefix = scenario.split('/')[0]
    if prefix != last_prefix and last_prefix is not None:
        print()
    last_prefix = prefix

    t_b = before[(scenario, N)]
    t_a = after[(scenario, N)]
    speedup = t_b / t_a if t_a > 0 else float('nan')
    flag = "  ◄" if speedup > 1.10 else ("  ▼" if speedup < 0.90 else "")
    print(f"{scenario:<32} {N:>7}  {t_b:>12.4e}  {t_a:>12.4e}  {speedup:>8.2f}x{flag}")
