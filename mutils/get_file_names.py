from pathlib import Path

path = "F:/Data2/backfill_1min/bothaville_70091100060/"
dataDir = Path(path)
files = sorted(dataDir.glob("*.csv"))
files = [int(x.stem) for x in files]

print(files)

[1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 3]

[1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 3]
