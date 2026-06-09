import argparse
import csv
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
DB_PATH = DATA_DIR / "metadata.db"

parser = argparse.ArgumentParser()
parser.add_argument("--physics", type=str, default=None)
args = parser.parse_args()

# --------------------------------- export settings -----------------------------------
PHYSICS = args.physics
assert PHYSICS is None or PHYSICS.isidentifier(), (
    "physics must be a valid identifier (used as a table name)"
)

# (table, CSV file name in data/abc); the per-physics table is added only with
EXPORTS = [("geometry", DATA_DIR / "metadata_geometry.csv")]
if PHYSICS is not None:
    EXPORTS.append((f"sim_{PHYSICS}", DATA_DIR / f"metadata_{PHYSICS}.csv"))

# ---------------------------------- dump to csv --------------------------------------
connection = sqlite3.connect(DB_PATH)

for table, out_path in EXPORTS:
    cursor = connection.execute(f"SELECT * FROM {table} ORDER BY STL_ID")
    columns = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")

connection.close()
