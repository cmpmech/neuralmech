import argparse
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = (BASE_DIR / "../../data/abc").resolve()
DB_PATH = DATA_DIR / "metadata.db"

parser = argparse.ArgumentParser()
parser.add_argument("--physics", type=str, default=None)
args = parser.parse_args()

# --------------------------------- metadata schema -----------------------------------
PHYSICS = args.physics
assert PHYSICS is None or PHYSICS.isidentifier(), (
    "physics must be a valid identifier (used as a table name)"
)

# physics-independent geometry stats, written by voxelize_stl.py
GEOMETRY_TABLE = """
CREATE TABLE IF NOT EXISTS geometry (
    STL_ID       INTEGER PRIMARY KEY,
    CONNECTED    INTEGER,
    VOXELS_SOLID INTEGER,
    VOXELS_TOTAL INTEGER,
    VOXELS_NX    INTEGER,
    VOXELS_NY    INTEGER,
    VOXELS_NZ    INTEGER
)
"""

# ------------------------------------ initialize -------------------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)

connection = sqlite3.connect(DB_PATH)
connection.execute(GEOMETRY_TABLE)
tables = ["geometry"]

# per-physics simulation results, written by the solve drivers; skipped without
if PHYSICS is not None:
    connection.execute(f"""
    CREATE TABLE IF NOT EXISTS sim_{PHYSICS} (
        STL_ID         INTEGER PRIMARY KEY,
        STL_SIM_TIME   REAL,
        VOXEL_SIM_TIME REAL,
        VOXEL_L2       REAL,
        VOXEL_LINF     REAL
    )
    """)
    tables.append(f"sim_{PHYSICS}")

connection.commit()
connection.close()

print(f"initialized {DB_PATH}")
print(f"\ttables: {', '.join(tables)}")
