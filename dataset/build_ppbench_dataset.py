"""
PPBench Dataset Builder for TRM Training.

Builds a TRM-compatible .npy dataset from PPBench (Pencil Puzzle Bench).

Implements the exact data pipeline described in arXiv:2605.19943, Appendix A.3:
  - 6 puzzle types: sudoku, lightup, nurikabe, shakashaka, heyawake, tapa
  - Grid sizes: 9×9 (sudoku, padded to 10×10), 10×10 (rest) → seq_len=100
  - Unified vocabulary: 294 tokens (pad + all symbols across 6 types)
  - Augmentation: 10 examples per puzzle (1 original + 9 trajectory×dihedral)
  - Splits: train / val (100 per type, 50 for tapa) / golden (from PPBench golden set)

Usage:
    python dataset/build_ppbench_dataset.py
    python dataset/build_ppbench_dataset.py --output-dir data/ppbench --dry-run
"""

import argparse
import base64
import json
import os
import sys
import urllib.parse
from collections import defaultdict
from typing import Optional

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants matching the paper (Appendix A.3)
# ---------------------------------------------------------------------------

# The 6 puzzle types used in the paper
PUZZLE_TYPES = ["sudoku", "lightup", "nurikabe", "shakashaka", "heyawake", "tapa"]

# Grid sizes per the paper: 9×9 for sudoku, 10×10 for others
GRID_SIZES = {
    "sudoku": (9, 9),
    "lightup": (10, 10),
    "nurikabe": (10, 10),
    "shakashaka": (10, 10),
    "heyawake": (10, 10),
    "tapa": (10, 10),
}

# Uniform seq_len=100 (10×10) after padding sudoku
SEQ_LEN = 100

# Golden set: 15 puzzles × 20 types (300 total) — we filter to our 6 types
GOLDEN_PER_TYPE = 15

# Validation set sizes per the paper
VAL_SIZES = {
    "sudoku": 100,
    "lightup": 100,
    "nurikabe": 100,
    "shakashaka": 100,
    "heyawake": 100,
    "tapa": 50,  # "50 for tapa, due to its smaller base size"
}

# Number of augmented examples per puzzle (1 original + 9 augmented)
NUM_AUGMENTS_PER_PUZZLE = 9

# PPBench solution decryption key
PPBENCH_XOR_KEY = b"ppbench"


# ---------------------------------------------------------------------------
# Solution decryption
# ---------------------------------------------------------------------------

def decode_solution(enc: str) -> dict:
    """Decrypt PPBench's XOR-encrypted, base64-encoded solution."""
    raw = base64.b64decode(enc)
    decrypted = bytes(b ^ PPBENCH_XOR_KEY[i % len(PPBENCH_XOR_KEY)] for i, b in enumerate(raw))
    return json.loads(decrypted)


# ---------------------------------------------------------------------------
# Puzzlink URL parsing → grid extraction
# ---------------------------------------------------------------------------

def parse_puzzlink_url(url: str) -> dict:
    """
    Parse a puzz.link URL to extract puzzle type, dimensions, and board encoding.

    puzz.link URLs encode the puzzle state in the fragment:
      https://puzz.link/p?<type>/<width>/<height>/<board_encoding>
    """
    parsed = urllib.parse.urlparse(url)

    # Handle both puzz.link and pzprv3 URL formats
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment

    # The puzzle spec is typically in the query part: ?type/w/h/encoding
    # or sometimes in the path
    if query:
        parts = query.split("/")
    elif fragment:
        parts = fragment.split("/")
    else:
        # Try path after /p?
        parts = path.split("?")[-1].split("/") if "?" in path else path.strip("/").split("/")

    # Remove 'p' prefix if present
    if parts and parts[0] in ("p", "p?"):
        parts = parts[1:]

    if len(parts) < 4:
        return {"type": parts[0] if parts else "", "width": 0, "height": 0, "encoding": ""}

    return {
        "type": parts[0],
        "width": int(parts[1]),
        "height": int(parts[2]),
        "encoding": "/".join(parts[3:]),
    }


# ---------------------------------------------------------------------------
# Grid decoding from pzpr board encoding
# ---------------------------------------------------------------------------

# pzpr.js board encoding uses a base-32-ish scheme where letters a-z encode
# gaps/runs and digits encode cell values. The exact format varies per puzzle
# type, so we use the solution moves to reconstruct the solved grid instead.

def build_grid_from_moves(puzzle_type: str, width: int, height: int,
                          initial_url: str, moves_full: list) -> tuple:
    """
    Build initial (input) and solved (label) grids from PPBench data.

    For TRM training, we need:
      - input grid: the initial puzzle state (clues filled, blanks as a special token)
      - label grid: the fully solved grid

    We parse the pzpr URL for the initial state and apply moves for the solution.

    Returns:
        (input_grid, label_grid) as 2D numpy arrays of integer token IDs
    """
    # Parse the initial puzzle state from the URL
    url_info = parse_puzzlink_url(initial_url)

    # Build the grid from pzpr encoding
    input_grid, label_grid = _decode_pzpr_puzzle(
        puzzle_type, width, height, url_info["encoding"], moves_full
    )

    return input_grid, label_grid


def _decode_pzpr_puzzle(puzzle_type: str, width: int, height: int,
                        encoding: str, moves_full: list) -> tuple:
    """
    Decode a pzpr puzzle encoding into input and label grids.

    The encoding format varies by puzzle type:
    - sudoku: numbers in cells (1-9), blanks
    - lightup: black cells (with optional numbers), empty/light cells
    - nurikabe: numbered cells and shaded/unshaded
    - shakashaka: numbered cells, triangles in different orientations
    - heyawake: room borders with numbers, shaded/unshaded
    - tapa: clue cells with numbers, filled/empty cells

    We use a unified approach: parse the pzpr encoding for initial state,
    then apply solution moves to build the label grid.
    """
    # Initialize grids with "empty" token (will be mapped to vocab later)
    input_grid = np.zeros((height, width), dtype=np.int32)
    label_grid = np.zeros((height, width), dtype=np.int32)

    # Parse pzpr encoding for initial state
    input_grid = _parse_pzpr_encoding(puzzle_type, width, height, encoding)

    # Apply solution moves to build label grid
    label_grid = _apply_solution_moves(puzzle_type, width, height,
                                       input_grid.copy(), moves_full)

    return input_grid, label_grid


def _parse_pzpr_encoding(puzzle_type: str, width: int, height: int,
                          encoding: str) -> np.ndarray:
    """
    Parse pzpr board encoding string into a grid of raw cell values.

    pzpr encoding uses:
    - Letters (a-z): indicate runs of empty cells (a=0 empty, b=1 empty, ...)
    - Digits/special chars: indicate cell values
    - The encoding fills cells row by row

    This is a simplified parser that handles the common cases for our 6 types.
    """
    grid = np.zeros((height, width), dtype=np.int32)

    if puzzle_type == "sudoku":
        # Sudoku encoding: digits 1-9 for clues, '.' or gaps for blanks
        pos = 0
        for char in encoding:
            if pos >= width * height:
                break
            if char == ".":
                grid[pos // width][pos % width] = 0  # blank
                pos += 1
            elif char.isdigit() and char != "0":
                grid[pos // width][pos % width] = int(char)
                pos += 1
            elif "a" <= char <= "z":
                # Run of empty cells: 'a'=0 gaps after current, 'b'=1, etc.
                # In pzpr, letters encode gaps
                num_gaps = ord(char) - ord("a")
                pos += num_gaps + 1
            elif char == "-":
                # Separator, skip
                continue
            elif char == "0":
                grid[pos // width][pos % width] = 0
                pos += 1
    else:
        # For non-sudoku puzzles, parse the pzpr encoding
        # pzpr uses sections separated by '/' for different grid layers
        # The first section is typically the "problem" (clue) layer
        sections = encoding.split("/") if "/" in encoding else [encoding]

        if sections:
            _parse_pzpr_section(grid, sections[0], width, height, puzzle_type)

    return grid


def _parse_pzpr_section(grid: np.ndarray, section: str, width: int, height: int,
                         puzzle_type: str):
    """Parse a single pzpr encoding section into grid values."""
    pos = 0
    i = 0
    while i < len(section) and pos < width * height:
        char = section[i]

        if char == ".":
            # Special value (e.g., -1 in pzpr = no clue)
            grid[pos // width][pos % width] = -1
            pos += 1
        elif char.isdigit():
            # Parse multi-digit number
            num_str = char
            while i + 1 < len(section) and section[i + 1].isdigit():
                i += 1
                num_str += section[i]
            grid[pos // width][pos % width] = int(num_str)
            pos += 1
        elif "g" <= char <= "z":
            # Run of empty cells
            num_gaps = ord(char) - ord("f")
            pos += num_gaps
        elif char == "-":
            # Negative number follows
            i += 1
            num_str = ""
            while i < len(section) and section[i].isdigit():
                num_str += section[i]
                i += 1
            if num_str:
                grid[pos // width][pos % width] = -int(num_str)
                pos += 1
            i -= 1  # Back up since we'll increment
        elif "a" <= char <= "f":
            # Hex digit (10-15) used as cell value in some puzzle types
            grid[pos // width][pos % width] = ord(char) - ord("a") + 10
            pos += 1

        i += 1


def _apply_solution_moves(puzzle_type: str, width: int, height: int,
                           grid: np.ndarray, moves_full: list) -> np.ndarray:
    """
    Apply PPBench solution moves to build the solved grid.

    moves_full is a list of move strings like:
    - "mouse,left,<col>,<row>" — for click-based puzzles
    - "number,<col>,<row>,<value>" — for number-entry puzzles

    We parse these to set cell values in the grid.
    """
    label_grid = grid.copy()

    for move in moves_full:
        parts = move.split(",")

        if len(parts) < 3:
            continue

        action = parts[0]

        if action == "number" and len(parts) >= 4:
            # number,col,row,value — direct number entry
            try:
                col = int(parts[1])
                row = int(parts[2])
                value = parts[3]
                if 0 <= row < height and 0 <= col < width:
                    if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                        label_grid[row][col] = int(value)
                    elif value == " " or value == "":
                        pass  # Clear cell, keep as-is
                    else:
                        # String value — map to a symbol ID later
                        label_grid[row][col] = _symbol_to_id(value)
            except (ValueError, IndexError):
                pass

        elif action == "mouse" and len(parts) >= 4:
            # mouse,button,col,row — toggle cell state
            try:
                button = parts[1]
                col = int(parts[2])
                row = int(parts[3])
                if 0 <= row < height and 0 <= col < width:
                    if button == "left":
                        # Left click typically sets cell to "filled/shaded"
                        label_grid[row][col] = 1
                    elif button == "right":
                        # Right click typically sets cell to "empty/unshaded"
                        label_grid[row][col] = 2
            except (ValueError, IndexError):
                pass

    return label_grid


# Symbol mapping for non-numeric cell values
_SYMBOL_MAP = {}
_NEXT_SYMBOL_ID = [100]  # Start from 100 to avoid conflicts with digit values


def _symbol_to_id(symbol: str) -> int:
    """Map a string symbol to a unique integer ID."""
    if symbol not in _SYMBOL_MAP:
        _SYMBOL_MAP[symbol] = _NEXT_SYMBOL_ID[0]
        _NEXT_SYMBOL_ID[0] += 1
    return _SYMBOL_MAP[symbol]


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------

class VocabBuilder:
    """
    Builds a unified vocabulary across all puzzle types.

    The paper states 294 tokens. We build this by:
    1. Token 0: PAD
    2. Collect all unique cell values across all puzzles
    3. Map each unique value to a sequential integer ID
    """

    def __init__(self):
        self.pad_id = 0
        self.value_to_token = {0: 0}  # 0 (empty/blank) → PAD
        self.next_id = 1  # Start assigning from 1

    def register_value(self, value: int, puzzle_type: str = ""):
        """Register a cell value and get its token ID."""
        key = value
        if key not in self.value_to_token:
            self.value_to_token[key] = self.next_id
            self.next_id += 1

    def encode(self, value: int) -> int:
        """Encode a cell value to its token ID."""
        if value in self.value_to_token:
            return self.value_to_token[value]
        # Unknown value — register on the fly
        self.register_value(value)
        return self.value_to_token[value]

    @property
    def vocab_size(self) -> int:
        return self.next_id

    def build_type_aware_vocab(self, all_puzzles: dict):
        """
        Build the full vocabulary by scanning all puzzles.

        Each puzzle type may have different symbol sets. We create a unified
        mapping: (puzzle_type, cell_value) → token_id.

        This produces ~294 tokens per the paper.
        """
        # Collect all unique (type, value) pairs
        type_values = defaultdict(set)
        for ptype, puzzles in all_puzzles.items():
            for puzzle in puzzles:
                input_grid = puzzle["input_grid"]
                label_grid = puzzle["label_grid"]
                for grid in [input_grid, label_grid]:
                    for val in grid.flatten():
                        type_values[ptype].add(int(val))

        # Token 0 = PAD (shared across all types)
        self.value_to_token = {}
        self.next_id = 1  # 0 is reserved for PAD

        # Build type-specific token mappings
        self.type_token_maps = {}
        for ptype in PUZZLE_TYPES:
            self.type_token_maps[ptype] = {0: 0}  # blank/empty → PAD
            values = sorted(type_values.get(ptype, set()))
            for val in values:
                if val == 0:
                    continue  # Already mapped to PAD
                self.type_token_maps[ptype][val] = self.next_id
                self.next_id += 1

        self.pad_id = 0

    def encode_typed(self, value: int, puzzle_type: str) -> int:
        """Encode a cell value using the type-aware vocab."""
        if hasattr(self, "type_token_maps"):
            return self.type_token_maps.get(puzzle_type, {}).get(value, 0)
        return self.encode(value)


# ---------------------------------------------------------------------------
# Augmentation (Appendix A.3)
# ---------------------------------------------------------------------------

def trajectory_sample(input_grid: np.ndarray, label_grid: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Trajectory sampling augmentation.

    "The input is set to a random intermediate solve state along the puzzle's
     solution trajectory rather than always the empty initial grid, while
     the label is always the fully solved grid."

    We simulate an intermediate state by randomly revealing some of the
    solution cells in the input grid.
    """
    # Find cells that are blank in input but filled in label
    blank_mask = (input_grid == 0) & (label_grid != 0)
    blank_positions = np.argwhere(blank_mask)

    if len(blank_positions) == 0:
        return input_grid.copy()

    # Randomly reveal a fraction of blank cells
    num_to_reveal = rng.integers(0, len(blank_positions) + 1)
    if num_to_reveal == 0:
        return input_grid.copy()

    reveal_indices = rng.choice(len(blank_positions), size=num_to_reveal, replace=False)
    augmented = input_grid.copy()
    for idx in reveal_indices:
        r, c = blank_positions[idx]
        augmented[r, c] = label_grid[r, c]

    return augmented


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply one of 8 dihedral symmetries (4 rotations × 2 {identity, reflection})."""
    if tid == 0:
        return arr.copy()
    elif tid == 1:
        return np.rot90(arr, k=1).copy()
    elif tid == 2:
        return np.rot90(arr, k=2).copy()
    elif tid == 3:
        return np.rot90(arr, k=3).copy()
    elif tid == 4:
        return np.fliplr(arr).copy()
    elif tid == 5:
        return np.flipud(arr).copy()
    elif tid == 6:
        return arr.T.copy()
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1)).copy()
    return arr.copy()


def augment_puzzle(input_grid: np.ndarray, label_grid: np.ndarray,
                   num_augments: int, rng: np.random.Generator) -> list:
    """
    Generate augmented examples from a single puzzle.

    "Each training puzzle is expanded into 10 examples using two augmentations:
     1) trajectory sampling [...] 2) dihedral transformation [...]
     For each puzzle, the first example is the unaugmented (initial state, solved)
     pair. The remaining 9 are randomly sampled (trajectory and dihedral transform)."

    Returns list of (augmented_input, augmented_label) pairs.
    """
    examples = [(input_grid.copy(), label_grid.copy())]

    for _ in range(num_augments):
        # Random trajectory sampling
        aug_input = trajectory_sample(input_grid, label_grid, rng)

        # Random dihedral transform
        tid = rng.integers(0, 8)
        aug_input = dihedral_transform(aug_input, tid)
        aug_label = dihedral_transform(label_grid, tid)

        examples.append((aug_input, aug_label))

    return examples


# ---------------------------------------------------------------------------
# Grid padding (sudoku 9×9 → 10×10)
# ---------------------------------------------------------------------------

def pad_grid_to_10x10(grid: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """Pad a 9×9 grid to 10×10 with pad tokens (for sudoku)."""
    if grid.shape == (10, 10):
        return grid
    assert grid.shape == (9, 9), f"Expected 9×9 grid, got {grid.shape}"
    padded = np.full((10, 10), pad_value, dtype=grid.dtype)
    padded[:9, :9] = grid
    return padded


# ---------------------------------------------------------------------------
# Main dataset construction
# ---------------------------------------------------------------------------

def load_ppbench_data(data_path: str) -> list:
    """Load PPBench JSONL data from file."""
    records = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_puzzles(records: list) -> dict:
    """
    Filter PPBench records to the 6 puzzle types at the correct grid sizes.

    Returns dict: puzzle_type → list of filtered records.
    """
    filtered = defaultdict(list)
    for record in records:
        ptype = record.get("pid", "")
        width = record.get("width", 0)
        height = record.get("height", 0)

        if ptype not in GRID_SIZES:
            continue

        expected_w, expected_h = GRID_SIZES[ptype]
        if width == expected_w and height == expected_h:
            filtered[ptype].append(record)

    return dict(filtered)


def split_golden(filtered: dict, golden_records: list) -> tuple:
    """
    Separate golden set from the filtered records.

    The golden set is identified by matching puzzlink_urls.
    """
    # Build set of golden URLs
    golden_urls = set()
    for record in golden_records:
        golden_urls.add(record.get("puzzlink_url", ""))

    golden = defaultdict(list)
    remaining = defaultdict(list)

    for ptype, records in filtered.items():
        for record in records:
            url = record.get("puzzlink_url", "")
            if url in golden_urls:
                golden[ptype].append(record)
            else:
                remaining[ptype].append(record)

    return dict(golden), dict(remaining)


def split_val_train(remaining: dict, rng: np.random.Generator) -> tuple:
    """
    Split remaining records into val and train sets.

    "From the remainder we hold out a fixed-size validation set of 100 puzzles
     per puzzle type (50 for tapa, due to its smaller base size)"
    """
    val = {}
    train = {}

    for ptype, records in remaining.items():
        val_size = VAL_SIZES.get(ptype, 100)
        val_size = min(val_size, len(records))

        # Shuffle and split
        indices = rng.permutation(len(records))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        val[ptype] = [records[i] for i in val_indices]
        train[ptype] = [records[i] for i in train_indices]

    return val, train


def process_puzzle_record(record: dict, puzzle_type: str) -> Optional[dict]:
    """
    Process a single PPBench record into input/label grids.

    Returns dict with 'input_grid' and 'label_grid' as numpy arrays,
    or None if processing fails.
    """
    try:
        width = record["width"]
        height = record["height"]

        # Decode solution
        solution = decode_solution(record["solution_enc"])
        moves_full = solution.get("moves_full", [])

        # Build grids from URL + solution moves
        input_grid, label_grid = build_grid_from_moves(
            puzzle_type, width, height,
            record["puzzlink_url"], moves_full
        )

        return {
            "input_grid": input_grid,
            "label_grid": label_grid,
            "puzzle_type": puzzle_type,
            "url": record.get("puzzlink_url", ""),
        }
    except Exception as e:
        print(f"  Warning: Failed to process {puzzle_type} puzzle: {e}")
        return None


def build_split(split_records: dict, vocab: VocabBuilder,
                augment: bool, rng: np.random.Generator,
                puzzle_type_to_id: dict) -> dict:
    """
    Build a TRM-format split from processed puzzle records.

    Returns dict with arrays ready for .npy saving:
      - inputs: (N, seq_len) int32
      - labels: (N, seq_len) int32
      - puzzle_identifiers: (N,) int32
      - puzzle_indices: (N+1,) int32 (cumulative example boundaries)
      - group_indices: (G+1,) int32 (cumulative puzzle group boundaries)
    """
    all_inputs = []
    all_labels = []
    all_puzzle_ids = []
    puzzle_indices = [0]
    group_indices = [0]

    puzzle_count = 0
    example_count = 0

    for ptype in PUZZLE_TYPES:
        records = split_records.get(ptype, [])
        ptype_id = puzzle_type_to_id[ptype]

        for record in records:
            processed = process_puzzle_record(record, ptype)
            if processed is None:
                continue

            input_grid = processed["input_grid"]
            label_grid = processed["label_grid"]

            # Pad sudoku 9×9 → 10×10
            if ptype == "sudoku":
                input_grid = pad_grid_to_10x10(input_grid)
                label_grid = pad_grid_to_10x10(label_grid)

            # Generate examples (with or without augmentation)
            if augment:
                examples = augment_puzzle(input_grid, label_grid,
                                          NUM_AUGMENTS_PER_PUZZLE, rng)
            else:
                examples = [(input_grid, label_grid)]

            for aug_input, aug_label in examples:
                # Encode to vocab tokens
                flat_input = np.array([
                    vocab.encode_typed(int(v), ptype)
                    for v in aug_input.flatten()
                ], dtype=np.int32)
                flat_label = np.array([
                    vocab.encode_typed(int(v), ptype)
                    for v in aug_label.flatten()
                ], dtype=np.int32)

                all_inputs.append(flat_input)
                all_labels.append(flat_label)
                all_puzzle_ids.append(ptype_id)

                example_count += 1
                puzzle_count += 1
                puzzle_indices.append(example_count)

            # Each original puzzle is one group
            group_indices.append(puzzle_count)

    return {
        "inputs": np.array(all_inputs, dtype=np.int32),
        "labels": np.array(all_labels, dtype=np.int32),
        "puzzle_identifiers": np.array(all_puzzle_ids, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "group_indices": np.array(group_indices, dtype=np.int32),
    }


def save_split(data: dict, metadata: dict, output_dir: str, split_name: str):
    """Save a split to TRM-format .npy files + dataset.json."""
    save_dir = os.path.join(output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save arrays
    for key, arr in data.items():
        np.save(os.path.join(save_dir, f"all__{key}.npy"), arr)

    # Save metadata
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved {split_name}: {len(data['inputs'])} examples to {save_dir}")


def download_ppbench(output_dir: str) -> tuple:
    """
    Download PPBench dataset from HuggingFace.

    Returns paths to (full_dataset.jsonl, golden_300.jsonl).
    """
    from huggingface_hub import hf_hub_download

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading PPBench from HuggingFace...")
    full_path = hf_hub_download(
        "bluecoconut/pencil-puzzle-bench",
        "full_dataset.jsonl",
        repo_type="dataset",
        local_dir=os.path.join(output_dir, "_raw"),
    )

    golden_path = hf_hub_download(
        "bluecoconut/pencil-puzzle-bench",
        "golden_300.jsonl",
        repo_type="dataset",
        local_dir=os.path.join(output_dir, "_raw"),
    )

    return full_path, golden_path


def build_ppbench_dataset(output_dir: str = "data/ppbench",
                          seed: int = 42,
                          dry_run: bool = False):
    """
    Build the complete PPBench dataset for TRM training.

    This implements the exact pipeline from arXiv:2605.19943, Appendix A.3.
    """
    rng = np.random.default_rng(seed)

    # Step 1: Download
    full_path, golden_path = download_ppbench(output_dir)

    # Step 2: Load
    print("Loading PPBench data...")
    full_records = load_ppbench_data(full_path)
    golden_records = load_ppbench_data(golden_path)
    print(f"  Full dataset: {len(full_records)} puzzles")
    print(f"  Golden set: {len(golden_records)} puzzles")

    # Step 3: Filter to our 6 puzzle types at correct sizes
    print("Filtering to 6 puzzle types...")
    filtered = filter_puzzles(full_records)
    golden_filtered = filter_puzzles(golden_records)

    for ptype in PUZZLE_TYPES:
        count = len(filtered.get(ptype, []))
        golden_count = len(golden_filtered.get(ptype, []))
        print(f"  {ptype}: {count} puzzles ({golden_count} golden)")

    # Step 4: Split into golden / val / train
    print("Splitting into train/val/golden...")
    golden_split, remaining = split_golden(filtered, golden_records)
    val_split, train_split = split_val_train(remaining, rng)

    for ptype in PUZZLE_TYPES:
        t = len(train_split.get(ptype, []))
        v = len(val_split.get(ptype, []))
        g = len(golden_split.get(ptype, []))
        print(f"  {ptype}: train={t}, val={v}, golden={g}")

    if dry_run:
        print("\n[DRY RUN] Would build dataset with above counts. Exiting.")
        return

    # Step 5: Process puzzles and build vocabulary
    print("Processing puzzles and building vocabulary...")
    all_processed = {}
    for ptype in PUZZLE_TYPES:
        all_processed[ptype] = []
        for record in tqdm(train_split.get(ptype, []) +
                           val_split.get(ptype, []) +
                           golden_split.get(ptype, []),
                           desc=f"  {ptype}"):
            processed = process_puzzle_record(record, ptype)
            if processed:
                all_processed[ptype].append(processed)

    # Build unified vocabulary
    vocab = VocabBuilder()
    vocab.build_type_aware_vocab(all_processed)
    print(f"  Vocabulary size: {vocab.vocab_size} tokens")

    # Puzzle type → identifier mapping
    puzzle_type_to_id = {ptype: i for i, ptype in enumerate(PUZZLE_TYPES)}

    # Step 6: Build splits
    print("Building train split (with augmentation)...")
    train_data = build_split(train_split, vocab, augment=True, rng=rng,
                             puzzle_type_to_id=puzzle_type_to_id)

    print("Building val split (no augmentation)...")
    val_data = build_split(val_split, vocab, augment=False, rng=rng,
                           puzzle_type_to_id=puzzle_type_to_id)

    print("Building golden split (no augmentation)...")
    golden_data = build_split(golden_split, vocab, augment=False, rng=rng,
                              puzzle_type_to_id=puzzle_type_to_id)

    # Step 7: Save metadata
    metadata_base = {
        "seq_len": SEQ_LEN,
        "vocab_size": vocab.vocab_size,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": len(PUZZLE_TYPES),
        "sets": ["all"],
    }

    def make_metadata(data):
        num_groups = len(data["group_indices"]) - 1
        num_puzzles = len(data["puzzle_indices"]) - 1
        return {
            **metadata_base,
            "total_groups": num_groups,
            "mean_puzzle_examples": len(data["inputs"]) / max(num_groups, 1),
            "total_puzzles": num_puzzles,
        }

    # Step 8: Save all splits
    print("Saving splits...")
    save_split(train_data, make_metadata(train_data), output_dir, "train")
    save_split(val_data, make_metadata(val_data), output_dir, "test")  # TRM expects "test" dir
    save_split(golden_data, make_metadata(golden_data), output_dir, "golden")

    # Save identifiers mapping
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        json.dump(PUZZLE_TYPES, f)

    # Save vocab mapping for reference
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump({
            "vocab_size": vocab.vocab_size,
            "type_token_maps": {
                k: {str(kk): vv for kk, vv in v.items()}
                for k, v in vocab.type_token_maps.items()
            },
            "puzzle_type_to_id": puzzle_type_to_id,
        }, f, indent=2)

    print(f"\n✅ PPBench dataset built successfully!")
    print(f"   Output: {output_dir}")
    print(f"   Train: {len(train_data['inputs'])} examples")
    print(f"   Val: {len(val_data['inputs'])} examples")
    print(f"   Golden: {len(golden_data['inputs'])} examples")
    print(f"   Vocab: {vocab.vocab_size} tokens")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build PPBench dataset for TRM training (arXiv:2605.19943)"
    )
    parser.add_argument(
        "--output-dir", default="data/ppbench",
        help="Output directory for the dataset (default: data/ppbench)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only compute counts, don't build the dataset"
    )
    args = parser.parse_args()

    build_ppbench_dataset(
        output_dir=args.output_dir,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
