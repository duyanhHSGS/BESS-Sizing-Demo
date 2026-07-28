from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

LEFT_START = 1
LEFT_END = 30

MISSING_START = 31
MISSING_END = 60

RIGHT_START = 61
RIGHT_END = 90

# Random blend:
#   0.5 = exact average
#   range below allows one source day to influence the result slightly more.
BLEND_MIN = 0.40
BLEND_MAX = 0.60

# Small point-by-point random variation.
# 0.02 means roughly 2% standard deviation.
LOAD_NOISE_STD = 0.02
PV_NOISE_STD = 0.03

DEFAULT_SEED = 42

REQUIRED_COLUMNS = {
    "day_index",
    "step",
    "P_load_kW",
    "P_pv_kW",
    "day_type",
}


# ============================================================
# Calendar category
# ============================================================

def get_day_category(day_index: int) -> str:
    """
    Calendar pattern:

        working:              days 1, 2, 3
        first_weekend_day:    day 4
        second_weekend_day:   day 5
        working:              days 6, 7, 8, 9, 10
        first_weekend_day:    day 11
        second_weekend_day:   day 12
        ...

    Weekend pairs:
        4-5, 11-12, 18-19, 25-26, ...
    """

    position = (day_index - 1) % 7

    if position == 3:
        return "first_weekend_day"

    if position == 4:
        return "second_weekend_day"

    return "working"


def output_day_type(category: str) -> str:
    """
    The CSV only stores:
        working
        weekend

    The first/second weekend distinction is only used internally.
    """

    if category == "working":
        return "working"

    return "weekend"


# ============================================================
# Validation and source-day selection
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if df.empty:
        raise ValueError("The input CSV is empty.")

    if df[["day_index", "step"]].duplicated().any():
        duplicates = df.loc[
            df[["day_index", "step"]].duplicated(keep=False),
            ["day_index", "step"],
        ]

        raise ValueError(
            "Duplicate day_index/step rows found. Example:\n"
            f"{duplicates.head()}"
        )

    for column in ("day_index", "step", "P_load_kW", "P_pv_kW"):
        if df[column].isna().any():
            raise ValueError(f"Column {column!r} contains missing values.")


def build_source_pools(
    df: pd.DataFrame,
    start_day: int,
    end_day: int,
) -> dict[str, list[int]]:
    """
    Find complete source days in a known chunk and divide them into:

        working
        first_weekend_day
        second_weekend_day
    """

    chunk = df[df["day_index"].between(start_day, end_day)]

    if chunk.empty:
        raise ValueError(
            f"No rows found in known-day chunk {start_day}-{end_day}."
        )

    expected_steps = set(chunk["step"].unique())

    pools: dict[str, list[int]] = {
        "working": [],
        "first_weekend_day": [],
        "second_weekend_day": [],
    }

    for day_index, day_df in chunk.groupby("day_index"):
        day_index = int(day_index)
        available_steps = set(day_df["step"])

        if available_steps != expected_steps:
            print(
                f"Warning: ignoring incomplete source day {day_index}. "
                f"It has {len(available_steps)} steps; expected "
                f"{len(expected_steps)}."
            )
            continue

        category = get_day_category(day_index)
        pools[category].append(day_index)

    for category, days in pools.items():
        if not days:
            raise ValueError(
                f"No complete {category!r} source days found "
                f"inside days {start_day}-{end_day}."
            )

    return pools


def read_complete_day(
    df: pd.DataFrame,
    day_index: int,
    expected_steps: np.ndarray,
) -> pd.DataFrame:
    day = (
        df[df["day_index"] == day_index]
        .sort_values("step")
        .reset_index(drop=True)
    )

    actual_steps = day["step"].to_numpy()

    if not np.array_equal(actual_steps, expected_steps):
        raise ValueError(
            f"Source day {day_index} does not contain the expected "
            "step sequence."
        )

    return day


# ============================================================
# Missing-day generation
# ============================================================

def generate_missing_days(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each missing day:

    1. Determine whether it is:
       - working
       - first weekend day
       - second weekend day

    2. Randomly choose one same-category source day from days 1-30.

    3. Randomly choose one same-category source day from days 61-90.

    4. Randomly blend their complete profiles.

    5. Add a small amount of multiplicative randomness.

    Returns:
        generated_rows
        generation_log
    """

    known_df = df[
        ~df["day_index"].between(MISSING_START, MISSING_END)
    ].copy()

    left_pools = build_source_pools(
        known_df,
        LEFT_START,
        LEFT_END,
    )

    right_pools = build_source_pools(
        known_df,
        RIGHT_START,
        RIGHT_END,
    )

    all_steps = np.sort(
        known_df[
            known_df["day_index"].between(LEFT_START, LEFT_END)
        ]["step"].unique()
    )

    if len(all_steps) == 0:
        raise ValueError("Could not determine the expected step sequence.")

    generated_frames: list[pd.DataFrame] = []
    generation_log: list[dict[str, object]] = []

    for target_day in range(MISSING_START, MISSING_END + 1):
        category = get_day_category(target_day)

        left_day_index = int(rng.choice(left_pools[category]))
        right_day_index = int(rng.choice(right_pools[category]))

        left_day = read_complete_day(
            known_df,
            left_day_index,
            all_steps,
        )

        right_day = read_complete_day(
            known_df,
            right_day_index,
            all_steps,
        )

        # Randomized average.
        #
        # Example:
        #   blend_weight = 0.47
        #
        # Result:
        #   47% left source + 53% right source
        blend_weight = float(
            rng.uniform(BLEND_MIN, BLEND_MAX)
        )

        load_left = left_day["P_load_kW"].to_numpy(dtype=float)
        load_right = right_day["P_load_kW"].to_numpy(dtype=float)

        pv_left = left_day["P_pv_kW"].to_numpy(dtype=float)
        pv_right = right_day["P_pv_kW"].to_numpy(dtype=float)

        generated_load = (
            blend_weight * load_left
            + (1.0 - blend_weight) * load_right
        )

        generated_pv = (
            blend_weight * pv_left
            + (1.0 - blend_weight) * pv_right
        )

        # Add small independent variation at each step.
        load_noise = rng.normal(
            loc=1.0,
            scale=LOAD_NOISE_STD,
            size=len(all_steps),
        )

        pv_noise = rng.normal(
            loc=1.0,
            scale=PV_NOISE_STD,
            size=len(all_steps),
        )

        generated_load *= load_noise
        generated_pv *= pv_noise

        # Electrical power cannot be negative.
        generated_load = np.maximum(generated_load, 0.0)
        generated_pv = np.maximum(generated_pv, 0.0)

        generated_day = pd.DataFrame(
            {
                "day_index": target_day,
                "step": all_steps,
                "P_load_kW": generated_load,
                "P_pv_kW": generated_pv,
                "day_type": output_day_type(category),
            }
        )

        generated_frames.append(generated_day)

        generation_log.append(
            {
                "generated_day": target_day,
                "category": category,
                "csv_day_type": output_day_type(category),
                "left_source_day": left_day_index,
                "right_source_day": right_day_index,
                "left_weight": blend_weight,
                "right_weight": 1.0 - blend_weight,
            }
        )

    generated = pd.concat(
        generated_frames,
        ignore_index=True,
    )

    log_df = pd.DataFrame(generation_log)

    return generated, log_df


# ============================================================
# Main operation
# ============================================================

def process_csv(
    input_path: Path,
    output_path: Path,
    log_path: Path,
    seed: int,
) -> None:
    df = pd.read_csv(input_path)
    validate_dataframe(df)

    rng = np.random.default_rng(seed)

    generated, generation_log = generate_missing_days(df, rng)

    # Delete any old rows for days 31-60, then insert regenerated rows.
    untouched = df[
        ~df["day_index"].between(MISSING_START, MISSING_END)
    ].copy()

    result = pd.concat(
        [untouched, generated],
        ignore_index=True,
    )

    result = (
        result.sort_values(["day_index", "step"])
        .reset_index(drop=True)
    )

    # Preserve the original column order.
    result = result[df.columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)
    generation_log.to_csv(log_path, index=False)

    print()
    print("Generation completed.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Log:    {log_path}")
    print(f"Seed:   {seed}")
    print()
    print(generation_log.to_string(index=False))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate missing days 31-60 by randomly averaging "
            "same-calendar-category days from known chunks 1-30 "
            "and 61-90."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="dense-90-youngone.csv",
        help="Input CSV path. Default: input.csv",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="filled_days.csv",
        help="Output CSV path. Default: filled_days.csv",
    )

    parser.add_argument(
        "--log",
        default="generated_day_sources.csv",
        help=(
            "CSV recording which source days generated each missing day. "
            "Default: generated_day_sources.csv"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    process_csv(
        input_path=Path(args.input),
        output_path=Path(args.output),
        log_path=Path(args.log),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()