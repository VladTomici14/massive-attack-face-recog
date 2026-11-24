#!/usr/bin/env python3
"""
terminal_art.py

Splits the terminal into 3 columns. Each row in each column shows:

    > <random text> ----------

where the random text has a random length, and the trailing dashes
pad the line out so it fills the full width of that column.
"""

import os
import random
import shutil
import string
import time
import argparse

PREFIX = "> "
SEP = " "  # space between random text and the dash padding
GAP = "  "  # gap between columns


BOLD = "\033[1m"
RED = "\033[31m"
RESET = "\033[0m"
RED_CHANCE = 0.2  # chance any given line is printed in red


def random_text(min_len=8, max_len=30):
    length = random.randint(min_len, max_len)
    return "".join(random.choices(string.ascii_letters, k=length))


def build_row(col_width):
    """Build a single row string that exactly fills col_width characters."""
    # Leave room for the prefix and separator
    overhead = len(PREFIX) + len(SEP)
    max_text_len = max(1, col_width - overhead - 1)  # -1 to guarantee room for >=1 dash

    text = random_text(min_len=1, max_len=max(1, min(30, max_text_len)))

    # Number of dashes = remaining space after prefix + text + separator
    dash_count = col_width - overhead - len(text)
    dash_count = max(0, dash_count)

    row = f"{PREFIX}{text}{SEP}{'-' * dash_count}"

    # Safety: trim or pad to exact width in case of edge cases
    if len(row) > col_width:
        row = row[:col_width]
    else:
        row = row.ljust(col_width)

    color = RED if random.random() < RED_CHANCE else ""

    return f"{BOLD}{color}{row}{RESET}"


def generate_frame(num_columns=3, num_rows=None, visible_ratio=1.0):
    term_width, term_height = shutil.get_terminal_size(fallback=(80, 24))

    if num_rows is None:
        num_rows = term_height - 1  # leave a line for the prompt

    total_gap = len(GAP) * (num_columns - 1)
    col_width = (term_width - total_gap) // num_columns

    lines = []
    for _ in range(num_rows):
        row_parts = []
        for _ in range(num_columns):
            if random.random() < visible_ratio:
                row_parts.append(build_row(col_width))
            else:
                row_parts.append(" " * col_width)
        lines.append(GAP.join(row_parts))

    return "\n".join(lines)


def hide_cursor():
    print("\033[?25l", end="", flush=True)


def show_cursor():
    print("\033[?25h", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Continuously animated random ASCII art in 3 columns.")
    parser.add_argument("-r", "--rows", type=int, default=None,
                         help="Number of rows to generate (default: fill terminal height)")
    parser.add_argument("-c", "--columns", type=int, default=3,
                         help="Number of columns (default: 3)")
    parser.add_argument("--interval", type=float, default=0.05,
                         help="Seconds between frames (default: 0.05)")
    parser.add_argument("--phase-duration", type=float, default=5.0,
                         help="Seconds to stay in each phase (full / sparse) before switching (default: 5.0)")
    parser.add_argument("--sparse-ratio", type=float, default=0.4,
                         help="Fraction of lines shown during the sparse phase, 0-1 (default: 0.4)")
    args = parser.parse_args()

    hide_cursor()
    showing_all = True
    phase_start = time.time()
    try:
        while True:
            now = time.time()
            if now - phase_start >= args.phase_duration:
                showing_all = not showing_all
                phase_start = now

            visible_ratio = 1.0 if showing_all else args.sparse_ratio

            os.system("cls" if os.name == "nt" else "clear")
            print(generate_frame(num_columns=args.columns, num_rows=args.rows,
                                  visible_ratio=visible_ratio))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        print("\nStopped.")


if __name__ == "__main__":
    main()
