#!/usr/bin/env python3
"""new_log.py - Prompt log entry generator.

Usage:
    python tools/new_log.py --track <TRACK> --tool <TOOL>

Arguments:
    --track: Target track identifier. Valid options:
             A, B, C, D, E, F, G, H, I, J, or 00_setup
    --tool:  AI tool name (e.g. claude, antigravity, gpt-4)

Description:
    Copies TEMPLATE.md from prompts/<track>/ to a new file named:
        prompts/<track>/NNN_YYYY-MM-DD_<tool>.md
    where NNN auto-increments based on existing log files in that track.
    Pre-fills the Date / time, AI tool and model version, and Track fields.
    Prints the resulting file path to standard output.
"""

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

VALID_TRACKS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "00_setup"
]


def sanitize_tool_name(tool: str) -> str:
    """Sanitize tool name for use in filenames."""
    sanitized = re.sub(r"[^\w\-]", "_", tool.strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "tool"


def get_next_log_number(track_dir: Path) -> int:
    """Find the next auto-incrementing integer (NNN) in track directory."""
    highest = 0
    pattern = re.compile(r"^(\d{3})_.*\.md$")
    if track_dir.exists():
        for item in track_dir.iterdir():
            if item.is_file():
                match = pattern.match(item.name)
                if match:
                    highest = max(highest, int(match.group(1)))
    return highest + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a new prompt log entry from the track template.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--track",
        required=True,
        help="Track identifier (A-J or 00_setup)",
    )
    parser.add_argument(
        "--tool",
        required=True,
        help="AI tool name (e.g. claude, antigravity)",
    )

    args = parser.parse_args()

    # Track validation (case-insensitive mapping if user typed 'a' instead of 'A')
    track_input = args.track.strip()
    matched_track = None
    for vt in VALID_TRACKS:
        if vt.lower() == track_input.lower():
            matched_track = vt
            break

    if not matched_track:
        sys.stderr.write(
            f"Error: Invalid track '{args.track}'. Valid tracks are: {', '.join(VALID_TRACKS)}\n"
        )
        sys.exit(1)

    tool_raw = args.tool.strip()
    if not tool_raw:
        sys.stderr.write("Error: Tool name cannot be empty.\n")
        sys.exit(1)

    tool_slug = sanitize_tool_name(tool_raw)

    repo_root = Path(__file__).resolve().parent.parent
    track_dir = repo_root / "prompts" / matched_track

    if not track_dir.exists():
        track_dir.mkdir(parents=True, exist_ok=True)

    template_file = track_dir / "TEMPLATE.md"
    if not template_file.exists():
        fallback = repo_root / "prompts" / "TEMPLATE.md"
        if fallback.exists():
            template_file = fallback
        else:
            sys.stderr.write(
                f"Error: Template file not found at {template_file}\n"
            )
            sys.exit(1)

    template_content = template_file.read_text(encoding="utf-8")

    next_num = get_next_log_number(track_dir)
    num_str = f"{next_num:03d}"
    today_date = datetime.now().strftime("%Y-%m-%d")
    now_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = f"{num_str}_{today_date}_{tool_slug}.md"
    target_path = track_dir / filename

    # Pre-fill fields
    content = template_content
    content = re.sub(
        r"^- Date / time:.*$",
        f"- Date / time: {now_timestamp}",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^- AI tool and model version:.*$",
        f"- AI tool and model version: {tool_raw}",
        content,
        flags=re.MULTILINE,
    )
    content = re.sub(
        r"^- Track:.*$",
        f"- Track: {matched_track}",
        content,
        flags=re.MULTILINE,
    )

    target_path.write_text(content, encoding="utf-8")

    try:
        rel_path = target_path.relative_to(repo_root)
        print(rel_path)
    except ValueError:
        print(target_path)


if __name__ == "__main__":
    main()
