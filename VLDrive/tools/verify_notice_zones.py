#!/usr/bin/env python3
"""Replay every global-path vertex twice; export boundaries, token IDs and a route map."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.notice_policy import (DEFAULT_ZONES, NoticeTracker, RouteNoticePolicy,
                                 compose_instruction, file_sha256)
from tools.rviz_notice_zones import marker_specs


def verify(policy, tokenizer=None):
    tracker = NoticeTracker(policy)
    events = []
    prompts = {}
    for lap in range(2):
        for index, p in enumerate(policy.points):
            state = tracker.update(p[0], p[1], "Proceed along this route.")
            expected = policy.at_index(index)
            # Duplicate route vertices may snap to an adjacent equal-position index.
            if state["zone_ids"] != expected["zone_ids"]:
                raise AssertionError("Live/index notice mismatch at {}".format(index))
            if state["entered"] or state["exited"]:
                events.append(dict(lap=lap + 1, index=index, **{
                    k: state[k] for k in ("entered", "exited", "zone_ids", "notice_text")}))
            text = state["llm_text"]
            if text not in prompts:
                item = dict(navigation=state["instruction"], notice=state["notice_text"], text=text)
                if tokenizer is not None:
                    ids = tokenizer(text)["input_ids"]
                    if len(ids) > 128:
                        raise ValueError("Notice prompt exceeds configured 128-token limit")
                    item.update(token_ids=ids, token_count=len(ids))
                prompts[text] = item
    for zone in policy.zones:
        for lap in (1, 2):
            entered = [e["index"] for e in events if e["lap"] == lap and zone["id"] in e["entered"]]
            exited = [e["index"] for e in events if e["lap"] == lap and zone["id"] in e["exited"]]
            if entered != [zone["start"]["route_index"]] or exited != [policy.first_index_after(zone["end"]["route_index"])]:
                raise AssertionError("Wrong entry/exit: {} {} {}".format(zone["id"], entered, exited))
        start = zone["start"]["route_index"]
        state = policy.at_index(start)
        markers = marker_specs(policy, state, policy.points[start][:2])
        text = " ".join(markers[-1]["text"].splitlines()[1:])
        if text != state["notice_text"]:
            raise AssertionError("RViz text differs from model notice")
    return dict(status="passed", zones_sha256=policy.sha256,
                global_path_sha256=policy.spec["global_path_sha256"],
                route_points=len(policy.points), laps=2,
                checked_positions=2 * len(policy.points), events=events,
                prompts=list(prompts.values()), zones=policy.zones)


def plot_route(policy, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(10, 12))
    ax.plot([p[0] for p in policy.points], [p[1] for p in policy.points], color=".75", lw=2)
    for number, zone in enumerate(policy.zones):
        start, end = zone["start"]["route_index"], zone["end"]["route_index"]
        pts = policy.points[start:end + 1]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=zone["color"], lw=4,
                label="{} [{}..{}]".format(zone["id"], start, end))
        for key, marker in (("start", "o"), ("end", "s")):
            p = zone[key]["xyz"]
            ax.scatter(p[0], p[1], marker=marker, s=35, color=zone["color"], edgecolors="black", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("MORAI world x (m)")
    ax.set_ylabel("MORAI world y (m)")
    ax.set_title("Notices snapped to existing global-path vertices\nCircle: entry / Square: last active vertex")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(alpha=.2)
    figure.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--tokenizer", type=Path, help="Local TinyLlama tokenizer directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    policy = RouteNoticePolicy(args.zones)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    report = verify(policy, tokenizer)
    if tokenizer is not None:
        report["tokenizer_files"] = {p.name: file_sha256(p) for p in args.tokenizer.iterdir() if p.is_file()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.plot:
        plot_route(policy, args.output / "notice_zones.png")
    print(json.dumps({k: report[k] for k in ("status", "route_points", "laps", "checked_positions")}))


if __name__ == "__main__":
    main()
