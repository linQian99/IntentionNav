"""Run a legacy experiment with its historical source paths in a private mount view.

The host repository stays clean. Original code/manifests are not rewritten.
Requires the installed bubblewrap and working unprivileged mount namespaces.
This is path compatibility, not a security sandbox. Results use the real filesystem.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "experiments/sources.json"
MARKER = "INAV_LEGACY_EVAL_VIEW"


def mappings(catalog: dict) -> list[tuple[Path, Path]]:
    entries = catalog["records"] + catalog.get("artifacts", [])
    return [(ROOT / row.get("source_path", row["original_path"]), ROOT / row["original_path"])
            for row in entries]


def old_argument(value: str, routes: list[tuple[Path, Path]]) -> str:
    # Translate path arguments, including --source=/path. Do not rewrite arbitrary text.
    key, separator, candidate = value.partition("=") if value.startswith("--") else ("", "", value)
    if candidate.startswith("/") or candidate.startswith(("experiments/", "./experiments/")):
        path = Path(os.path.abspath(candidate))
        for actual, legacy in routes:
            try:
                suffix = path.relative_to(actual)
            except ValueError:
                continue
            return key + separator + str(legacy / suffix)
    return value


def build_command(command: list[str], catalog: dict) -> tuple[list[str], dict[str, str]]:
    routes = mappings(catalog)
    routes.sort(key=lambda pair: len(pair[0].parts), reverse=True)
    record_routes = []
    if (ROOT / "experiments/records.json").is_file():
        from catalog_experiment_records import load
        record_routes = [(ROOT / row["path"], ROOT / row["original_path"])
                         for row in load()["records"] if row["action"] != "snapshot"]
    env = dict(os.environ)
    translated = [old_argument(arg, routes) for arg in command]
    for key, value in list(env.items()):
        if key.endswith("PATH"):
            env[key] = os.pathsep.join(old_argument(p, routes) for p in value.split(os.pathsep))
        elif value.startswith("/"):
            env[key] = old_argument(value, routes)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if os.environ.get(MARKER) == str(ROOT):
        return translated, env
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("Legacy path execution requires bubblewrap (bwrap); no source paths were recreated on the host")
    # Mount over private directories, never create mountpoints in the host repo/eval.
    # Keep normal mounts, devices, networking and process IDs; only project paths differ.
    args = [bwrap, "--dev-bind", "/", "/", "--tmpfs", str(ROOT),
            "--tmpfs", str(ROOT / "eval")]
    if record_routes:
        args += ["--tmpfs", str(ROOT / "refine-logs")]
    mount_routes = routes + record_routes
    old_roots = {legacy.relative_to(ROOT).parts[0] for _, legacy in mount_routes}
    for child in sorted(ROOT.iterdir()):
        if child.name == "eval" or child.name in old_roots:
            continue
        if child.is_symlink():
            args += ["--symlink", os.readlink(child), str(child)]
        else:
            args += ["--bind", str(child), str(child)]
    old_agents = {legacy.name for _, legacy in routes if legacy.parent == ROOT / "eval"}
    for child in sorted((ROOT / "eval").iterdir()):
        if child.name in old_agents:
            continue
        if child.is_symlink():
            args += ["--symlink", os.readlink(child), str(child)]
        else:
            args += ["--bind", str(child), str(child)]
    if record_routes:
        # Bind existing report/archive files, then expose missing historical names
        # only inside this private directory. Never create host compatibility links.
        for child in sorted((ROOT / "refine-logs").iterdir()):
            if child.is_symlink():
                args += ["--symlink", os.readlink(child), str(child)]
            else:
                args += ["--bind", str(child), str(child)]
    for actual, legacy in mount_routes:
        if not actual.exists():
            raise FileNotFoundError(actual)
        if actual != legacy:
            args += ["--ro-bind", str(actual), str(actual)]
        args += ["--ro-bind", str(actual), str(legacy)]
    # New top-level source creation must not disappear into the temporary view.
    # Existing results/work_dirs etc remain writable and persist on the host.
    if record_routes:
        args += ["--remount-ro", str(ROOT / "refine-logs")]
    args += ["--remount-ro", str(ROOT / "eval"), "--remount-ro", str(ROOT),
             "--chdir", old_argument(os.getcwd(), routes), "--", *translated]
    env[MARKER] = str(ROOT)
    return args, env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide -- COMMAND [ARGS...]")
    catalog = json.loads(CATALOG.read_text())
    argv, environment = build_command(command, catalog)
    os.execvpe(argv[0], argv, environment)


if __name__ == "__main__":
    main()
