#!/usr/bin/env bash

set -u

sleep_time=10

usage() {
    echo "Usage: $0 [-s sleep_seconds] path1 [path2 ...]"
    exit 1
}

while getopts "s:h" opt; do
    case "$opt" in
        s) sleep_time="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

shift $((OPTIND - 1))

if [ "$#" -eq 0 ]; then
    usage
fi

echo "Watching $# file(s), sleep=${sleep_time}s"
echo "Press Ctrl+C to stop."

while true; do
    for path in "$@"; do
        echo "[plot] $path"
        uv run python -m minimal_dino.plot_metrics "$path"
    done

    sleep "$sleep_time"
done