#!/usr/bin/env bash
# Validate complete-map outputs; use the Python CLI directly for partial clips.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_ROOT="${1:-$PROJECT_ROOT/series_output}"
PYTHON="${PYTHON:-python}"

if [ ! -d "$OUTPUT_ROOT" ]; then
    echo "Output directory not found: $OUTPUT_ROOT" >&2
    exit 1
fi

count=0
failures=0
while IFS= read -r -d '' event_log; do
    count=$((count + 1))
    if ! "$PYTHON" "$SCRIPT_DIR/validate_pipeline_outputs.py" "$(dirname "$event_log")" --full-map; then
        failures=$((failures + 1))
    fi
done < <(find "$OUTPUT_ROOT" -type f -name event_log.jsonl -print0)

if [ "$count" -eq 0 ]; then
    echo "No event logs found: $OUTPUT_ROOT" >&2
    exit 1
fi
echo "Validated $count logs; $failures failed"
test "$failures" -eq 0
