#!/bin/bash
set -euo pipefail

: "${MAX_FAIL_PERCENT:=1}"

./x test ui 2>&1 | tee test_output.txt || true

LINE=$(tail test_output.txt | grep -P '\d+ passed; \d+ failed; \d+ ignored;' | tail -1)
PASSED=$(echo "$LINE" | grep -oP '\d+(?= passed)')
FAILED=$(echo "$LINE" | grep -oP '\d+(?= failed)')
IGNORED=$(echo "$LINE" | grep -oP '\d+(?= ignored)')
TOTAL=$((PASSED + FAILED + IGNORED))
echo
echo "passed=$PASSED; failed=$FAILED; ignored=$IGNORED; total=$TOTAL"

FAIL_PERCENT=$(awk "BEGIN { printf \"%.2f\", $FAILED * 100 / $TOTAL }")
if awk "BEGIN { exit !($FAIL_PERCENT >= $MAX_FAIL_PERCENT) }"; then
  echo "::error::Failure rate ${FAIL_PERCENT}% >= ${MAX_FAIL_PERCENT}% ($FAILED/$TOTAL)"
  exit 1
fi
echo "OK: ${FAIL_PERCENT}% failures ($FAILED/$TOTAL), $IGNORED ignored (max allowed: ${MAX_FAIL_PERCENT}%)"
