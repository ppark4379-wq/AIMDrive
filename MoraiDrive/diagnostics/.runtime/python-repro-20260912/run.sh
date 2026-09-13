#!/usr/bin/env bash
set -u
cd -- "$(dirname -- "$0")" || exit 2
sha256sum -c SHA256SUMS > checksum-check.txt || { cat checksum-check.txt; exit 2; }
count=${1:-20}
[[ "$count" =~ ^[1-9][0-9]*$ ]] && (( count <= 100 )) || exit 2
out="results-$(date +%Y%m%d-%H%M%S)-$$"
mkdir "$out"
uname -a > "$out/environment.txt"
lscpu >> "$out/environment.txt"
cat /etc/os-release >> "$out/environment.txt"
fail=0
for ((i=1;i<=count;i++)); do
  timeout 35 ./lib/ld-linux-x86-64.so.2 --library-path "$PWD/lib" "$PWD/usr/bin/python3.8" -I -S -X faulthandler "$PWD/probe.py" stdlib > "$out/run-$i.log" 2>&1
  rc=$?
  if (( rc != 0 )) || ! grep -q '"status": "pass"' "$out/run-$i.log"; then
    fail=$((fail+1)); echo "run=$i FAIL exit=$rc" | tee -a "$out/summary.txt"
  else
    echo "run=$i PASS" | tee -a "$out/summary.txt"
  fi
done
echo "failures=$fail/$count logs=$PWD/$out" | tee -a "$out/summary.txt"
(( fail == 0 ))
