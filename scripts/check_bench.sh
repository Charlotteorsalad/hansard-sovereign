#!/usr/bin/env bash
# Check whether the benchmark run has finished. Usage: bash scripts/check_bench.sh
if pgrep -af benchmark_quantization >/dev/null; then
  done=$(grep -c "^q[0-9]" scratch_bench.log 2>/dev/null || echo 0)
  echo "Still running: ${done}/48 done"
  echo "Recent progress:"
  grep "^q[0-9]" scratch_bench.log 2>/dev/null | tail -3
else
  if [ -f results/quantization_benchmark.csv ]; then
    rows=$(($(wc -l < results/quantization_benchmark.csv) - 1))
    echo "Done: CSV written (${rows} rows): results/quantization_benchmark.csv"
  else
    echo "Process exited but no CSV was produced. Last errors:"
    tail -15 scratch_bench.err 2>/dev/null
  fi
fi
