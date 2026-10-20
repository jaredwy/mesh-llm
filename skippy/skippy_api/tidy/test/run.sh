#!/usr/bin/env bash
# run.sh <clang-tidy> <FileCheck> <plugin> <fixture.cpp> <stubs dir>
# A fixture names its checks on a `// CHECKS:` line, may override options on a
# `// CONFIG:` line (clang-tidy --config syntax), and pins diagnostics with
# FileCheck `// CHECK:` lines. Any warning without a CHECK line fails the test.
set -euo pipefail
tidy="$1"; filecheck="$2"; plugin="$3"; fixture="$4"; stubs="$5"
checks="$(sed -n 's|^// CHECKS: ||p' "$fixture")"
config="$(sed -n 's|^// CONFIG: ||p' "$fixture")"
"$tidy" --load="$plugin" -checks="-*,${checks}" --config="${config}" --header-filter='.*' \
    "$fixture" -- -std=c++23 -I "$stubs" 2>/dev/null \
  | "$filecheck" "$fixture" --check-prefix=CHECK --implicit-check-not='warning: ' --allow-empty
