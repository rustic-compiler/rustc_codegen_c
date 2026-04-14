#!/bin/bash
#
# Wrapper around x.py for rustc_codegen_c builds.
# Disables architecture-specific SIMD extensions by default
# so that build scripts (blake3, etc.) produce portable output.
#
# To enable SIMD, set SIMD_ENABLED=1
#

script_dir="$(cd "$(dirname "$0")"; pwd)"
rust_dir="$(cd "$script_dir/../../"; pwd)"

# Disable SIMD in build scripts unless SIMD_ENABLED is set.
if [ "$SIMD_ENABLED" != "1" ]; then
    export CARGO_FEATURE_NO_NEON=1
    export CARGO_FEATURE_NO_AVX512=1
    export CARGO_FEATURE_NO_AVX2=1
    export CARGO_FEATURE_NO_SSE41=1
    export CARGO_FEATURE_NO_SSE2=1
fi

exec "$rust_dir/x" "$@"
