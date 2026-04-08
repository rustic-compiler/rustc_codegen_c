/// Generates a standalone `native_stubs.c` file that provides portable C
/// fallback implementations for symbols normally compiled from assembly
/// by crate build scripts (psm, blake3).
///
/// All public symbols are `__attribute__((weak))` so real implementations
/// from rlibs take precedence during normal rustc linking.  The stubs are
/// only exercised by the standalone Makefile build.
///
/// Emitted once at link time into `csources/` -- never in the preamble.

/// Returns the complete C source for `native_stubs.c`.
pub fn generate() -> String {
    include_str!("c/native_stubs.c").to_string()
}
