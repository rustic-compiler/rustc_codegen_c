<div align="center">
    <h1>
        🧲
        <br>
        rustc_codegen_c
    </h1>
    <p>
        Compile full-set Rust code into C
    </p>
</div>

The `rustc_codegen_c` crate implements an experimental codegen backend that
transpiles Rust's MIR into C source code, then compiles the C code with a
system C compiler (e.g., `gcc` or `clang`) to produce object files.

This enables Rust compilation for targets where an LLVM backend is unavailable
but a C compiler exists.
