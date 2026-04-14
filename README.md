<div align="center">
    <h1>
        🧲
        <br>
        rustc_codegen_c
    </h1>
    <p>
        rustic compiler compiles full-set Rust code into C
    </p>
</div>

The `rustc_codegen_c` crate implements an experimental codegen backend that
transpiles Rust's MIR into C source code, then compiles the C code with a
system C compiler (e.g., `gcc` or `clang`) to produce object files.

This enables Rust compilation for targets where an LLVM backend is unavailable
but a C compiler exists.

A Rust compiler built with `rustc_codegen_c` passes 99.2% of the `rustc` UI
test suite (`./x test --stage=2 ui`), confirming that the backend preserves Rust semantics
at the MIR level with high fidelity.

Related projects:

- [rust-lang/rustc\_codegen\_c](https://github.com/rust-lang/rustc_codegen_c)
- [FractalFir/rustc\_codegen\_clr](https://github.com/FractalFir/rustc_codegen_clr)
- [thepowersgang/mrustc](https://github.com/thepowersgang/mrustc)

## Quick start

### Download the compiler

```
# Donwload and link
curl -L "https://github.com/rustic-compiler/rustc_codegen_c/releases/latest/download/rustic-toolchain-$(rustc --print=host-tuple)-gcc.tar.gz" | tar xz -C rustic-sysroot
rustup toolchain link rustic rustic-sysroot

# Run
cargo new hello && cd hello
rustup run rustic cargo run --release
```

### Build `rustc` from pre-built C files

```
curl -L "https://github.com/rustic-compiler/rustc_codegen_c/releases/latest/download/rustic-rustc-csources-$(rustc --print=host-tuple)-gcc.tar.gz" | tar xz -C rustic-rustc-csources
cd rustic-rustc-csources
make build
```

## Build your Rust program from C sources

To transpile a Rust program into C and then build it from the generated C
sources:

```
RUSTC_BOOTSTRAP=1 rustup run rustic cargo build --release -Z build-std
cd target/release
make build
```

## Known problems

Some crates (e.g., `ring`) rely on CPU-specific extended instructions (such as
Neon) and may fail to compile with the C backend. You may be able to
work around this by changing the compilation target to one without those
features.

## Build the compiler

```
# Clone repos
git clone https://github.com/rustic-compiler/rustc_codegen_c.git
git clone https://github.com/rustic-compiler/rust.git

# Make sym link
cd rust/compiler
ln -s ../../rustc_codegen_c ./rustc_codegen_c

# Setup Rust build config
cd ../
cat << EOF > bootstrap.toml
profile = "dist"
[llvm]
download-ci-llvm = true
[rust]
codegen-backends = ["c"]
EOF

# Build rustc and std
./compiler/rustc_codegen_c/y build --stage=2 compiler library

# Run
rustup toolchain link rustic build/host/stage2
cargo new hello && cd hello
rustup run rustic cargo run --release
```
