/// Module writing: serializes the CModule to a `.c` file and invokes
/// the system C compiler to produce an object file.
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use rustc_codegen_ssa::back::write::{CodegenContext, ModuleConfig};
use rustc_codegen_ssa::{CodegenResults, CompiledModule};
use rustc_session::config::OutputFilenames;

use crate::CCodegenBackend;
use crate::module::CModule;

/// Write a CModule to a `.c` file and compile it to an object file.
pub(crate) fn codegen(
    cgcx: &CodegenContext<CCodegenBackend>,
    module: rustc_codegen_ssa::ModuleCodegen<CModule>,
    _config: &ModuleConfig,
) -> CompiledModule {
    let c_source = module.module_llvm.to_c_source();

    // Write .c file
    let c_path = cgcx.output_filenames.temp_path_ext_for_cgu(
        "c",
        &module.name,
        cgcx.invocation_temp.as_deref(),
    );
    if let Err(e) = fs::write(&c_path, &c_source) {
        panic!("failed to write C source: {e}");
    }

    // Compile .c to .o using system C compiler
    let obj_path = cgcx.output_filenames.temp_path_ext_for_cgu(
        "o",
        &module.name,
        cgcx.invocation_temp.as_deref(),
    );

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut cmd = Command::new(&cc);
    cmd.arg("-c")
        .arg("-o")
        .arg(&obj_path)
        .arg(&c_path)
        .arg("-std=c11")
        .arg("-w") // suppress warnings for generated code
        .arg("-Werror=implicit-function-declaration") // catch missing declarations
        .arg("-fwrapv") // guarantee two's complement wrapping for signed overflow
        .arg("-fno-strict-aliasing") // generated code uses type-punned pointer access
        .arg("-funwind-tables") // emit .eh_frame so the unwinder can walk through C-compiled
        // frames (required for proc macros loaded into the compiler)
        .arg("-fno-stack-protector") // disable stack protector for generated code -- the C
        // backend may generate patterns that trip the canary
        // without actual corruption (e.g. aggregate stores)
        .arg("-ffunction-sections") // put each function in its own section so --gc-sections
        .arg("-fdata-sections") // can strip unreachable code/data at link time
        .arg("-fno-optimize-sibling-calls"); // disable tail call elimination -- Rust does not
    // guarantee TCO; clang at -O1 aggressively converts
    // recursion to loops, preventing stack overflow
    // detection (e.g. tests/ui/runtime/out-of-stack.rs)

    // -fPIC: skip for MSVC-like compilers where it's not applicable
    let cc_lower = cc.to_lowercase();
    if !cc_lower.contains("cl.exe") && !cc_lower.ends_with("\\cl") {
        cmd.arg("-fPIC");
    }

    // On aarch64 targets, disable outline-atomics to prevent infinite
    // recursion: our weak __aarch64_* implementations use __sync_*
    // builtins, which the C compiler would otherwise compile back into
    // calls to the very same outline-atomics functions.
    if cgcx.target_arch == "aarch64" {
        cmd.arg("-mno-outline-atomics");
    }

    // Add optimization level.
    // TODO: temporarily forced to -O1 for faster iteration.
    // -O0 causes SIGSEGV because generated C relies on optimizer
    // for stack temporary elimination.
    cmd.arg("-O1");

    match cmd.output() {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                panic!("C compiler failed for {}: {stderr}", c_path.display());
            }
        }
        Err(e) => {
            panic!("failed to invoke C compiler `{cc}`: {e}");
        }
    }

    // Copy C source to csources/
    emit_csource_artifact(cgcx, &c_source, &module.name);

    CompiledModule {
        name: module.name.clone(),
        kind: module.kind,
        object: Some(obj_path),
        dwarf_object: None,
        bytecode: None,
        assembly: None,
        llvm_ir: None,
        links_from_incr_cache: vec![],
    }
}

/// Resolve the output directory for csources/ and Makefile.
///
/// When cargo sets `--out-dir` to `target/<profile>/deps/`, we move up
/// one level so that artifacts land alongside the final executable in
/// `target/<profile>/`.
fn resolve_out_dir(outputs: &rustc_session::config::OutputFilenames) -> Option<PathBuf> {
    let out_ref = outputs.with_extension("");
    let mut out_dir = out_ref.parent()?.to_path_buf();
    if out_dir.file_name().map_or(false, |n| n == "deps") {
        if let Some(parent) = out_dir.parent() {
            out_dir = parent.to_path_buf();
        }
    }
    Some(out_dir)
}

/// Copy the C source to `csources/`.
///
/// If `RUSTC_CSOURCES_DIR` is set, C sources go there (shared across all
/// crates); otherwise they go to `<out_dir>/csources/`.
fn emit_csource_artifact(
    cgcx: &CodegenContext<CCodegenBackend>,
    c_source: &str,
    module_name: &str,
) {
    let csources_dir = if let Ok(dir) = std::env::var("RUSTC_CSOURCES_DIR") {
        PathBuf::from(dir)
    } else {
        let out_dir = match resolve_out_dir(&cgcx.output_filenames) {
            Some(d) => d,
            None => return,
        };
        out_dir.join("csources")
    };

    let _ = fs::create_dir_all(&csources_dir);

    let safe_name =
        module_name.replace(|c: char| !c.is_alphanumeric() && c != '_' && c != '-', "_");
    let _ = fs::write(csources_dir.join(format!("{safe_name}.c")), c_source);
}

/// Called once at link time to emit the final Makefile with complete
/// dependency information (native libs collected from all crates).
pub(crate) fn emit_final_makefile(codegen_results: &CodegenResults, outputs: &OutputFilenames) {
    let out_dir = if let Ok(dir) = std::env::var("RUSTC_CSOURCES_DIR") {
        // When RUSTC_CSOURCES_DIR is set, place the Makefile next to csources.
        match PathBuf::from(&dir).parent() {
            Some(p) if p.as_os_str().is_empty() => PathBuf::from("."),
            Some(p) => p.to_path_buf(),
            None => PathBuf::from("."),
        }
    } else {
        match resolve_out_dir(outputs) {
            Some(d) => d,
            None => return,
        }
    };

    let crate_name = outputs
        .with_extension("")
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "a.out".to_string());

    // Collect shared (dylib) native library names from all dependency
    // crates.  Static libs from build scripts are bundled in rlibs and
    // replaced by native_stubs.c for the Makefile build, so skip them.
    // Also skip libraries that are implicitly linked by cc.
    const IMPLICIT_LIBS: &[&str] = &["c", "gcc", "gcc_s", "gcc_eh", "compiler-rt"];

    let mut native_libs = BTreeSet::new();
    for libs in codegen_results.crate_info.native_libraries.values() {
        for lib in libs {
            use rustc_hir::attrs::NativeLibKind;
            if let NativeLibKind::Dylib { .. } = lib.kind {
                let name = lib.name.as_str();
                if !IMPLICIT_LIBS.contains(&name) {
                    native_libs.insert(name.to_string());
                }
            }
        }
    }
    for lib in &codegen_results.crate_info.used_libraries {
        use rustc_hir::attrs::NativeLibKind;
        if let NativeLibKind::Dylib { .. } = lib.kind {
            let name = lib.name.as_str();
            if !IMPLICIT_LIBS.contains(&name) {
                native_libs.insert(name.to_string());
            }
        }
    }
    // Always include core system libraries.
    for name in ["pthread", "dl", "m"] {
        native_libs.insert(name.to_string());
    }

    let _ = fs::write(
        out_dir.join("Makefile"),
        generate_makefile(&crate_name, &native_libs),
    );

    // Write native_stubs.c -- portable C fallbacks for build-script
    // native code (psm, blake3).  Separate TU avoids type conflicts
    // with codegen-emitted forward declarations.
    let csources_dir = if let Ok(dir) = std::env::var("RUSTC_CSOURCES_DIR") {
        PathBuf::from(dir)
    } else {
        out_dir.join("csources")
    };
    let _ = fs::create_dir_all(&csources_dir);
    let _ = fs::write(
        csources_dir.join("native_stubs.c"),
        crate::native_stubs::generate(),
    );
}

/// Generate Makefile content with `tarball` and `build` targets.
///
/// The output is portable across Linux and macOS by detecting the OS
/// at make-time via `uname -s`.
fn generate_makefile(crate_name: &str, native_libs: &BTreeSet<String>) -> String {
    let libs: String = native_libs.iter().map(|l| format!(" -l{l}")).collect();

    format!(
        "\
# Generated by rustc_codegen_c
# Makefile for building from transpiled C sources

CC ?= cc

CFLAGS := -std=c11 -fwrapv -fno-strict-aliasing -funwind-tables \\
          -fno-stack-protector -ffunction-sections -fdata-sections -fPIC -O1

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS := -Wl,-dead_strip{libs}
else
  LDFLAGS := -Wl,--gc-sections -Wl,--allow-multiple-definition{libs}
endif

SRCDIR := csources
SRCS := $(wildcard $(SRCDIR)/*.c)
OBJS := $(SRCS:.c=.o)
OUTPUT := {crate_name}
TARBALL := csources.tar.gz

.PHONY: build tarball clean

build: $(OBJS)
\t$(CC) $(OBJS) $(LDFLAGS) -o $(OUTPUT)

tarball:
\ttar czf $(TARBALL) Makefile $(SRCDIR)

$(SRCDIR)/%.o: $(SRCDIR)/%.c
\t$(CC) $(CFLAGS) -c -o $@ $<

clean:
\trm -f $(OBJS) $(OUTPUT) $(TARBALL)
"
    )
}
