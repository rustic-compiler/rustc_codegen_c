/// Module writing: serializes the CModule to a `.c` file and invokes
/// the system C compiler to produce an object file.
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use rustc_codegen_ssa::CompiledModule;
use rustc_codegen_ssa::back::write::{CodegenContext, ModuleConfig};

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
        .arg("-fdata-sections"); // can strip unreachable code/data at link time

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

    // Copy C source to csources/ and emit Makefile alongside the output
    emit_makefile_artifacts(cgcx, &c_source, &module.name);

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

/// Copy the C source to `csources/` and write a Makefile to the output directory.
fn emit_makefile_artifacts(
    cgcx: &CodegenContext<CCodegenBackend>,
    c_source: &str,
    module_name: &str,
) {
    let out_dir = match resolve_out_dir(&cgcx.output_filenames) {
        Some(d) => d,
        None => return,
    };

    let crate_name = cgcx
        .output_filenames
        .with_extension("")
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "a.out".to_string());

    // Create csources/ and write the C source there
    let csources_dir = out_dir.join("csources");
    let _ = fs::create_dir_all(&csources_dir);

    let safe_name =
        module_name.replace(|c: char| !c.is_alphanumeric() && c != '_' && c != '-', "_");
    let _ = fs::write(csources_dir.join(format!("{safe_name}.c")), c_source);

    // Write Makefile (idempotent: same content regardless of which CGU writes it)
    let _ = fs::write(
        out_dir.join("Makefile"),
        generate_makefile(&crate_name, &cgcx.target_arch),
    );
}

/// Generate Makefile content with `tarball` and `build` targets.
fn generate_makefile(crate_name: &str, target_arch: &str) -> String {
    let mut s = String::new();
    s.push_str("# Generated by rustc_codegen_c\n");
    s.push_str("# Makefile for building from transpiled C sources\n\n");

    s.push_str("CC ?= cc\n\n");

    s.push_str("CFLAGS := -std=c11 -fwrapv -fno-strict-aliasing -funwind-tables \\\n");
    s.push_str("          -fno-stack-protector -ffunction-sections -fdata-sections -fPIC -O1\n");
    if target_arch == "aarch64" {
        s.push_str("CFLAGS += -mno-outline-atomics\n");
    }

    s.push_str("\nLDFLAGS := -Wl,--gc-sections -lpthread -ldl -lm\n\n");

    s.push_str("SRCDIR := csources\n");
    s.push_str("SRCS := $(wildcard $(SRCDIR)/*.c)\n");
    s.push_str("OBJS := $(SRCS:.c=.o)\n");
    s.push_str(&format!("OUTPUT := {crate_name}\n"));
    s.push_str("TARBALL := csources.tar.gz\n\n");

    s.push_str(".PHONY: tarball build clean\n\n");

    s.push_str("# Create tarball of C source files and Makefile\n");
    s.push_str("tarball:\n");
    s.push_str("\ttar czf $(TARBALL) Makefile $(SRCDIR)\n\n");

    s.push_str("# Build from C source files (or from extracted tarball)\n");
    s.push_str("build: $(OBJS)\n");
    s.push_str("\t$(CC) $(OBJS) $(LDFLAGS) -o $(OUTPUT)\n\n");

    s.push_str("$(SRCDIR)/%.o: $(SRCDIR)/%.c\n");
    s.push_str("\t$(CC) $(CFLAGS) -c -o $@ $<\n\n");

    s.push_str("clean:\n");
    s.push_str("\trm -f $(OBJS) $(OUTPUT) $(TARBALL)\n");

    s
}
