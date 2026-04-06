/// Module writing: serializes the CModule to a `.c` file and invokes
/// the system C compiler to produce an object file.
use std::fs;
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
        .arg("-fno-stack-protector"); // disable stack protector for generated code -- the C
    // backend may generate patterns that trip the canary
    // without actual corruption (e.g. aggregate stores)

    // -fPIC: skip for MSVC-like compilers where it's not applicable
    let cc_lower = cc.to_lowercase();
    if !cc_lower.contains("cl.exe") && !cc_lower.ends_with("\\cl") {
        cmd.arg("-fPIC");
    }

    // No -mno-outline-atomics: we provide weak implementations of the
    // __aarch64_* outline-atomics symbols (guarded by #ifdef __aarch64__
    // in the generated C), so the C compiler can use outline atomics
    // freely on aarch64 targets.

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

    // Keep C source for debugging
    // (the file persists after compilation)

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
