/// Allocator codegen: generates C wrapper functions for Rust's global allocator.
///
/// Also emits the setjmp/longjmp-based unwind infrastructure that all
/// codegen units depend on.
use rustc_ast::expand::allocator::{AllocatorMethod, NO_ALLOC_SHIM_IS_UNSTABLE, global_fn_name};
use rustc_middle::ty::TyCtxt;
use rustc_symbol_mangling::mangle_internal_symbol;

use crate::module::CModule;

pub(crate) fn codegen(
    tcx: TyCtxt<'_>,
    module: &mut CModule,
    _module_name: &str,
    methods: &[AllocatorMethod],
) {
    emit_aligned_alloc_helpers(module);

    for method in methods {
        let name = mangle_internal_symbol(tcx, &global_fn_name(method.name));
        let def = match method.name.as_str() {
            "alloc" => format!(
                "void *{name}(size_t size, size_t align) {{\n  \
                     return __rustc_aligned_alloc(size, align);\n\
                 }}\n"
            ),
            "dealloc" => format!(
                "void {name}(void *ptr, size_t size, size_t align) {{\n  \
                     (void)size; (void)align;\n  \
                     __rustc_aligned_free(ptr);\n\
                 }}\n"
            ),
            "realloc" => format!(
                "void *{name}(void *ptr, size_t old_size, size_t align, size_t new_size) {{\n  \
                     void *new_ptr = __rustc_aligned_alloc(new_size, align);\n  \
                     if (!new_ptr) return (void *)0;\n  \
                     memcpy(new_ptr, ptr, old_size < new_size ? old_size : new_size);\n  \
                     __rustc_aligned_free(ptr);\n  \
                     return new_ptr;\n\
                 }}\n"
            ),
            "alloc_zeroed" => format!(
                "void *{name}(size_t size, size_t align) {{\n  \
                     void *ptr = __rustc_aligned_alloc(size, align);\n  \
                     if (ptr) memset(ptr, 0, size);\n  \
                     return ptr;\n\
                 }}\n"
            ),
            "alloc_error_handler" => format!(
                "void {name}(size_t size, size_t align) {{\n  \
                     abort();\n\
                 }}\n"
            ),
            _ => format!("/* unknown allocator method: {} */\n", method.name),
        };
        module.function_defs.push(def);
    }

    // Emit __rust_no_alloc_shim_is_unstable_v2
    let shim_name = mangle_internal_symbol(tcx, NO_ALLOC_SHIM_IS_UNSTABLE);
    module
        .function_defs
        .push(format!("void {shim_name}(void) {{}}\n"));

    // Emit unwind infrastructure (setjmp/longjmp + _Unwind_RaiseException).
    // Note: __rust_try is now emitted as a weak definition in every module's
    // preamble (see module.rs to_c_source), so we don't emit it here.
    emit_unwind_infrastructure(module);

    // Architecture-specific builtins (guarded by C preprocessor in output)
    crate::builtins::emit_aarch64_outline_atomics(module);

    crate::builtins::emit_int128_division(module);
}

/// Emit portable aligned allocation helpers.
fn emit_aligned_alloc_helpers(module: &mut CModule) {
    module
        .function_defs
        .push(include_str!("c/aligned_alloc.c").to_string());
}

/// Emit the thread-local unwind chain definition and a hidden-visibility
/// override of `_Unwind_RaiseException` that initiates longjmp-based
/// unwinding instead of DWARF-based unwinding.
fn emit_unwind_infrastructure(module: &mut CModule) {
    // __rustc_setjmp, __rustc_longjmp are macros defined in the preamble.
    // __rustc_unwind_chain is extern-declared in the preamble; its single
    // definition with default visibility lives here so that all codegen
    // units (including those in libstd.so) share one TLS slot.
    module.function_defs.push(
        "__attribute__((visibility(\"default\"))) __thread struct __rustc_unwind_context *__rustc_unwind_chain;\n"
            .to_string(),
    );

    // _Unwind_RaiseException is now emitted as a weak definition in every
    // module's preamble (see preamble.c), so we don't emit it here.
}

/// Emit `__rust_try`: the exception-catching trampoline for `catch_unwind`.
///
/// Uses setjmp/longjmp: pushes a context onto the thread-local unwind
/// chain, calls `try_fn`, and if a longjmp arrives, calls `catch_fn`
/// with the exception pointer.
fn emit_rust_try(module: &mut CModule) {
    module
        .function_defs
        .push(include_str!("c/rust_try.c").to_string());
}
