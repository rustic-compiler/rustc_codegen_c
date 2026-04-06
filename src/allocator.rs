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

    // Emit unwind infrastructure (setjmp/longjmp + _Unwind_RaiseException + __rust_try)
    emit_unwind_infrastructure(module);
    emit_rust_try(module);

    // Architecture-specific builtins (guarded by C preprocessor in output)
    crate::builtins::emit_aarch64_outline_atomics(module);

    crate::builtins::emit_int128_division(module);
}

/// Emit portable aligned allocation helpers.
fn emit_aligned_alloc_helpers(module: &mut CModule) {
    // Forward-declare libc functions used below. We avoid including
    // <stdlib.h> because its full declarations can conflict with
    // signatures in the generated code.
    module.function_defs.push(
        "#if defined(_WIN32)\n\
         void *_aligned_malloc(size_t, size_t);\n\
         void _aligned_free(void *);\n\
         #else\n\
         void free(void *);\n\
         int posix_memalign(void **, size_t, size_t);\n\
         #endif\n\
         static void *__rustc_aligned_alloc(size_t size, size_t align) {\n\
         #if defined(_WIN32)\n  \
             return _aligned_malloc(size, align);\n\
         #else\n  \
             if (align < sizeof(void *)) align = sizeof(void *);\n  \
             void *ptr;\n  \
             if (posix_memalign(&ptr, align, size) != 0) return (void *)0;\n  \
             return ptr;\n\
         #endif\n\
         }\n\
         static void __rustc_aligned_free(void *ptr) {\n\
         #if defined(_WIN32)\n  \
             _aligned_free(ptr);\n\
         #else\n  \
             free(ptr);\n\
         #endif\n\
         }\n"
        .to_string(),
    );
}

/// Emit the thread-local unwind chain definition and a hidden-visibility
/// override of `_Unwind_RaiseException` that initiates longjmp-based
/// unwinding instead of DWARF-based unwinding.
fn emit_unwind_infrastructure(module: &mut CModule) {
    // __rustc_setjmp, __rustc_longjmp, and __rustc_unwind_chain are
    // emitted as weak definitions in every module's preamble (see
    // module.rs to_c_source).

    // Override _Unwind_RaiseException: longjmp to the innermost
    // invoke/try handler instead of DWARF-based stack unwinding.
    module.function_defs.push(
        r#"__attribute__((visibility("hidden")))
int _Unwind_RaiseException(void *exception_object) {
  if (__rustc_unwind_chain) {
    __rustc_unwind_chain->exception_ptr = exception_object;
    __rustc_longjmp(__rustc_unwind_chain->buf, 1);
  }
  abort();
  return 3;
}
"#
        .to_string(),
    );
}

/// Emit `__rust_try`: the exception-catching trampoline for `catch_unwind`.
///
/// Uses setjmp/longjmp: pushes a context onto the thread-local unwind
/// chain, calls `try_fn`, and if a longjmp arrives, calls `catch_fn`
/// with the exception pointer.
fn emit_rust_try(module: &mut CModule) {
    module.function_defs.push(
        r#"int __rust_try(void (*try_fn)(void *), void *data, void (*catch_fn)(void *, void *)) {
  struct __rustc_unwind_context __ctx;
  __ctx.prev = __rustc_unwind_chain;
  __ctx.exception_ptr = (void *)0;
  __rustc_unwind_chain = &__ctx;
  if (__rustc_setjmp(__ctx.buf) == 0) {
    try_fn(data);
    __rustc_unwind_chain = __ctx.prev;
    return 0;
  } else {
    void *__exn = __ctx.exception_ptr;
    __rustc_unwind_chain = __ctx.prev;
    catch_fn(data, __exn);
    return 1;
  }
}
"#
        .to_string(),
    );
}
