/// Platform-specific compiler builtins that the C backend must provide.
///
/// These override compiler_builtins implementations that rely on inline
/// assembly (which codegen_c cannot handle) with pure-C equivalents
/// using C11 atomics and portable arithmetic.
use crate::module::CModule;

/// Emit aarch64 outline-atomics implementations using C11 atomics.
///
/// compiler_builtins provides these via inline assembly which our C backend
/// doesn't support. We override them with weak C implementations using
/// C11 atomic operations. The generated C is guarded by `#ifdef __aarch64__`
/// so it compiles correctly on any target architecture.
pub(crate) fn emit_aarch64_outline_atomics(module: &mut CModule) {
    module
        .function_defs
        .push("#ifdef __aarch64__\n".to_string());

    // CAS: __aarch64_cas{1,2,4,8,16}_{relax,acq,rel,acq_rel}
    // Use __sync builtins here instead of C11 atomics because:
    // 1. These functions ARE the outline-atomics implementation that the
    //    C compiler would call for C11 atomics, so using C11 atomics here
    //    would create infinite recursion.
    // 2. The 128-bit CAS16 requires libatomic with C11 atomics, but
    //    __sync_val_compare_and_swap compiles inline.
    for size in &[1u32, 2, 4, 8] {
        let ty = match size {
            1 => "uint8_t",
            2 => "uint16_t",
            4 => "uint32_t",
            8 => "uint64_t",
            _ => unreachable!(),
        };
        for order in &["relax", "acq", "rel", "acq_rel"] {
            let fn_name = format!("__aarch64_cas{size}_{order}");
            module.function_defs.push(format!(
                "#pragma weak {fn_name}\n\
                 {ty} {fn_name}({ty} expected, {ty} desired, {ty} *ptr) {{\n  \
                     return __sync_val_compare_and_swap(ptr, expected, desired);\n\
                 }}\n"
            ));
        }
    }
    // CAS16 (128-bit)
    for order in &["relax", "acq", "rel", "acq_rel"] {
        let fn_name = format!("__aarch64_cas16_{order}");
        module.function_defs.push(format!(
            "#pragma weak {fn_name}\n\
             uint128_t {fn_name}(uint128_t expected, uint128_t desired, uint128_t *ptr) {{\n  \
                 return __sync_val_compare_and_swap(ptr, expected, desired);\n\
             }}\n"
        ));
    }

    // SWP: __aarch64_swp{1,2,4,8}_{relax,acq,rel,acq_rel}
    for size in &[1u32, 2, 4, 8] {
        let ty = match size {
            1 => "uint8_t",
            2 => "uint16_t",
            4 => "uint32_t",
            8 => "uint64_t",
            _ => unreachable!(),
        };
        for order in &["relax", "acq", "rel", "acq_rel"] {
            let fn_name = format!("__aarch64_swp{size}_{order}");
            module.function_defs.push(format!(
                "#pragma weak {fn_name}\n\
                 {ty} {fn_name}({ty} val, {ty} *ptr) {{\n  \
                     {ty} old;\n  \
                     do {{ old = *ptr; }} while (!__sync_bool_compare_and_swap(ptr, old, val));\n  \
                     return old;\n\
                 }}\n"
            ));
        }
    }

    // LDADD: __aarch64_ldadd{1,2,4,8}_{relax,acq,rel,acq_rel}
    for size in &[1u32, 2, 4, 8] {
        let ty = match size {
            1 => "uint8_t",
            2 => "uint16_t",
            4 => "uint32_t",
            8 => "uint64_t",
            _ => unreachable!(),
        };
        for order in &["relax", "acq", "rel", "acq_rel"] {
            let fn_name = format!("__aarch64_ldadd{size}_{order}");
            module.function_defs.push(format!(
                "#pragma weak {fn_name}\n\
                 {ty} {fn_name}({ty} val, {ty} *ptr) {{\n  \
                     return __sync_fetch_and_add(ptr, val);\n\
                 }}\n"
            ));
        }
    }

    // LDCLR: __aarch64_ldclr{1,2,4,8}_{relax,acq,rel,acq_rel}
    for size in &[1u32, 2, 4, 8] {
        let ty = match size {
            1 => "uint8_t",
            2 => "uint16_t",
            4 => "uint32_t",
            8 => "uint64_t",
            _ => unreachable!(),
        };
        for order in &["relax", "acq", "rel", "acq_rel"] {
            let fn_name = format!("__aarch64_ldclr{size}_{order}");
            module.function_defs.push(format!(
                "#pragma weak {fn_name}\n\
                 {ty} {fn_name}({ty} val, {ty} *ptr) {{\n  \
                     return __sync_fetch_and_and(ptr, ~val);\n\
                 }}\n"
            ));
        }
    }

    // LDEOR: __aarch64_ldeor{1,2,4,8}_{relax,acq,rel,acq_rel}
    for size in &[1u32, 2, 4, 8] {
        let ty = match size {
            1 => "uint8_t",
            2 => "uint16_t",
            4 => "uint32_t",
            8 => "uint64_t",
            _ => unreachable!(),
        };
        for order in &["relax", "acq", "rel", "acq_rel"] {
            let fn_name = format!("__aarch64_ldeor{size}_{order}");
            module.function_defs.push(format!(
                "#pragma weak {fn_name}\n\
                 {ty} {fn_name}({ty} val, {ty} *ptr) {{\n  \
                     return __sync_fetch_and_xor(ptr, val);\n\
                 }}\n"
            ));
        }
    }

    // LDSET: __aarch64_ldset{1,2,4,8}_{relax,acq,rel,acq_rel}
    for size in &[1u32, 2, 4, 8] {
        let ty = match size {
            1 => "uint8_t",
            2 => "uint16_t",
            4 => "uint32_t",
            8 => "uint64_t",
            _ => unreachable!(),
        };
        for order in &["relax", "acq", "rel", "acq_rel"] {
            let fn_name = format!("__aarch64_ldset{size}_{order}");
            module.function_defs.push(format!(
                "#pragma weak {fn_name}\n\
                 {ty} {fn_name}({ty} val, {ty} *ptr) {{\n  \
                     return __sync_fetch_and_or(ptr, val);\n\
                 }}\n"
            ));
        }
    }

    module
        .function_defs
        .push("#endif /* __aarch64__ */\n".to_string());
}

/// Emit correct 128-bit integer division functions.
///
/// `compiler_builtins` provides Rust implementations of `__udivti3`,
/// `__umodti3`, etc., but the C-compiled versions produce incorrect
/// results. We provide pure-C fallback implementations that use
/// `uint128_t` shifts/comparisons (handled inline by the C
/// compiler) and a binary long-division loop.
///
/// These are emitted as `weak` symbols so that when `compiler_builtins`
/// is also linked (e.g. `-Z build-std`), its strong definitions take
/// precedence and no multiple-definition errors occur.
pub(crate) fn emit_int128_division(module: &mut CModule) {
    module
        .function_defs
        .push(include_str!("c/int128_division.c").to_string());
}
