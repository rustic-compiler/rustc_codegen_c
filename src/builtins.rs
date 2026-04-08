/// Platform-specific compiler builtins that the C backend must provide.
///
/// These override compiler_builtins implementations that rely on inline
/// assembly (which codegen_c cannot handle) with pure-C equivalents
/// using GCC/Clang __sync builtins and __int128 arithmetic.
use crate::module::CModule;

/// Emit aarch64 outline-atomics implementations using __sync builtins.
///
/// compiler_builtins provides these via inline assembly which our C backend
/// doesn't support. We override them with weak C implementations using
/// __sync builtins. The generated C is guarded by `#ifdef __aarch64__` so
/// it compiles correctly on any target architecture.
pub(crate) fn emit_aarch64_outline_atomics(module: &mut CModule) {
    module
        .function_defs
        .push("#ifdef __aarch64__\n".to_string());

    // CAS: __aarch64_cas{1,2,4,8,16}_{relax,acq,rel,acq_rel}
    for size in &[1u32, 2, 4, 8] {
        let ty = match size {
            1 => "uint8_t",
            2 => "uint16_t",
            4 => "uint32_t",
            8 => "uint64_t",
            _ => unreachable!(),
        };
        for order in &["relax", "acq", "rel", "acq_rel"] {
            module.function_defs.push(format!(
                "__attribute__((weak))\n\
                 {ty} __aarch64_cas{size}_{order}({ty} expected, {ty} desired, {ty} *ptr) {{\n  \
                     return __sync_val_compare_and_swap(ptr, expected, desired);\n\
                 }}\n"
            ));
        }
    }
    // CAS16 (128-bit)
    for order in &["relax", "acq", "rel", "acq_rel"] {
        module.function_defs.push(format!(
            "__attribute__((weak))\n\
             uint128_t __aarch64_cas16_{order}(uint128_t expected, uint128_t desired, uint128_t *ptr) {{\n  \
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
            module.function_defs.push(format!(
                "__attribute__((weak))\n\
                 {ty} __aarch64_swp{size}_{order}({ty} val, {ty} *ptr) {{\n  \
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
            module.function_defs.push(format!(
                "__attribute__((weak))\n\
                 {ty} __aarch64_ldadd{size}_{order}({ty} val, {ty} *ptr) {{\n  \
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
            module.function_defs.push(format!(
                "__attribute__((weak))\n\
                 {ty} __aarch64_ldclr{size}_{order}({ty} val, {ty} *ptr) {{\n  \
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
            module.function_defs.push(format!(
                "__attribute__((weak))\n\
                 {ty} __aarch64_ldeor{size}_{order}({ty} val, {ty} *ptr) {{\n  \
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
            module.function_defs.push(format!(
                "__attribute__((weak))\n\
                 {ty} __aarch64_ldset{size}_{order}({ty} val, {ty} *ptr) {{\n  \
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
/// `unsigned __int128` shifts/comparisons (handled inline by the C
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
