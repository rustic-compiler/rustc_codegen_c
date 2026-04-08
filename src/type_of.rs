/// Layout-to-C-type conversion utilities.
use rustc_abi::{BackendRepr, Primitive, Scalar, Size};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Ty};
use rustc_target::callconv::{ArgAbi, CastTarget, FnAbi, PassMode};

use crate::context::CodegenCx;
use crate::types::{CTypeKind, TypeRef};

/// Convert a TyAndLayout to a C type.
pub(crate) fn layout_to_c_type<'tcx>(cx: &CodegenCx<'tcx>, layout: TyAndLayout<'tcx>) -> TypeRef {
    if layout.is_zst() {
        // ZST: use a zero-length array or void
        return cx.intern_type(CTypeKind::Struct {
            fields: vec![],
            packed: false,
            name: None,
        });
    }

    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            // Use intptr_t/uintptr_t when the Rust type is usize/isize.
            // We must NOT use PtrWidth for all pointer-width integers
            // because genuine u64/i64 (e.g. f64 bit patterns) must stay
            // uint64_t/int64_t -- on 32-bit targets uintptr_t would be
            // 32 bits, breaking 64-bit operations.
            match layout.ty.kind() {
                ty::Uint(ty::UintTy::Usize) => {
                    cx.intern_type(CTypeKind::PtrWidth { signed: false })
                }
                ty::Int(ty::IntTy::Isize) => cx.intern_type(CTypeKind::PtrWidth { signed: true }),
                _ => scalar_to_c_type(cx, scalar),
            }
        }
        BackendRepr::ScalarPair(a, b) => {
            let a_ty = scalar_field_to_c_type(cx, a, layout, 0);
            let b_ty = scalar_field_to_c_type(cx, b, layout, 1);
            cx.intern_type(CTypeKind::Struct {
                fields: vec![a_ty, b_ty],
                packed: false,
                name: None,
            })
        }
        BackendRepr::SimdVector { element, count } => {
            let elem_ty = scalar_to_c_type(cx, element);
            cx.intern_type(CTypeKind::Vector {
                element: elem_ty,
                len: count,
            })
        }
        _ => {
            // For aggregates, create a struct based on field layout or use a byte array
            let size = layout.size.bytes();
            if size == 0 {
                return cx.intern_type(CTypeKind::Struct {
                    fields: vec![],
                    packed: false,
                    name: None,
                });
            }

            // Wrap in a struct so the type can be used as a function return
            // type, parameter, or variable (arrays can't be used directly
            // in C function signatures or assignments).
            let byte_ty = cx.intern_type(CTypeKind::Int {
                bits: 8,
                signed: true,
            });
            let arr_ty = cx.intern_type(CTypeKind::Array {
                element: byte_ty,
                len: size,
            });
            cx.intern_type(CTypeKind::Struct {
                fields: vec![arr_ty],
                packed: false,
                name: None,
            })
        }
    }
}

/// Convert a scalar to a C type.
///
/// Signedness is preserved: unsigned Rust types (u8, u32, etc.) map to
/// unsigned C types (uint8_t, uint32_t), and signed types map to signed
/// C types. This allows C operations to have correct semantics without
/// needing explicit unsigned casts at every use site.
///
/// For non-standard integer widths (e.g. i24) that have no C equivalent,
/// we round up to the next standard width. Loads and stores are handled
/// by codegen_ssa using the layout size (not the type width), so the
/// extra bits are harmless.
pub(crate) fn scalar_to_c_type(cx: &CodegenCx<'_>, scalar: Scalar) -> TypeRef {
    // Rust `bool` is stored as Int(I8) with valid_range 0..=1.
    // Map it to C `_Bool` so that NOT uses logical `!` (giving 0/1)
    // rather than bitwise `~` (which on uint8_t gives 0xFE for ~1).
    // This must be checked before the primitive match because
    // bool's primitive is Int(I8, false) with 8 bits, not 1.
    if scalar.is_bool() {
        return cx.intern_type(CTypeKind::Bool);
    }
    match scalar.primitive() {
        Primitive::Int(int, signed) => {
            let bits = int.size().bits() as u32;
            // C only has standard integer widths. Round up non-standard
            // widths (e.g. 24 -> 32) so that the generated C compiles.
            let c_bits = match bits {
                0..=8 => 8,
                9..=16 => 16,
                17..=32 => 32,
                33..=64 => 64,
                _ => 128,
            };
            cx.intern_type(CTypeKind::Int {
                bits: c_bits,
                signed,
            })
        }
        Primitive::Float(float) => {
            let bits = float.size().bits() as u32;
            cx.intern_type(CTypeKind::Float { bits })
        }
        Primitive::Pointer(_) => cx.intern_type(CTypeKind::Ptr),
    }
}

/// Convert a scalar field of a ScalarPair to a C type, using the
/// pair's layout to detect usize/isize fields (e.g. slice length).
///
/// This is only safe to call when the pair_layout represents a
/// struct-like type (not a function argument split into two).
pub(crate) fn scalar_field_to_c_type<'tcx>(
    cx: &CodegenCx<'tcx>,
    scalar: Scalar,
    pair_layout: TyAndLayout<'tcx>,
    field_idx: usize,
) -> TypeRef {
    // Try to get the field's Rust type; if it's usize/isize, use PtrWidth.
    // field() can panic for some types (e.g. function arguments), so
    // use catch_unwind as a safety net.
    let field_ty = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pair_layout.field(cx, field_idx).ty
    }));
    if let Ok(field_ty) = field_ty {
        match field_ty.kind() {
            ty::Uint(ty::UintTy::Usize) => {
                return cx.intern_type(CTypeKind::PtrWidth { signed: false });
            }
            ty::Int(ty::IntTy::Isize) => {
                return cx.intern_type(CTypeKind::PtrWidth { signed: true });
            }
            _ => {}
        }
    }
    scalar_to_c_type(cx, scalar)
}

/// Convert a CastTarget to a C type.
///
/// On aarch64, CastTarget is typically a `Uniform` with one or more
/// register-sized units.  We represent this as a single integer type
/// whose size matches the cast size (rounded up to the register unit).
/// For prefixed CastTargets (used on other architectures), we create a
/// struct so that `store_cast`/`load_cast` can use `extract_value`/
/// `insert_value`.
pub(crate) fn cast_target_to_c_type(cx: &CodegenCx<'_>, target: &CastTarget) -> TypeRef {
    // Simple case (no prefix): use a single integer type sized to the
    // full cast (register-rounded).  This matches the old __int128
    // approach and avoids ABI differences between struct and integer
    // returns on some C compilers.
    if target.prefix.iter().all(|x| x.is_none()) {
        let size = target.size(cx).bytes();
        if size == 0 {
            return cx.intern_type(CTypeKind::Void);
        }
        let bits = (size * 8) as u32;
        // Round up to a valid C integer width
        let c_bits = match bits {
            0..=8 => 8,
            9..=16 => 16,
            17..=32 => 32,
            33..=64 => 64,
            _ => 128,
        };
        return cx.intern_type(CTypeKind::Int {
            bits: c_bits,
            signed: true,
        });
    }

    // Has prefix: build struct { prefix_regs..., rest_unit }
    let rest_unit_ty = cx.intern_type(CTypeKind::Int {
        bits: (target.rest.unit.size.bytes() * 8) as u32,
        signed: true,
    });
    let rest_count = if target.rest.total == Size::ZERO {
        0
    } else {
        target
            .rest
            .total
            .bytes()
            .div_ceil(target.rest.unit.size.bytes())
    };
    let mut fields = Vec::new();
    for prefix in target.prefix.iter() {
        if let Some(reg) = prefix {
            let bits = (reg.size.bytes() * 8) as u32;
            fields.push(cx.intern_type(CTypeKind::Int { bits, signed: true }));
        }
    }
    for _ in 0..rest_count {
        fields.push(rest_unit_ty);
    }
    cx.intern_type(CTypeKind::Struct {
        fields,
        packed: false,
        name: None,
    })
}

/// Get the C return type from a FnAbi.
///
/// For indirect return (large types), we use the actual layout type as
/// the C return type.  The C compiler handles sret natively (e.g. via
/// x8 on aarch64 or via a hidden first parameter on other platforms).
/// This ensures ABI compatibility when functions are called through
/// function pointers across compilation units or ABIs.
pub(crate) fn fn_abi_ret_type<'tcx>(
    cx: &CodegenCx<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> TypeRef {
    match fn_abi.ret.mode {
        PassMode::Ignore => cx.intern_type(CTypeKind::Void),
        PassMode::Direct(_) | PassMode::Pair(_, _) => layout_to_c_type(cx, fn_abi.ret.layout),
        PassMode::Cast { ref cast, .. } => cast_target_to_c_type(cx, cast),
        PassMode::Indirect { .. } => {
            // Use the actual layout type for ALL ABIs.  The C compiler
            // handles sret natively (e.g. via x8 on aarch64).  This
            // ensures ABI compatibility when function pointers are
            // transmuted between Rust and C ABIs (as in proc_macro
            // bridge's Closure).
            layout_to_c_type(cx, fn_abi.ret.layout)
        }
    }
}

/// Get the C type for a function argument.
pub(crate) fn fn_abi_arg_type<'tcx>(cx: &CodegenCx<'tcx>, arg: &ArgAbi<'tcx, Ty<'tcx>>) -> TypeRef {
    match arg.mode {
        PassMode::Ignore => cx.intern_type(CTypeKind::Void),
        PassMode::Direct(_) => layout_to_c_type(cx, arg.layout),
        PassMode::Pair(_, _) => {
            // Scalar pair passed as two arguments; return first element type
            layout_to_c_type(cx, arg.layout)
        }
        PassMode::Cast { ref cast, .. } => cast_target_to_c_type(cx, cast),
        PassMode::Indirect { .. } => cx.intern_type(CTypeKind::Ptr),
    }
}

/// Convert a FnAbi to a C function signature type.
pub(crate) fn fn_abi_to_c_type<'tcx>(
    cx: &CodegenCx<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
) -> TypeRef {
    let ret = fn_abi_ret_type(cx, fn_abi);

    let mut args = Vec::new();

    // No explicit sret parameter for ANY ABI.  The C compiler handles
    // sret natively via the platform convention.  This ensures ABI
    // compatibility when function pointers are transmuted between Rust
    // and C ABIs (as in proc_macro bridge's Closure).

    for arg in fn_abi.args.iter() {
        match arg.mode {
            PassMode::Ignore => continue,
            PassMode::Pair(_, _) => {
                // Scalar pair: two separate arguments.
                // Use layout_to_c_type which handles usize detection for
                // the full pair, then extract the struct field types.
                let pair_ty = layout_to_c_type(cx, arg.layout);
                let types = cx.types.borrow();
                if let CTypeKind::Struct { fields, .. } = types.get(pair_ty) {
                    for &f in fields {
                        args.push(f);
                    }
                } else {
                    drop(types);
                    args.push(layout_to_c_type(cx, arg.layout));
                }
            }
            PassMode::Indirect {
                meta_attrs: Some(_),
                ..
            } => {
                // Unsized indirect: pointer + metadata
                let ptr_ty = cx.intern_type(CTypeKind::Ptr);
                args.push(ptr_ty);
                args.push(ptr_ty);
            }
            _ => {
                args.push(fn_abi_arg_type(cx, arg));
            }
        }
    }

    cx.intern_type(CTypeKind::Function {
        ret,
        args,
        variadic: fn_abi.c_variadic,
    })
}
