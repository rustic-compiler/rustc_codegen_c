/// Intrinsic codegen: maps Rust intrinsics to C library functions
/// and compiler builtins.
use rustc_abi::Size;
use rustc_codegen_ssa::mir::operand::OperandRef;
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::*;
use rustc_middle::mir;
use rustc_middle::ty::{self, Instance};
use rustc_span::{Span, sym};

use crate::builder::Builder;
use crate::c_ast::{CBinOp, CExpr, CStmt};
use crate::context::{CFunclet, DebugLoc, DebugVar};
use crate::module::BasicBlockId;
use crate::types::CTypeKind;
use crate::values::{CValueKind, ValueRef};

// --- IntrinsicCallBuilderMethods ---

impl<'a, 'tcx> IntrinsicCallBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OperandRef<'tcx, ValueRef>],
        result_dest: PlaceRef<'tcx, ValueRef>,
        _span: Span,
    ) -> Result<(), ty::Instance<'tcx>> {
        let name = self.cx.tcx.item_name(instance.def_id());

        match name {
            sym::black_box => {
                args[0].val.store(self, result_dest);
                Ok(())
            }
            sym::volatile_load | sym::unaligned_volatile_load => {
                let tp_ty = instance.args.type_at(0);
                let ptr = args[0].immediate();
                let load_ty = self.backend_type(self.layout_of(tp_ty));
                let val = self.volatile_load(load_ty, ptr);
                self.store(val, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::volatile_store => {
                let dst = args[0].immediate();
                let val = args[1].immediate();
                let p = self.cx.render_value(dst);
                let v = self.cx.render_value(val);
                let ty = self.cx.values.borrow().get_type(val);
                let t = self.cx.render_type(ty);
                self.emit(CStmt::assign(
                    CExpr::deref(format!("volatile {t} *"), CExpr::var(&p)),
                    CExpr::var(v),
                ));
                Ok(())
            }
            sym::prefetch_read_data
            | sym::prefetch_write_data
            | sym::prefetch_read_instruction
            | sym::prefetch_write_instruction => Ok(()),
            sym::va_copy => {
                let dest = args[0].immediate();
                let src = args[1].immediate();
                let d = self.cx.render_value(dest);
                let s = self.cx.render_value(src);
                // va_copy is a statement macro in C
                self.emit(CStmt::expr(CExpr::call(
                    CExpr::var("va_copy"),
                    vec![
                        CExpr::deref("va_list *", CExpr::var(&d)),
                        CExpr::deref("va_list *", CExpr::var(&s)),
                    ],
                )));
                Ok(())
            }
            sym::va_arg => {
                let tp_ty = instance.args.type_at(0);
                let result_layout = self.layout_of(tp_ty);
                let result_ty = self.backend_type(result_layout);
                let result_ty_str = self.cx.render_type(result_ty);
                let list = args[0].immediate();
                let l = self.cx.render_value(list);
                // va_arg is a macro: va_arg(*(va_list*)ptr, type)
                let expr_str = format!(
                    "va_arg(*(va_list *){l}, {result_ty_str})"
                );
                let val = self.new_temp_with_stmt(
                    result_ty,
                    CExpr::raw(expr_str),
                );
                self.store(val, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::catch_unwind => {
                let try_fn = args[0].immediate();
                let data = args[1].immediate();
                let catch_fn = args[2].immediate();
                let f = self.cx.render_value(try_fn);
                let d = self.cx.render_value(data);
                let c = self.cx.render_value(catch_fn);
                let i32_ty = self.cx.type_i32();
                let result = self.new_temp_with_stmt(
                    i32_ty,
                    CExpr::call(
                        CExpr::var("__rust_try"),
                        vec![
                            CExpr::cast("void (*)(void *)", CExpr::var(&f)),
                            CExpr::var(&d),
                            CExpr::cast("void (*)(void *, void *)", CExpr::var(&c)),
                        ],
                    ),
                );
                self.store(result, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::is_val_statically_known => {
                let val = self.cx.const_bool(false);
                self.store(val, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::select_unpredictable => {
                let cond = args[0].immediate();
                let a = args[1].immediate();
                let b = args[2].immediate();
                let result = self.select(cond, a, b);
                self.store(result, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::compare_bytes => {
                let lhs = self.cx.render_value(args[0].immediate());
                let rhs = self.cx.render_value(args[1].immediate());
                let len = self.cx.render_value(args[2].immediate());
                let ret_ty = self.cx.type_i32();
                let result = self.new_temp_with_stmt(
                    ret_ty,
                    CExpr::call(
                        CExpr::var("memcmp"),
                        vec![CExpr::var(&lhs), CExpr::var(&rhs), CExpr::var(&len)],
                    ),
                );
                self.store(result, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::ctlz | sym::ctlz_nonzero => codegen_ctlz(self, args, result_dest),
            sym::cttz | sym::cttz_nonzero => codegen_cttz(self, args, result_dest),
            sym::ctpop => codegen_ctpop(self, args, result_dest),
            sym::bswap => codegen_bswap(self, args, result_dest),
            sym::bitreverse => codegen_bitreverse(self, args, result_dest),
            sym::saturating_add | sym::saturating_sub => {
                codegen_saturating(self, name, args, result_dest)
            }
            // Unary f32 math intrinsics
            sym::fabsf32
            | sym::sqrtf32
            | sym::sinf32
            | sym::cosf32
            | sym::expf32
            | sym::exp2f32
            | sym::logf32
            | sym::log10f32
            | sym::log2f32
            | sym::floorf32
            | sym::ceilf32
            | sym::truncf32
            | sym::roundf32
            | sym::round_ties_even_f32 => {
                codegen_unary_math(self, name, args, result_dest, MathWidth::F32)
            }
            // Unary f64 math intrinsics
            sym::fabsf64
            | sym::sqrtf64
            | sym::sinf64
            | sym::cosf64
            | sym::expf64
            | sym::exp2f64
            | sym::logf64
            | sym::log10f64
            | sym::log2f64
            | sym::floorf64
            | sym::ceilf64
            | sym::truncf64
            | sym::roundf64
            | sym::round_ties_even_f64 => {
                codegen_unary_math(self, name, args, result_dest, MathWidth::F64)
            }
            // Binary f32 math intrinsics
            sym::copysignf32 | sym::powf32 | sym::minnumf32 | sym::maxnumf32 => {
                codegen_binary_math(self, name, args, result_dest, MathWidth::F32)
            }
            // Binary f64 math intrinsics
            sym::copysignf64 | sym::powf64 | sym::minnumf64 | sym::maxnumf64 => {
                codegen_binary_math(self, name, args, result_dest, MathWidth::F64)
            }
            // powi (float ^ int)
            sym::powif32 => {
                let a = self.cx.render_value(args[0].immediate());
                let b = self.cx.render_value(args[1].immediate());
                let ty = self.cx.values.borrow().get_type(args[0].immediate());
                let result = self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("powf"),
                        vec![CExpr::var(&a), CExpr::cast("float", CExpr::var(&b))],
                    ),
                );
                self.store(result, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::powif64 => {
                let a = self.cx.render_value(args[0].immediate());
                let b = self.cx.render_value(args[1].immediate());
                let ty = self.cx.values.borrow().get_type(args[0].immediate());
                let result = self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("pow"),
                        vec![CExpr::var(&a), CExpr::cast("double", CExpr::var(&b))],
                    ),
                );
                self.store(result, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            // fma / fmuladd (ternary: a * b + c)
            sym::fmaf32 | sym::fmuladdf32 => {
                let a = self.cx.render_value(args[0].immediate());
                let b = self.cx.render_value(args[1].immediate());
                let c = self.cx.render_value(args[2].immediate());
                let ty = self.cx.values.borrow().get_type(args[0].immediate());
                let result = self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("fmaf"),
                        vec![CExpr::var(&a), CExpr::var(&b), CExpr::var(&c)],
                    ),
                );
                self.store(result, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::fmaf64 | sym::fmuladdf64 => {
                let a = self.cx.render_value(args[0].immediate());
                let b = self.cx.render_value(args[1].immediate());
                let c = self.cx.render_value(args[2].immediate());
                let ty = self.cx.values.borrow().get_type(args[0].immediate());
                let result = self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("fma"),
                        vec![CExpr::var(&a), CExpr::var(&b), CExpr::var(&c)],
                    ),
                );
                self.store(result, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::raw_eq => {
                let lhs = self.cx.render_value(args[0].immediate());
                let rhs = self.cx.render_value(args[1].immediate());
                let tp_ty = instance.args.type_at(0);
                let layout = self.layout_of(tp_ty);
                let size = layout.size.bytes();
                let ret_ty = self.cx.type_i8();
                let result = self.new_temp_with_stmt(
                    ret_ty,
                    CExpr::binop(
                        CExpr::call(
                            CExpr::var("memcmp"),
                            vec![
                                CExpr::var(&lhs),
                                CExpr::var(&rhs),
                                CExpr::lit(&size.to_string()),
                            ],
                        ),
                        CBinOp::Eq,
                        CExpr::lit("0"),
                    ),
                );
                self.store(result, result_dest.val.llval, result_dest.val.align);
                Ok(())
            }
            sym::rotate_left | sym::rotate_right => codegen_rotate(self, name, args, result_dest),
            sym::unchecked_funnel_shl | sym::unchecked_funnel_shr => {
                codegen_funnel_shift(self, name, args, result_dest)
            }
            // SIMD intrinsics
            sym::simd_extract => codegen_simd_extract(self, args, result_dest),
            sym::simd_eq
            | sym::simd_ne
            | sym::simd_lt
            | sym::simd_le
            | sym::simd_gt
            | sym::simd_ge => codegen_simd_cmp(self, name, args, result_dest),
            sym::simd_and | sym::simd_or | sym::simd_xor => {
                codegen_simd_bitwise(self, name, args, result_dest)
            }
            sym::simd_shl | sym::simd_shr => codegen_simd_shift(self, name, args, result_dest),
            sym::simd_insert => codegen_simd_insert(self, args, result_dest),
            sym::simd_cast | sym::simd_as => codegen_simd_cast(self, args, result_dest),
            sym::simd_shuffle | sym::simd_shuffle_const_generic => {
                codegen_simd_shuffle(self, args, result_dest)
            }
            _ if name.as_str().starts_with("simd_") => {
                // Fallback for unimplemented SIMD intrinsics
                let size = result_dest.layout.size;
                let ptr = result_dest.val.llval;
                let p = self.cx.render_value(ptr);
                self.emit(CStmt::expr(CExpr::call(
                    CExpr::var("memset"),
                    vec![
                        CExpr::var(&p),
                        CExpr::lit("0"),
                        CExpr::lit(&size.bytes().to_string()),
                    ],
                )));
                Ok(())
            }
            sym::typed_swap_nonoverlapping => codegen_typed_swap(self, instance, args),
            sym::carrying_mul_add => codegen_carrying_mul_add(self, instance, args, result_dest),
            _ => Err(instance),
        }
    }

    fn codegen_llvm_intrinsic_call(
        &mut self,
        _instance: ty::Instance<'tcx>,
        _args: &[OperandRef<'tcx, ValueRef>],
        _is_cleanup: bool,
    ) -> ValueRef {
        self.cx.const_null(self.cx.type_ptr())
    }

    fn abort(&mut self) {
        self.emit(CStmt::expr(CExpr::call(CExpr::var("abort"), vec![])));
    }

    fn assume(&mut self, _val: ValueRef) {}

    fn expect(&mut self, cond: ValueRef, expected: bool) -> ValueRef {
        let c = self.cx.render_value(cond);
        let bool_ty = self.cx.intern_type(CTypeKind::Bool);
        self.new_temp_with_stmt(
            bool_ty,
            CExpr::call(
                CExpr::var("__builtin_expect"),
                vec![CExpr::var(&c), CExpr::lit(if expected { "1" } else { "0" })],
            ),
        )
    }

    fn type_checked_load(
        &mut self,
        llvtable: ValueRef,
        vtable_byte_offset: u64,
        _typeid: ValueRef,
    ) -> ValueRef {
        let ptr_ty = self.cx.type_ptr();
        let vtable = self.cx.render_value(llvtable);
        self.new_temp_with_stmt(
            ptr_ty,
            CExpr::deref(
                "void **",
                CExpr::paren(CExpr::binop(
                    CExpr::cast("char *", CExpr::var(&vtable)),
                    CBinOp::Add,
                    CExpr::lit(format!("{vtable_byte_offset}")),
                )),
            ),
        )
    }

    fn va_start(&mut self, val: ValueRef) -> ValueRef {
        let v = self.cx.render_value(val);
        // va_start takes a va_list and the last named parameter.
        // val is a pointer to a va_list, so dereference it.
        // Use 0 as dummy last-param (GCC/Clang accept this).
        self.emit(CStmt::raw(format!(
            "va_start(*(va_list *){v}, 0);"
        )));
        val
    }

    fn va_end(&mut self, val: ValueRef) -> ValueRef {
        let v = self.cx.render_value(val);
        // val is a pointer to a va_list, so dereference it.
        self.emit(CStmt::raw(format!(
            "va_end(*(va_list *){v});"
        )));
        val
    }
}

// --- CoverageInfoBuilderMethods (stub) ---

impl<'a, 'tcx> CoverageInfoBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn add_coverage(&mut self, _instance: Instance<'tcx>, _kind: &mir::coverage::CoverageKind) {}
}

// --- DebugInfoBuilderMethods (stub) ---

impl<'a, 'tcx> DebugInfoBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn dbg_var_addr(
        &mut self,
        _dbg_var: DebugVar,
        _dbg_loc: DebugLoc,
        _variable_alloca: ValueRef,
        _direct_offset: Size,
        _indirect_offsets: &[Size],
        _fragment: &Option<std::ops::Range<Size>>,
    ) {
    }

    fn dbg_var_value(
        &mut self,
        _dbg_var: DebugVar,
        _dbg_loc: DebugLoc,
        _value: ValueRef,
        _direct_offset: Size,
        _indirect_offsets: &[Size],
        _fragment: &Option<std::ops::Range<Size>>,
    ) {
    }

    fn set_dbg_loc(&mut self, _dbg_loc: DebugLoc) {}
    fn clear_dbg_loc(&mut self) {}
    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {}
    fn set_var_name(&mut self, _value: ValueRef, _name: &str) {}
}

// --- AsmBuilderMethods (stub) ---

impl<'a, 'tcx> AsmBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        _template: &[rustc_ast::InlineAsmTemplatePiece],
        _operands: &[InlineAsmOperandRef<'tcx, Self>],
        _options: rustc_ast::InlineAsmOptions,
        _line_spans: &[Span],
        _instance: Instance<'_>,
        _dest: Option<BasicBlockId>,
        _catch_funclet: Option<(BasicBlockId, Option<&CFunclet>)>,
    ) {
        self.emit(CStmt::raw("/* inline asm not supported */"));
        if let Some(dest) = _dest {
            let label = self.block_label(dest);
            self.emit(CStmt::goto(label));
        }
    }
}

// =====================================================================
// Intrinsic helper functions
// =====================================================================

enum MathWidth {
    F32,
    F64,
}

fn codegen_unary_math<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    name: rustc_span::Symbol,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
    width: MathWidth,
) -> Result<(), ty::Instance<'tcx>> {
    let v = bx.cx.render_value(args[0].immediate());
    let ty = bx.cx.values.borrow().get_type(args[0].immediate());
    let cfn = match (name, &width) {
        (sym::fabsf32, _) => "fabsf",
        (sym::fabsf64, _) => "fabs",
        (sym::sqrtf32, _) => "sqrtf",
        (sym::sqrtf64, _) => "sqrt",
        (sym::sinf32, _) => "sinf",
        (sym::sinf64, _) => "sin",
        (sym::cosf32, _) => "cosf",
        (sym::cosf64, _) => "cos",
        (sym::expf32, _) => "expf",
        (sym::expf64, _) => "exp",
        (sym::exp2f32, _) => "exp2f",
        (sym::exp2f64, _) => "exp2",
        (sym::logf32, _) => "logf",
        (sym::logf64, _) => "log",
        (sym::log10f32, _) => "log10f",
        (sym::log10f64, _) => "log10",
        (sym::log2f32, _) => "log2f",
        (sym::log2f64, _) => "log2",
        (sym::floorf32, _) => "floorf",
        (sym::floorf64, _) => "floor",
        (sym::ceilf32, _) => "ceilf",
        (sym::ceilf64, _) => "ceil",
        (sym::truncf32, _) => "truncf",
        (sym::truncf64, _) => "trunc",
        (sym::roundf32, _) => "roundf",
        (sym::roundf64, _) => "round",
        (sym::round_ties_even_f32, _) => "rintf",
        (sym::round_ties_even_f64, _) => "rint",
        _ => unreachable!(),
    };
    let result = bx.new_temp_with_stmt(ty, CExpr::call(CExpr::var(cfn), vec![CExpr::var(&v)]));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_binary_math<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    name: rustc_span::Symbol,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
    width: MathWidth,
) -> Result<(), ty::Instance<'tcx>> {
    let a = bx.cx.render_value(args[0].immediate());
    let b = bx.cx.render_value(args[1].immediate());
    let ty = bx.cx.values.borrow().get_type(args[0].immediate());
    let cfn = match (name, &width) {
        (sym::copysignf32, _) => "copysignf",
        (sym::copysignf64, _) => "copysign",
        (sym::powf32, _) => "powf",
        (sym::powf64, _) => "pow",
        (sym::minnumf32, _) => "fminf",
        (sym::minnumf64, _) => "fmin",
        (sym::maxnumf32, _) => "fmaxf",
        (sym::maxnumf64, _) => "fmax",
        _ => unreachable!(),
    };
    let result = bx.new_temp_with_stmt(
        ty,
        CExpr::call(CExpr::var(cfn), vec![CExpr::var(&a), CExpr::var(&b)]),
    );
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_ctlz<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let val = args[0].immediate();
    let v = bx.cx.render_value(val);
    let ty = bx.cx.values.borrow().get_type(val);
    let bits = bx.cx.int_width(ty);
    let ret_ty = bx.cx.type_i32();
    let expr = if bits <= 32 {
        format!("({v} == 0 ? {bits} : __builtin_clz((unsigned int){v}) - (32 - {bits}))")
    } else if bits <= 64 {
        format!("({v} == 0 ? {bits} : __builtin_clzll((unsigned long long){v}) - (64 - {bits}))")
    } else {
        format!(
            "((unsigned long long)((unsigned __int128)({v}) >> 64) != 0 \
                 ? __builtin_clzll((unsigned long long)((unsigned __int128)({v}) >> 64)) \
                 : 64 + ((unsigned long long)({v}) == 0 ? 64 : __builtin_clzll((unsigned long long)({v}))))"
        )
    };
    let result = bx.new_temp_with_stmt(ret_ty, CExpr::raw(expr));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_cttz<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let val = args[0].immediate();
    let v = bx.cx.render_value(val);
    let ty = bx.cx.values.borrow().get_type(val);
    let bits = bx.cx.int_width(ty);
    let ret_ty = bx.cx.type_i32();
    let expr = if bits <= 32 {
        format!("({v} == 0 ? {bits} : __builtin_ctz((unsigned int){v}))")
    } else if bits <= 64 {
        format!("({v} == 0 ? {bits} : __builtin_ctzll((unsigned long long){v}))")
    } else {
        format!(
            "((unsigned long long)({v}) != 0 \
                 ? __builtin_ctzll((unsigned long long)({v})) \
                 : 64 + ((unsigned long long)((unsigned __int128)({v}) >> 64) == 0 ? 64 : __builtin_ctzll((unsigned long long)((unsigned __int128)({v}) >> 64))))"
        )
    };
    let result = bx.new_temp_with_stmt(ret_ty, CExpr::raw(expr));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_ctpop<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let val = args[0].immediate();
    let v = bx.cx.render_value(val);
    let ty = bx.cx.values.borrow().get_type(val);
    let bits = bx.cx.int_width(ty);
    let ret_ty = bx.cx.type_i32();
    let expr = if bits <= 32 {
        format!("__builtin_popcount((unsigned int){v})")
    } else if bits <= 64 {
        format!("__builtin_popcountll((unsigned long long){v})")
    } else {
        format!(
            "(__builtin_popcountll((unsigned long long)({v})) + __builtin_popcountll((unsigned long long)((unsigned __int128)({v}) >> 64)))"
        )
    };
    let result = bx.new_temp_with_stmt(ret_ty, CExpr::raw(expr));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_bswap<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let val = args[0].immediate();
    let v = bx.cx.render_value(val);
    let ty = bx.cx.values.borrow().get_type(val);
    let bits = bx.cx.int_width(ty);
    let expr = match bits {
        8 => v.clone(),
        16 => format!("__builtin_bswap16({v})"),
        32 => format!("__builtin_bswap32({v})"),
        64 => format!("__builtin_bswap64({v})"),
        128 => format!(
            "((unsigned __int128)__builtin_bswap64((uint64_t)({v})) << 64 | \
             (unsigned __int128)__builtin_bswap64((uint64_t)((unsigned __int128)({v}) >> 64)))"
        ),
        _ => format!("{v} /* bswap unsupported for {bits}-bit */"),
    };
    let result = bx.new_temp_with_stmt(ty, CExpr::raw(expr));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_bitreverse<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let val = args[0].immediate();
    let v = bx.cx.render_value(val);
    let ty = bx.cx.values.borrow().get_type(val);
    let bits = bx.cx.int_width(ty);
    let t = bx.cx.render_type(ty);

    let result_name = bx.cx.new_temp(ty);
    let rn = bx.cx.render_value(result_name);
    {
        let mut module = bx.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&bx.current_fn) {
            func.add_local_decl(format!("{t} {rn};"));
        }
    }
    let iter = bx.cx.new_temp(bx.cx.type_i32());
    let itn = bx.cx.render_value(iter);
    {
        let mut module = bx.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&bx.current_fn) {
            func.add_local_decl(format!("int32_t {itn};"));
        }
    }
    let ut = if let Some(unsigned_ty) = bx.unsigned_type(ty) {
        bx.cx.render_type(unsigned_ty)
    } else {
        t.clone()
    };
    bx.emit(CStmt::raw(format!("{rn} = 0;")));
    bx.emit(CStmt::raw(format!("for ({itn} = 0; {itn} < {bits}; {itn}++) {{ {rn} = ({t})((({ut}){rn} << 1) | ((({ut}){v} >> {itn}) & 1)); }}")));
    bx.store(result_name, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_saturating<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    name: rustc_span::Symbol,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let lhs = args[0].immediate();
    let rhs = args[1].immediate();
    let ty = bx.cx.values.borrow().get_type(lhs);
    let bits = bx.cx.int_width(ty);
    let l = bx.cx.render_value(lhs);
    let r = bx.cx.render_value(rhs);
    let t = bx.cx.render_type(ty);

    let result = bx.cx.new_temp(ty);
    let result_name = bx.cx.render_value(result);
    let overflow = bx.cx.new_temp(ty);
    let overflow_name = bx.cx.render_value(overflow);
    {
        let mut module = bx.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&bx.current_fn) {
            func.add_local_decl(format!("{t} {result_name};"));
            func.add_local_decl(format!("{t} {overflow_name};"));
        }
    }
    let builtin = if name == sym::saturating_add {
        "__builtin_add_overflow"
    } else {
        "__builtin_sub_overflow"
    };
    let is_signed = matches!(
        bx.cx.types.borrow().get(ty),
        CTypeKind::Int { signed: true, .. }
    );
    let clamp = if is_signed {
        let (min_val, max_val) = if bits <= 64 {
            let max = (1i128 << (bits - 1)) - 1;
            let min = -(1i128 << (bits - 1));
            (format!("({t}){min}LL"), format!("({t}){max}LL"))
        } else {
            ("0".into(), "0".into())
        };
        if name == sym::saturating_add {
            format!("({r} > 0 ? {max_val} : {min_val})")
        } else {
            format!("({r} > 0 ? {min_val} : {max_val})")
        }
    } else {
        let max_val = if bits < 128 {
            format!(
                "({t}){}ULL",
                if bits == 64 {
                    "18446744073709551615".to_string()
                } else {
                    format!("{}", (1u128 << bits) - 1)
                }
            )
        } else {
            format!("({t})-1")
        };
        if name == sym::saturating_add {
            max_val
        } else {
            format!("({t})0")
        }
    };
    bx.emit(CStmt::raw(format!(
        "{overflow_name} = {builtin}({l}, {r}, &{result_name});"
    )));
    bx.emit(CStmt::raw(format!(
        "if ({overflow_name}) {result_name} = {clamp};"
    )));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_rotate<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    name: rustc_span::Symbol,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let val = args[0].immediate();
    let shift = args[1].immediate();
    let ty = bx.cx.values.borrow().get_type(val);
    let bits = bx.cx.int_width(ty);
    let v = bx.cx.render_value(val);
    let s = bx.cx.render_value(shift);
    let ut = if let Some(unsigned_ty) = bx.unsigned_type(ty) {
        bx.cx.render_type(unsigned_ty)
    } else {
        bx.cx.render_type(ty)
    };
    let t = bx.cx.render_type(ty);
    let expr = if name == sym::rotate_left {
        format!("({t})(({ut}){v} << ({s} % {bits}) | ({ut}){v} >> (({bits} - {s}) % {bits}))")
    } else {
        format!("({t})(({ut}){v} >> ({s} % {bits}) | ({ut}){v} << (({bits} - {s}) % {bits}))")
    };
    let result = bx.new_temp_with_stmt(ty, CExpr::raw(expr));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_funnel_shift<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    name: rustc_span::Symbol,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let a = args[0].immediate();
    let b = args[1].immediate();
    let shift = args[2].immediate();
    let ty = bx.cx.values.borrow().get_type(a);
    let bits = bx.cx.int_width(ty);
    let va = bx.cx.render_value(a);
    let vb = bx.cx.render_value(b);
    let vs = bx.cx.render_value(shift);
    let t = bx.cx.render_type(ty);
    let ut = if let Some(unsigned_ty) = bx.unsigned_type(ty) {
        bx.cx.render_type(unsigned_ty)
    } else {
        t.clone()
    };
    let expr = if name == sym::unchecked_funnel_shl {
        format!("({t})((({ut}){va} << {vs}) | (({ut}){vb} >> ({bits} - {vs})))")
    } else {
        format!("({t})((({ut}){va} >> {vs}) | (({ut}){vb} << ({bits} - {vs})))")
    };
    let result = bx.new_temp_with_stmt(ty, CExpr::raw(expr));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

// =====================================================================
// SIMD intrinsic helpers
// =====================================================================

fn codegen_simd_extract<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let vec = args[0].immediate();
    let idx = args[1].immediate();
    let v = bx.cx.render_value(vec);
    let i = bx.cx.render_value(idx);
    let ret_ty = bx.cx.values.borrow().get_type(vec);
    let elem_ty = {
        let types = bx.cx.types.borrow();
        match types.get(ret_ty) {
            CTypeKind::Vector { element, .. } => *element,
            _ => ret_ty,
        }
    };
    let result = bx.new_temp_with_stmt(elem_ty, CExpr::index(CExpr::var(&v), CExpr::var(&i)));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_simd_cmp<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    name: rustc_span::Symbol,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let a = args[0].immediate();
    let b = args[1].immediate();
    let va = bx.cx.render_value(a);
    let vb = bx.cx.render_value(b);
    let c_op = match name {
        sym::simd_eq => "==",
        sym::simd_ne => "!=",
        sym::simd_lt => "<",
        sym::simd_le => "<=",
        sym::simd_gt => ">",
        sym::simd_ge => ">=",
        _ => unreachable!(),
    };
    let a_ty = bx.cx.values.borrow().get_type(a);
    let result = bx.new_temp_with_stmt(a_ty, CExpr::raw(format!("{va} {c_op} {vb}")));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_simd_bitwise<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    name: rustc_span::Symbol,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let a = args[0].immediate();
    let b = args[1].immediate();
    let va = bx.cx.render_value(a);
    let vb = bx.cx.render_value(b);
    let c_op = match name {
        sym::simd_and => "&",
        sym::simd_or => "|",
        sym::simd_xor => "^",
        _ => unreachable!(),
    };
    let a_ty = bx.cx.values.borrow().get_type(a);
    let result = bx.new_temp_with_stmt(a_ty, CExpr::raw(format!("{va} {c_op} {vb}")));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_simd_shift<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    name: rustc_span::Symbol,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let a = args[0].immediate();
    let b = args[1].immediate();
    let va = bx.cx.render_value(a);
    let vb = bx.cx.render_value(b);
    let c_op = if name == sym::simd_shl { "<<" } else { ">>" };
    let a_ty = bx.cx.values.borrow().get_type(a);
    let result = bx.new_temp_with_stmt(a_ty, CExpr::raw(format!("{va} {c_op} {vb}")));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_simd_insert<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let vec = args[0].immediate();
    let idx = args[1].immediate();
    let val = args[2].immediate();
    let v = bx.cx.render_value(vec);
    let i = bx.cx.render_value(idx);
    let e = bx.cx.render_value(val);
    let vec_ty = bx.cx.values.borrow().get_type(vec);
    let result = bx.new_temp_with_stmt(vec_ty, CExpr::raw(v));
    let rn = bx.cx.render_value(result);
    bx.emit(CStmt::assign(
        CExpr::index(CExpr::var(&rn), CExpr::var(&i)),
        CExpr::var(&e),
    ));
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_simd_cast<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let a = args[0].immediate();
    let va = bx.cx.render_value(a);
    let ret_layout = result_dest.layout;
    let ret_ty = bx.cx.backend_type(ret_layout);
    let ret_t = bx.cx.render_type(ret_ty);
    let result = bx.new_temp_with_stmt(
        ret_ty,
        CExpr::call(
            CExpr::var("__builtin_convertvector"),
            vec![CExpr::var(&va), CExpr::raw(ret_t)],
        ),
    );
    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_simd_shuffle<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let in_vec = args[0].immediate();
    let in_vec2 = args[1].immediate();
    let indices = args[2].immediate();

    let in_ty = bx.cx.values.borrow().get_type(in_vec);
    let (elem_ty, in_len) = {
        let types = bx.cx.types.borrow();
        match types.get(in_ty) {
            CTypeKind::Vector { element, len } => (*element, *len as usize),
            _ => (in_ty, 1usize),
        }
    };

    let ret_ty = bx.cx.backend_type(result_dest.layout);
    let out_len = {
        let types = bx.cx.types.borrow();
        match types.get(ret_ty) {
            CTypeKind::Vector { len, .. } => *len as usize,
            _ => 1usize,
        }
    };

    let total_len = in_len * 2;
    let v1 = bx.cx.render_value(in_vec);
    let v2 = bx.cx.render_value(in_vec2);
    let elem_t = bx.cx.render_type(elem_ty);

    let arr_temp = bx.cx.new_temp(elem_ty);
    let arr_name = format!("{}_shuf", bx.cx.render_value(arr_temp));
    {
        let mut module = bx.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&bx.current_fn) {
            func.add_local_decl(format!("{elem_t} {arr_name}[{total_len}];"));
        }
    }
    bx.emit(CStmt::raw(format!(
        "memcpy(&{arr_name}[0], &{v1}, {in_len} * sizeof({elem_t}));"
    )));
    bx.emit(CStmt::raw(format!(
        "memcpy(&{arr_name}[{in_len}], &{v2}, {in_len} * sizeof({elem_t}));"
    )));

    let index_vals: Vec<u64> = {
        let vals = bx.cx.values.borrow();
        match &vals.get(indices).kind {
            CValueKind::VectorConst { elements } => elements
                .iter()
                .map(|e| vals.as_u64(*e).unwrap_or(0))
                .collect(),
            _ => (0..out_len as u64).collect(),
        }
    };

    let result = bx.cx.new_temp(ret_ty);
    let rn = bx.cx.render_value(result);
    let result_decl = bx.cx.render_type_decl(ret_ty, &rn);
    {
        let mut module = bx.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&bx.current_fn) {
            func.add_local_decl(format!("{result_decl};"));
        }
    }
    for (i, &idx_val) in index_vals.iter().enumerate() {
        bx.emit(CStmt::assign(
            CExpr::index(CExpr::var(&rn), CExpr::lit(&i.to_string())),
            CExpr::index(CExpr::var(&arr_name), CExpr::lit(&idx_val.to_string())),
        ));
    }

    bx.store(result, result_dest.val.llval, result_dest.val.align);
    Ok(())
}

fn codegen_typed_swap<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    instance: ty::Instance<'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
) -> Result<(), ty::Instance<'tcx>> {
    let pointee_ty = instance.args.type_at(0);
    let pointee_layout = bx.layout_of(pointee_ty);
    let size = pointee_layout.size.bytes();
    let align = pointee_layout.align.abi;
    let x = bx.cx.render_value(args[0].immediate());
    let y = bx.cx.render_value(args[1].immediate());
    bx.emit(CStmt::raw(format!(
        "{{ _Alignas({}) unsigned char _swap_tmp[{size}];",
        align.bytes()
    )));
    bx.emit(CStmt::raw(format!("  memcpy(_swap_tmp, {x}, {size});")));
    bx.emit(CStmt::raw(format!("  memcpy({x}, {y}, {size});")));
    bx.emit(CStmt::raw(format!("  memcpy({y}, _swap_tmp, {size}); }}")));
    Ok(())
}

fn codegen_carrying_mul_add<'a, 'tcx>(
    bx: &mut Builder<'a, 'tcx>,
    instance: ty::Instance<'tcx>,
    args: &[OperandRef<'tcx, ValueRef>],
    result_dest: PlaceRef<'tcx, ValueRef>,
) -> Result<(), ty::Instance<'tcx>> {
    let int_ty = instance.args.type_at(0);
    let (size, signed) = int_ty.int_size_and_signed(bx.cx.tcx);
    let bits = size.bits();
    let a = bx.cx.render_value(args[0].immediate());
    let b = bx.cx.render_value(args[1].immediate());
    let c = bx.cx.render_value(args[2].immediate());
    let d = bx.cx.render_value(args[3].immediate());
    let narrow_ty = bx.cx.intern_type(CTypeKind::Int {
        bits: bits as u32,
        signed,
    });
    let dest = bx.cx.render_value(result_dest.val.llval);

    if bits <= 64 {
        let wide_bits = bits * 2;
        let (wide_t, narrow_t) = if signed {
            (format!("int{wide_bits}_t"), format!("int{bits}_t"))
        } else {
            (format!("uint{wide_bits}_t"), format!("uint{bits}_t"))
        };
        let wide_ty = bx.cx.intern_type(CTypeKind::Int {
            bits: wide_bits as u32,
            signed,
        });
        let wide = bx.new_temp_with_stmt(
            wide_ty,
            CExpr::raw(format!(
                "({wide_t}){a} * ({wide_t}){b} + ({wide_t}){c} + ({wide_t}){d}"
            )),
        );
        let wide_v = bx.cx.render_value(wide);
        let low = bx.new_temp_with_stmt(narrow_ty, CExpr::cast(&narrow_t, CExpr::var(&wide_v)));
        let high = bx.new_temp_with_stmt(
            narrow_ty,
            CExpr::raw(format!("({narrow_t})(({wide_t}){wide_v} >> {bits})")),
        );
        let low_v = bx.cx.render_value(low);
        let high_v = bx.cx.render_value(high);
        bx.emit(CStmt::assign(
            CExpr::deref(format!("{narrow_t} *"), CExpr::var(&dest)),
            CExpr::var(&low_v),
        ));
        bx.emit(CStmt::assign(
            CExpr::deref(
                format!("{narrow_t} *"),
                CExpr::paren(CExpr::binop(
                    CExpr::cast("char *", CExpr::var(&dest)),
                    CBinOp::Add,
                    CExpr::lit(format!("{}", size.bytes())),
                )),
            ),
            CExpr::var(&high_v),
        ));
    } else {
        let t = if signed { "int128_t" } else { "uint128_t" };
        bx.emit(CStmt::raw(format!("{{ /* carrying_mul_add u128 */")));
        bx.emit(CStmt::raw(format!("  uint64_t _a_lo = (uint64_t){a};")));
        bx.emit(CStmt::raw(format!(
            "  uint64_t _a_hi = (uint64_t)((uint128_t){a} >> 64);"
        )));
        bx.emit(CStmt::raw(format!("  uint64_t _b_lo = (uint64_t){b};")));
        bx.emit(CStmt::raw(format!(
            "  uint64_t _b_hi = (uint64_t)((uint128_t){b} >> 64);"
        )));
        bx.emit(CStmt::raw(format!(
            "  uint128_t _lo_lo = (uint128_t)_a_lo * _b_lo;"
        )));
        bx.emit(CStmt::raw(format!(
            "  uint128_t _hi_lo = (uint128_t)_a_hi * _b_lo;"
        )));
        bx.emit(CStmt::raw(format!(
            "  uint128_t _lo_hi = (uint128_t)_a_lo * _b_hi;"
        )));
        bx.emit(CStmt::raw(format!(
            "  uint128_t _hi_hi = (uint128_t)_a_hi * _b_hi;"
        )));
        bx.emit(CStmt::raw(format!("  uint128_t _mid = _hi_lo + _lo_hi;")));
        bx.emit(CStmt::raw(format!(
            "  uint128_t _carry_mid = (_mid < _hi_lo) ? ((uint128_t)1 << 64) : 0;"
        )));
        bx.emit(CStmt::raw(format!(
            "  uint128_t _low = _lo_lo + (_mid << 64);"
        )));
        bx.emit(CStmt::raw(format!(
            "  uint128_t _carry_low = (_low < _lo_lo) ? 1 : 0;"
        )));
        bx.emit(CStmt::raw(format!(
            "  uint128_t _high = _hi_hi + (_mid >> 64) + _carry_mid + _carry_low;"
        )));
        bx.emit(CStmt::raw(format!("  _low += (uint128_t){c};")));
        bx.emit(CStmt::raw(format!(
            "  if (_low < (uint128_t){c}) _high += 1;"
        )));
        bx.emit(CStmt::raw(format!("  _low += (uint128_t){d};")));
        bx.emit(CStmt::raw(format!(
            "  if (_low < (uint128_t){d}) _high += 1;"
        )));
        bx.emit(CStmt::raw(format!("  *(({t} *){dest}) = ({t})_low;")));
        bx.emit(CStmt::raw(format!(
            "  *(({t} *)((char *){dest} + {})) = ({t})_high;",
            size.bytes()
        )));
        bx.emit(CStmt::raw(format!("}}")));
    }
    Ok(())
}
