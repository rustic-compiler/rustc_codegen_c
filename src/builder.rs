/// The C code builder: emits C statements for each MIR operation.
///
/// The builder implements `BuilderMethods` and all its supertraits.
/// Each operation creates a C statement and appends it to the current
/// basic block. Intermediate results are stored in temporary variables.
use std::cmp;
use std::ops::Deref;

use rustc_abi::{Align, Scalar, Size, WrappingRange};
use rustc_ast::expand::typetree::FncTree;
use rustc_codegen_ssa::MemFlags;
use rustc_codegen_ssa::common::{
    AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope,
};
use rustc_codegen_ssa::mir::operand::{OperandRef, OperandValue};
use rustc_codegen_ssa::mir::place::{PlaceRef, PlaceValue};
use rustc_codegen_ssa::traits::*;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs;
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTypingEnv, LayoutError, LayoutOfHelpers,
};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_span::Span;
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};

use crate::c_ast::{CBinOp, CExpr, CStmt, CUnaryOp};
use crate::context::{CFunclet, CodegenCx, DebugLoc, DebugScope, DebugVar};
use crate::module::BasicBlockId;
use crate::types::{CTypeKind, TypeRef};
use crate::values::{CValueKind, ValueRef};

/// The C code builder.
pub(crate) struct Builder<'a, 'tcx> {
    pub(crate) cx: &'a CodegenCx<'tcx>,
    pub(crate) current_bb: BasicBlockId,
    pub(crate) current_fn: String,
}

impl<'a, 'tcx> Deref for Builder<'a, 'tcx> {
    type Target = CodegenCx<'tcx>;
    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    /// Emit a C statement to the current basic block.
    pub(crate) fn emit(&self, stmt: CStmt) {
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.emit(self.current_bb, stmt);
        }
    }

    /// Create a new temporary variable and emit its declaration + assignment.
    pub(crate) fn new_temp_with_stmt(&self, ty: TypeRef, expr: CExpr) -> ValueRef {
        let val = self.cx.new_temp(ty);
        let name = self.cx.render_value(val);
        let decl_str = self.cx.render_type_decl(ty, &name);

        // Add local var declaration
        {
            let mut module = self.cx.module.borrow_mut();
            if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                func.add_local_decl(format!("{decl_str};"));
            }
        }

        // Emit assignment
        self.emit(CStmt::assign(CExpr::var(&name), expr));
        val
    }

    /// Emit a binary operation.
    fn binop(&mut self, op: CBinOp, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        let ty = self.cx.values.borrow().get_type(lhs);
        let l = self.cx.render_value(lhs);
        let r = self.cx.render_value(rhs);
        self.new_temp_with_stmt(ty, CExpr::binop(CExpr::var(l), op, CExpr::var(r)))
    }

    /// Emit a cast operation.
    fn cast(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        let v = self.cx.render_value(val);
        let t = self.cx.render_type(dest_ty);
        self.new_temp_with_stmt(dest_ty, CExpr::cast(t, CExpr::var(v)))
    }

    /// Get the unsigned variant of an integer type.
    /// Returns None if the type is not a signed integer.
    pub(crate) fn unsigned_type(&self, ty: TypeRef) -> Option<TypeRef> {
        let types = self.cx.types.borrow();
        match types.get(ty) {
            CTypeKind::Int { bits, signed: true } => {
                let bits = *bits;
                drop(types);
                Some(self.cx.intern_type(CTypeKind::Int {
                    bits,
                    signed: false,
                }))
            }
            CTypeKind::PtrWidth { signed: true } => {
                drop(types);
                Some(self.cx.intern_type(CTypeKind::PtrWidth { signed: false }))
            }
            _ => None,
        }
    }

    /// Emit a binary operation with operands cast to unsigned.
    fn unsigned_binop(&mut self, op: CBinOp, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        let ty = self.cx.values.borrow().get_type(lhs);
        if let Some(unsigned_ty) = self.unsigned_type(ty) {
            let l = self.cx.render_value(lhs);
            let r = self.cx.render_value(rhs);
            let ut = self.cx.render_type(unsigned_ty);
            let t = self.cx.render_type(ty);
            self.new_temp_with_stmt(
                ty,
                CExpr::cast(
                    t,
                    CExpr::paren(CExpr::binop(
                        CExpr::cast(&ut, CExpr::var(l)),
                        op,
                        CExpr::cast(ut, CExpr::var(r)),
                    )),
                ),
            )
        } else {
            self.binop(op, lhs, rhs)
        }
    }

    /// Convert Rust AtomicOrdering to C11 memory_order string.
    fn atomic_ordering_to_c(order: rustc_middle::ty::AtomicOrdering) -> &'static str {
        use rustc_middle::ty::AtomicOrdering;
        match order {
            AtomicOrdering::Relaxed => "memory_order_relaxed",
            AtomicOrdering::Acquire => "memory_order_acquire",
            AtomicOrdering::Release => "memory_order_release",
            AtomicOrdering::AcqRel => "memory_order_acq_rel",
            AtomicOrdering::SeqCst => "memory_order_seq_cst",
        }
    }

    /// Get the block label.
    pub(crate) fn block_label(&self, bb: BasicBlockId) -> String {
        let module = self.cx.module.borrow();
        if let Some(func) = module.open_functions.get(&self.current_fn) {
            if let Some(block) = func.blocks.get(&bb.0) {
                return block.label.clone();
            }
        }
        format!("bb{}", bb.0)
    }
}

// --- BackendTypes (delegated from CodegenCx) ---

impl<'a, 'tcx> BackendTypes for Builder<'a, 'tcx> {
    type Value = ValueRef;
    type Metadata = ValueRef;
    type Function = ValueRef;
    type BasicBlock = BasicBlockId;
    type Type = TypeRef;
    type Funclet = CFunclet;
    type DIScope = DebugScope;
    type DILocation = DebugLoc;
    type DIVariable = DebugVar;
}

// --- HasTyCtxt, HasTypingEnv, etc. (delegate to CodegenCx) ---

impl<'a, 'tcx> rustc_middle::ty::layout::HasTyCtxt<'tcx> for Builder<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }
}

impl<'a, 'tcx> HasTypingEnv<'tcx> for Builder<'a, 'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        self.cx.typing_env()
    }
}

impl<'a, 'tcx> rustc_abi::HasDataLayout for Builder<'a, 'tcx> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        self.cx.data_layout()
    }
}

impl<'a, 'tcx> LayoutOfHelpers<'tcx> for Builder<'a, 'tcx> {
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        self.cx.handle_layout_err(err, span, ty)
    }
}

impl<'a, 'tcx> FnAbiOfHelpers<'tcx> for Builder<'a, 'tcx> {
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        self.cx.handle_fn_abi_err(err, span, fn_abi_request)
    }
}

// =====================================================================
// BuilderMethods
// =====================================================================

impl<'a, 'tcx> BuilderMethods<'a, 'tcx> for Builder<'a, 'tcx> {
    type CodegenCx = CodegenCx<'tcx>;

    fn build(cx: &'a CodegenCx<'tcx>, llbb: BasicBlockId) -> Self {
        // O(1) lookup via reverse map
        let module = cx.module.borrow();
        let fn_name = module
            .block_to_function
            .get(&llbb.0)
            .cloned()
            .unwrap_or_default();
        drop(module);

        Builder {
            cx,
            current_bb: llbb,
            current_fn: fn_name,
        }
    }

    fn cx(&self) -> &CodegenCx<'tcx> {
        self.cx
    }

    fn llbb(&self) -> BasicBlockId {
        self.current_bb
    }

    fn set_span(&mut self, _span: Span) {
        // Could emit source location comments
    }

    fn append_block(cx: &'a CodegenCx<'tcx>, llfn: ValueRef, name: &str) -> BasicBlockId {
        let fn_name = cx.render_value(llfn);
        let mut module = cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&fn_name) {
            let label = if name.is_empty() {
                format!("bb{}", func.block_counter)
            } else {
                format!(
                    "bb_{}_{}",
                    func.block_counter,
                    CodegenCx::sanitize_name(name)
                )
            };
            let bb = func.new_block(label);
            module.block_to_function.insert(bb.0, fn_name);
            return bb;
        }
        // Function not open yet; return a placeholder
        BasicBlockId(u32::MAX)
    }

    fn append_sibling_block(&mut self, name: &str) -> BasicBlockId {
        let fn_name = self.current_fn.clone();
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&fn_name) {
            let label = format!(
                "bb_{}_{}",
                func.block_counter,
                CodegenCx::sanitize_name(name)
            );
            let bb = func.new_block(label);
            module.block_to_function.insert(bb.0, fn_name);
            return bb;
        }
        BasicBlockId(u32::MAX)
    }

    fn switch_to_block(&mut self, llbb: BasicBlockId) {
        self.current_bb = llbb;
    }

    fn ret_void(&mut self) {
        let retbuf = {
            let module = self.cx.module.borrow();
            module
                .open_functions
                .get(&self.current_fn)
                .and_then(|f| f.retbuf_name.clone())
        };
        if let Some(name) = retbuf {
            self.emit(CStmt::ret_val(CExpr::var(name)));
        } else {
            self.emit(CStmt::ret_void());
        }
        let bb = self.current_bb;
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.set_terminated(bb);
        }
    }

    fn ret(&mut self, v: ValueRef) {
        // Check if the current function returns void (indirect return).
        // In that case the result was already written to the output
        // pointer (param 0) by codegen_ssa, so just emit `return;`.
        let is_void = {
            let module = self.cx.module.borrow();
            module
                .open_functions
                .get(&self.current_fn)
                .map(|f| matches!(self.cx.types.borrow().get(f.return_type), CTypeKind::Void))
                .unwrap_or(false)
        };
        if is_void {
            self.emit(CStmt::ret_void());
        } else {
            let val = self.cx.render_value(v);
            self.emit(CStmt::ret_val(CExpr::var(val)));
        }
        let bb = self.current_bb;
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.set_terminated(bb);
        }
    }

    fn br(&mut self, dest: BasicBlockId) {
        let label = self.block_label(dest);
        self.emit(CStmt::goto(label));
        let bb = self.current_bb;
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.set_terminated(bb);
        }
    }

    fn cond_br(&mut self, cond: ValueRef, then_llbb: BasicBlockId, else_llbb: BasicBlockId) {
        let c = self.cx.render_value(cond);
        let then_label = self.block_label(then_llbb);
        let else_label = self.block_label(else_llbb);
        self.emit(CStmt::CondGoto {
            cond: CExpr::var(c),
            then_label,
            else_label,
        });
        let bb = self.current_bb;
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.set_terminated(bb);
        }
    }

    fn switch(
        &mut self,
        v: ValueRef,
        else_llbb: BasicBlockId,
        cases: impl ExactSizeIterator<Item = (u128, BasicBlockId)>,
    ) {
        let val = self.cx.render_value(v);
        let else_label = self.block_label(else_llbb);
        // Cast the switch value to unsigned to ensure case constants
        // (which are u128 bit patterns) match correctly. Without this,
        // a signed int8_t value like -16 (0xF0) would not match
        // case 240ULL, because C promotes int8_t to int (-16) which
        // doesn't equal 240.
        let val_ty = self.cx.values.borrow().get_type(v);
        let types = self.cx.types.borrow();
        let switch_expr = match types.get(val_ty) {
            CTypeKind::Int { bits, signed: true } => {
                let unsigned_ty = format!("uint{bits}_t");
                CExpr::cast(unsigned_ty, CExpr::var(val))
            }
            CTypeKind::PtrWidth { signed: true } => CExpr::cast("uintptr_t", CExpr::var(val)),
            _ => CExpr::var(val),
        };
        drop(types);
        let ast_cases: Vec<(CExpr, String)> = cases
            .map(|(constant, bb)| {
                let label = self.block_label(bb);
                let case_expr = if constant > u64::MAX as u128 {
                    let hi = (constant >> 64) as u64;
                    let lo = constant as u64;
                    CExpr::raw(format!("((uint128_t){hi}ULL << 64 | {lo}ULL)"))
                } else {
                    CExpr::lit(format!("{constant}ULL"))
                };
                (case_expr, label)
            })
            .collect();
        self.emit(CStmt::Switch {
            expr: switch_expr,
            cases: ast_cases,
            default: else_label,
        });
        let bb = self.current_bb;
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.set_terminated(bb);
        }
    }

    fn invoke(
        &mut self,
        llty: TypeRef,
        _fn_attrs: Option<&CodegenFnAttrs>,
        _fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: ValueRef,
        args: &[ValueRef],
        then: BasicBlockId,
        catch: BasicBlockId,
        _funclet: Option<&CFunclet>,
        _instance: Option<Instance<'tcx>>,
    ) -> ValueRef {
        // setjmp/longjmp-based invoke: push an unwind context onto the
        // thread-local chain.  If the callee (or anything it calls) triggers
        // _Unwind_RaiseException, our override longjmps back here and we
        // branch to the cleanup/catch block.
        //
        // All invokes within a function share a single __unwind_ctx to
        // minimize stack usage (jmp_buf is ~300 bytes).  This is safe
        // because invokes are sequential -- only one setjmp is active at
        // a time within a given function frame.
        {
            let mut module = self.cx.module.borrow_mut();
            let func = module.open_functions.get_mut(&self.current_fn).unwrap();
            if func.invoke_counter == 0 {
                func.add_local_decl("void *__exn_ptr;".to_string());
                func.add_local_decl("struct __rustc_unwind_context __unwind_ctx;".to_string());
            }
            func.invoke_counter += 1;
        }

        let catch_label = self.block_label(catch);
        let then_label = self.block_label(then);

        // Push context onto chain
        self.emit(CStmt::assign(
            CExpr::field(CExpr::var("__unwind_ctx"), "prev"),
            CExpr::var("__rustc_unwind_chain"),
        ));
        self.emit(CStmt::assign(
            CExpr::field(CExpr::var("__unwind_ctx"), "exception_ptr"),
            CExpr::cast("void *", CExpr::lit("0")),
        ));
        self.emit(CStmt::assign(
            CExpr::var("__rustc_unwind_chain"),
            CExpr::addr_of(CExpr::var("__unwind_ctx")),
        ));
        // __builtin_setjmp returns 0 normally; non-zero after __builtin_longjmp
        self.emit(CStmt::raw(format!(
            "if (__rustc_setjmp(__unwind_ctx.buf) != 0) {{ \
                __exn_ptr = __unwind_ctx.exception_ptr; \
                __rustc_unwind_chain = __unwind_ctx.prev; \
                goto {catch_label}; }}"
        )));

        // Normal path: call the function
        let result = self.call(llty, None, _fn_abi, llfn, args, None, _instance);

        // Pop context and continue to normal successor
        self.emit(CStmt::assign(
            CExpr::var("__rustc_unwind_chain"),
            CExpr::field(CExpr::var("__unwind_ctx"), "prev"),
        ));
        self.emit(CStmt::goto(then_label));

        let bb = self.current_bb;
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.set_terminated(bb);
        }
        result
    }

    fn unreachable(&mut self) {
        self.emit(CStmt::Unreachable);
        let bb = self.current_bb;
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.set_terminated(bb);
        }
    }

    // -- Arithmetic --

    fn add(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Add, lhs, rhs)
    }
    fn fadd(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Add, lhs, rhs)
    }
    fn fadd_fast(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Add, lhs, rhs)
    }
    fn fadd_algebraic(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Add, lhs, rhs)
    }
    fn sub(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Sub, lhs, rhs)
    }
    fn fsub(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Sub, lhs, rhs)
    }
    fn fsub_fast(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Sub, lhs, rhs)
    }
    fn fsub_algebraic(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Sub, lhs, rhs)
    }
    fn mul(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Mul, lhs, rhs)
    }
    fn fmul(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Mul, lhs, rhs)
    }
    fn fmul_fast(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Mul, lhs, rhs)
    }
    fn fmul_algebraic(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Mul, lhs, rhs)
    }
    fn udiv(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.unsigned_binop(CBinOp::Div, lhs, rhs)
    }
    fn exactudiv(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.unsigned_binop(CBinOp::Div, lhs, rhs)
    }
    fn sdiv(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Div, lhs, rhs)
    }
    fn exactsdiv(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Div, lhs, rhs)
    }
    fn fdiv(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Div, lhs, rhs)
    }
    fn fdiv_fast(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Div, lhs, rhs)
    }
    fn fdiv_algebraic(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Div, lhs, rhs)
    }
    fn urem(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.unsigned_binop(CBinOp::Rem, lhs, rhs)
    }
    fn srem(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::Rem, lhs, rhs)
    }
    fn frem(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        let ty = self.cx.values.borrow().get_type(lhs);
        let l = self.cx.render_value(lhs);
        let r = self.cx.render_value(rhs);
        self.new_temp_with_stmt(
            ty,
            CExpr::call(CExpr::var("fmod"), vec![CExpr::var(l), CExpr::var(r)]),
        )
    }
    fn frem_fast(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.frem(lhs, rhs)
    }
    fn frem_algebraic(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.frem(lhs, rhs)
    }

    fn shl(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        // Cast LHS to result type before shifting to avoid UB when a
        // narrow literal (e.g. 1ULL) is used for a wider operation
        // (e.g. 1u128 << 122).
        let ty = self.cx.values.borrow().get_type(lhs);
        let l = self.cx.render_value(lhs);
        let r = self.cx.render_value(rhs);
        let t = self.cx.render_type(ty);
        self.new_temp_with_stmt(
            ty,
            CExpr::binop(
                CExpr::cast(t, CExpr::paren(CExpr::var(l))),
                CBinOp::Shl,
                CExpr::paren(CExpr::var(r)),
            ),
        )
    }
    fn lshr(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        // Logical shift right: cast to unsigned, shift, cast back
        let ty = self.cx.values.borrow().get_type(lhs);
        let l = self.cx.render_value(lhs);
        let r = self.cx.render_value(rhs);
        if let Some(unsigned_ty) = self.unsigned_type(ty) {
            let ut = self.cx.render_type(unsigned_ty);
            let t = self.cx.render_type(ty);
            self.new_temp_with_stmt(
                ty,
                CExpr::cast(
                    t,
                    CExpr::paren(CExpr::binop(
                        CExpr::cast(ut, CExpr::paren(CExpr::var(l))),
                        CBinOp::Shr,
                        CExpr::paren(CExpr::var(r)),
                    )),
                ),
            )
        } else {
            let t = self.cx.render_type(ty);
            self.new_temp_with_stmt(
                ty,
                CExpr::binop(
                    CExpr::cast(t, CExpr::paren(CExpr::var(l))),
                    CBinOp::Shr,
                    CExpr::paren(CExpr::var(r)),
                ),
            )
        }
    }
    fn ashr(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        // Cast LHS to result type before shifting (same rationale as shl)
        let ty = self.cx.values.borrow().get_type(lhs);
        let l = self.cx.render_value(lhs);
        let r = self.cx.render_value(rhs);
        let t = self.cx.render_type(ty);
        self.new_temp_with_stmt(
            ty,
            CExpr::binop(
                CExpr::cast(t, CExpr::paren(CExpr::var(l))),
                CBinOp::Shr,
                CExpr::paren(CExpr::var(r)),
            ),
        )
    }

    fn and(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::BitAnd, lhs, rhs)
    }
    fn or(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::BitOr, lhs, rhs)
    }
    fn xor(&mut self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.binop(CBinOp::BitXor, lhs, rhs)
    }

    fn neg(&mut self, v: ValueRef) -> ValueRef {
        let ty = self.cx.values.borrow().get_type(v);
        let val = self.cx.render_value(v);
        self.new_temp_with_stmt(ty, CExpr::unary(CUnaryOp::Neg, CExpr::var(val)))
    }
    fn fneg(&mut self, v: ValueRef) -> ValueRef {
        self.neg(v)
    }
    fn not(&mut self, v: ValueRef) -> ValueRef {
        let ty = self.cx.values.borrow().get_type(v);
        let val = self.cx.render_value(v);
        let types = self.cx.types.borrow();
        let is_bool = matches!(
            types.get(ty),
            CTypeKind::Bool | CTypeKind::Int { bits: 1, .. }
        );
        drop(types);
        if is_bool {
            self.new_temp_with_stmt(ty, CExpr::unary(CUnaryOp::LogNot, CExpr::var(val)))
        } else {
            self.new_temp_with_stmt(ty, CExpr::unary(CUnaryOp::BitNot, CExpr::var(val)))
        }
    }

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        _ty: Ty<'tcx>,
        lhs: ValueRef,
        rhs: ValueRef,
    ) -> (ValueRef, ValueRef) {
        // Use GCC/Clang __builtin_*_overflow.
        // The value type now carries the correct signedness (e.g. uint32_t
        // for u32, int32_t for i32), so __builtin_*_overflow detects the
        // right kind of overflow based on the output type directly.
        let val_ty = self.cx.values.borrow().get_type(lhs);
        let result = self.cx.new_temp(val_ty);
        let result_name = self.cx.render_value(result);
        let type_str = self.cx.render_type(val_ty);

        {
            let mut module = self.cx.module.borrow_mut();
            if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                func.add_local_decl(format!("{type_str} {result_name};"));
            }
        }

        let l = self.cx.render_value(lhs);
        let r = self.cx.render_value(rhs);
        let builtin = match oop {
            OverflowOp::Add => "__builtin_add_overflow",
            OverflowOp::Sub => "__builtin_sub_overflow",
            OverflowOp::Mul => "__builtin_mul_overflow",
        };

        let bool_ty = self.cx.intern_type(CTypeKind::Bool);
        let overflow = self.cx.new_temp(bool_ty);
        let overflow_name = self.cx.render_value(overflow);

        {
            let mut module = self.cx.module.borrow_mut();
            if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                func.add_local_decl(format!("_Bool {overflow_name};"));
            }
        }

        self.emit(CStmt::assign(
            CExpr::var(&overflow_name),
            CExpr::call(
                CExpr::var(builtin),
                vec![
                    CExpr::var(l),
                    CExpr::var(r),
                    CExpr::addr_of(CExpr::var(&result_name)),
                ],
            ),
        ));
        (result, overflow)
    }

    fn from_immediate(&mut self, val: ValueRef) -> ValueRef {
        // Bool (C _Bool / LLVM i1) -> int8_t, mirroring LLVM's i1->i8 zext.
        // This is needed for transmute_scalar assertions that check
        // val_ty matches type_from_scalar.
        let ty = self.cx.values.borrow().get_type(val);
        if matches!(self.cx.types.borrow().get(ty), CTypeKind::Bool) {
            let i8_ty = self.cx.intern_type(CTypeKind::Int {
                bits: 8,
                signed: true,
            });
            let v = self.cx.render_value(val);
            self.new_temp_with_stmt(i8_ty, CExpr::cast("int8_t", CExpr::paren(CExpr::var(v))))
        } else {
            val
        }
    }
    fn to_immediate_scalar(&mut self, val: ValueRef, scalar: Scalar) -> ValueRef {
        // int8_t -> Bool for boolean scalars, mirroring LLVM's i8->i1 trunc.
        if scalar.is_bool() {
            let bool_ty = self.cx.intern_type(CTypeKind::Bool);
            let actual_ty = self.cx.values.borrow().get_type(val);
            if actual_ty != bool_ty {
                let v = self.cx.render_value(val);
                return self.new_temp_with_stmt(
                    bool_ty,
                    CExpr::cast("_Bool", CExpr::paren(CExpr::var(v))),
                );
            }
            return val;
        }
        // Ensure the value's stored type matches what scalar_to_c_type would produce.
        // This is needed because const_struct etc. may create values with struct types
        // that are later used as scalars. We re-wrap the value with the correct type
        // without changing the C expression.
        let expected_ty = crate::type_of::scalar_to_c_type(self.cx, scalar);
        let actual_ty = self.cx.values.borrow().get_type(val);
        if actual_ty != expected_ty {
            // Create a new value with the correct type but same C expression
            let expr = self.cx.render_value(val);
            self.cx
                .intern_value(CValueKind::InlineExpr(expr), expected_ty)
        } else {
            val
        }
    }

    // -- Memory --

    fn alloca(&mut self, size: Size, align: Align) -> ValueRef {
        let byte_ty = self.cx.intern_type(CTypeKind::Int {
            bits: 8,
            signed: true,
        });
        let _arr_ty = self.cx.intern_type(CTypeKind::Array {
            element: byte_ty,
            len: size.bytes(),
        });
        let ptr_ty = self.cx.type_ptr();
        let val = self.cx.new_temp(ptr_ty);
        let name = self.cx.render_value(val);
        let arr_name = format!("{name}_storage");

        {
            let mut module = self.cx.module.borrow_mut();
            if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                func.add_local_decl(format!(
                    "_Alignas({}) uint8_t {arr_name}[{}];",
                    align.bytes(),
                    size.bytes()
                ));
                // Initialize pointer in declaration so it's valid in ALL
                // basic blocks (including cleanup blocks reached via unwind
                // paths that don't go through the block where alloca was
                // originally called).
                func.add_local_decl(format!("void *{name} = (void *){arr_name};"));
            }
        }
        val
    }

    fn scalable_alloca(&mut self, elt: u64, align: Align, _element_ty: Ty<'_>) -> ValueRef {
        self.alloca(Size::from_bytes(elt), align)
    }

    fn load(&mut self, ty: TypeRef, ptr: ValueRef, _align: Align) -> ValueRef {
        let p = self.cx.render_value(ptr);
        let type_kind = self.cx.types.borrow().get(ty).clone();
        match &type_kind {
            CTypeKind::Array { .. } => {
                // Can't cast to array pointer in C; use memcpy
                let result = self.cx.new_temp(ty);
                let result_name = self.cx.render_value(result);
                let t = self.cx.render_type_decl(ty, &result_name);
                {
                    let mut module = self.cx.module.borrow_mut();
                    if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                        func.add_local_decl(format!("{t};"));
                    }
                }
                self.emit(CStmt::expr(CExpr::call(
                    CExpr::var("memcpy"),
                    vec![
                        CExpr::var(&result_name),
                        CExpr::var(&p),
                        CExpr::sizeof_expr(CExpr::var(&result_name)),
                    ],
                )));
                result
            }
            CTypeKind::Function { .. } => {
                // Function pointer load: *(void (**)(args))ptr
                // render_decl with name "*" gives "ret (**)(args)" which is pointer-to-fn-ptr
                let ptr_to_fnptr = self.cx.render_type_decl(ty, "*");
                self.new_temp_with_stmt(ty, CExpr::deref(ptr_to_fnptr, CExpr::var(&p)))
            }
            _ => {
                let t = self.cx.render_type(ty);
                self.new_temp_with_stmt(ty, CExpr::deref(format!("{t} *"), CExpr::var(&p)))
            }
        }
    }

    fn volatile_load(&mut self, ty: TypeRef, ptr: ValueRef) -> ValueRef {
        let p = self.cx.render_value(ptr);
        let t = self.cx.render_type(ty);
        self.new_temp_with_stmt(ty, CExpr::deref(format!("volatile {t} *"), CExpr::var(&p)))
    }

    fn atomic_load(
        &mut self,
        ty: TypeRef,
        ptr: ValueRef,
        _order: rustc_middle::ty::AtomicOrdering,
        size: Size,
    ) -> ValueRef {
        let p = self.cx.render_value(ptr);
        let t = self.cx.render_type(ty);
        if size.bytes() > 8 {
            // 128-bit atomics: use __sync_val_compare_and_swap to avoid
            // libatomic dependency (__atomic_load_16).
            self.new_temp_with_stmt(
                ty,
                CExpr::call(
                    CExpr::var("__sync_val_compare_and_swap"),
                    vec![
                        CExpr::cast(format!("{t} *"), CExpr::var(&p)),
                        CExpr::lit("0"),
                        CExpr::lit("0"),
                    ],
                ),
            )
        } else {
            self.new_temp_with_stmt(
                ty,
                CExpr::call(
                    CExpr::var("atomic_load"),
                    vec![CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&p))],
                ),
            )
        }
    }

    fn load_operand(&mut self, place: PlaceRef<'tcx, ValueRef>) -> OperandRef<'tcx, ValueRef> {
        let ty = place.layout;
        if ty.is_zst() {
            return OperandRef::zero_sized(ty);
        }

        let val = if self.is_backend_immediate(ty) {
            let llty = self.backend_type(ty);
            let mut llval = self.load(llty, place.val.llval, place.val.align);
            // Convert int8_t -> _Bool for boolean scalars so that
            // operations like `not` use logical `!` instead of
            // bitwise `~`.
            if let rustc_abi::BackendRepr::Scalar(scalar) = ty.backend_repr {
                llval = self.to_immediate_scalar(llval, scalar);
            }
            OperandValue::Immediate(llval)
        } else if let rustc_abi::BackendRepr::ScalarPair(a, b) = ty.backend_repr {
            let a_ty = self.scalar_pair_element_backend_type(ty, 0, true);
            let b_ty = self.scalar_pair_element_backend_type(ty, 1, true);
            let ptr = place.val.llval;

            let a_val = self.load(a_ty, ptr, place.val.align);

            // Compute offset like LLVM: a.size aligned to b.align
            let b_offset = a.size(self).align_to(b.align(self).abi);
            let b_ptr = if b_offset.bytes() > 0 {
                self.ptradd(ptr, self.const_usize(b_offset.bytes()))
            } else {
                ptr
            };
            let b_val = self.load(b_ty, b_ptr, place.val.align);
            OperandValue::Pair(a_val, b_val)
        } else {
            OperandValue::Ref(place.val)
        };

        OperandRef {
            val,
            layout: ty,
            move_annotation: None,
        }
    }

    fn write_operand_repeatedly(
        &mut self,
        elem: OperandRef<'tcx, ValueRef>,
        count: u64,
        dest: PlaceRef<'tcx, ValueRef>,
    ) {
        let stride = dest.layout.field(self.cx(), 0).size;
        for i in 0..count {
            let offset = stride * i;
            let dest_ptr = if offset.bytes() > 0 {
                let off = self.const_usize(offset.bytes());
                self.ptradd(dest.val.llval, off)
            } else {
                dest.val.llval
            };
            let dest_place = PlaceRef {
                val: PlaceValue::new_sized(dest_ptr, dest.layout.align.abi),
                layout: dest.layout.field(self.cx(), 0),
            };
            elem.val.store(self, dest_place);
        }
    }

    fn range_metadata(&mut self, _load: ValueRef, _range: WrappingRange) {
        // No-op: C doesn't have range metadata
    }

    fn nonnull_metadata(&mut self, _load: ValueRef) {
        // No-op
    }

    fn store(&mut self, val: ValueRef, ptr: ValueRef, _align: Align) -> ValueRef {
        let v = self.cx.render_value(val);
        let p = self.cx.render_value(ptr);
        let ty = self.cx.values.borrow().get_type(val);
        let is_array = matches!(self.cx.types.borrow().get(ty), CTypeKind::Array { .. });
        if is_array {
            self.emit(CStmt::expr(CExpr::call(
                CExpr::var("memcpy"),
                vec![
                    CExpr::var(&p),
                    CExpr::addr_of(CExpr::var(&v)),
                    CExpr::sizeof_expr(CExpr::var(&v)),
                ],
            )));
        } else {
            let t = self.cx.render_type(ty);
            self.emit(CStmt::assign(
                CExpr::deref(format!("{t} *"), CExpr::var(&p)),
                CExpr::var(&v),
            ));
        }
        val
    }

    fn store_with_flags(
        &mut self,
        val: ValueRef,
        ptr: ValueRef,
        align: Align,
        _flags: MemFlags,
    ) -> ValueRef {
        self.store(val, ptr, align)
    }

    fn atomic_store(
        &mut self,
        val: ValueRef,
        ptr: ValueRef,
        _order: rustc_middle::ty::AtomicOrdering,
        size: Size,
    ) {
        let v = self.cx.render_value(val);
        let p = self.cx.render_value(ptr);
        let ty = self.cx.values.borrow().get_type(val);
        let t = self.cx.render_type(ty);
        if size.bytes() > 8 {
            // 128-bit atomics: use CAS loop to avoid libatomic dependency.
            let old = self.cx.new_temp(ty);
            let old_name = self.cx.render_value(old);
            {
                let mut module = self.cx.module.borrow_mut();
                if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                    func.add_local_decl(format!("{t} {old_name};"));
                }
            }
            let cast_ptr = CExpr::cast(format!("{t} *"), CExpr::var(&p));
            self.emit(CStmt::DoWhile {
                body: vec![CStmt::assign(
                    CExpr::var(&old_name),
                    CExpr::deref(format!("{t} *"), CExpr::var(&p)),
                )],
                cond: CExpr::unary(
                    CUnaryOp::LogNot,
                    CExpr::call(
                        CExpr::var("__sync_bool_compare_and_swap"),
                        vec![cast_ptr, CExpr::var(&old_name), CExpr::var(&v)],
                    ),
                ),
            });
        } else {
            self.emit(CStmt::expr(CExpr::call(
                CExpr::var("atomic_store"),
                vec![
                    CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&p)),
                    CExpr::var(&v),
                ],
            )));
        }
    }

    fn gep(&mut self, ty: TypeRef, ptr: ValueRef, indices: &[ValueRef]) -> ValueRef {
        // Use uintptr_t integer arithmetic instead of C pointer arithmetic
        // to avoid UB when the pointer doesn't point to a valid C object
        // (e.g. tagged pointers, sentinel values). LLVM's GEP without
        // `inbounds` is defined as pure integer arithmetic.
        let ptr_ty = self.cx.type_ptr();
        let p = self.cx.render_value(ptr);
        let idx0 = self.cx.render_value(indices[0]);
        let t = self.cx.render_type(ty);

        let base = CExpr::cast("uintptr_t", CExpr::var(&p));
        let scaled_idx0 = CExpr::binop(
            CExpr::cast("intptr_t", CExpr::paren(CExpr::var(&idx0))),
            CBinOp::Mul,
            CExpr::cast("intptr_t", CExpr::sizeof_ty(&t)),
        );

        if indices.len() == 1 {
            let sum = CExpr::binop(base, CBinOp::Add, scaled_idx0);
            self.new_temp_with_stmt(ptr_ty, CExpr::cast("void *", CExpr::paren(sum)))
        } else if indices.len() == 2 {
            let idx1 = self.cx.render_value(indices[1]);
            let sum = CExpr::binop(
                CExpr::binop(base, CBinOp::Add, scaled_idx0),
                CBinOp::Add,
                CExpr::cast("intptr_t", CExpr::paren(CExpr::var(&idx1))),
            );
            self.new_temp_with_stmt(ptr_ty, CExpr::cast("void *", CExpr::paren(sum)))
        } else {
            let sum = CExpr::binop(base, CBinOp::Add, scaled_idx0);
            self.new_temp_with_stmt(ptr_ty, CExpr::cast("void *", CExpr::paren(sum)))
        }
    }

    fn inbounds_gep(&mut self, ty: TypeRef, ptr: ValueRef, indices: &[ValueRef]) -> ValueRef {
        self.gep(ty, ptr, indices)
    }

    fn trunc(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        self.cast(val, dest_ty)
    }
    fn sext(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        self.cast(val, dest_ty)
    }
    fn fptoui_sat(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        // Saturating float-to-unsigned-int: clamp to [0, MAX], NaN -> 0
        let v = self.cx.render_value(val);
        let dt = self.cx.render_type(dest_ty);
        let types = self.cx.types.borrow();
        let max_val = match types.get(dest_ty) {
            CTypeKind::Int { bits, .. } if *bits < 128 => {
                let bits = *bits;
                drop(types);
                if bits == 64 {
                    "18446744073709551615.0".to_string()
                } else {
                    format!("{}.0", (1u128 << bits) - 1)
                }
            }
            CTypeKind::PtrWidth { .. } => {
                drop(types);
                // Use UINTPTR_MAX for portable saturation bound
                "(double)UINTPTR_MAX".to_string()
            }
            _ => {
                drop(types);
                return self.cast(val, dest_ty);
            }
        };
        // NaN check: x != x is true iff x is NaN (works for all float types)
        let isnan = CExpr::binop(CExpr::var(v.clone()), CBinOp::Ne, CExpr::var(v.clone()));
        let zero = CExpr::cast(dt.clone(), CExpr::lit("0"));
        let cast_v = CExpr::cast(dt.clone(), CExpr::var(v.clone()));
        let cast_max = CExpr::cast(dt, CExpr::lit(max_val.clone()));
        // v > max_val ? (dt)max_val : (dt)v
        let inner = CExpr::ternary(
            CExpr::binop(CExpr::var(v.clone()), CBinOp::Gt, CExpr::lit(max_val)),
            cast_max,
            cast_v,
        );
        // v < 0.0 ? (dt)0 : inner
        let mid = CExpr::ternary(
            CExpr::binop(CExpr::var(v), CBinOp::Lt, CExpr::lit("0.0")),
            zero.clone(),
            inner,
        );
        // isnan(v) ? (dt)0 : mid
        self.new_temp_with_stmt(dest_ty, CExpr::ternary(isnan, zero, mid))
    }
    fn fptosi_sat(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        // Saturating float-to-signed-int: clamp to [MIN, MAX], NaN -> 0
        let v = self.cx.render_value(val);
        let dt = self.cx.render_type(dest_ty);
        let types = self.cx.types.borrow();
        let (min_val, max_val) = match types.get(dest_ty) {
            CTypeKind::Int { bits, signed: true } if *bits <= 64 => {
                let bits = *bits;
                drop(types);
                let min = -(1i128 << (bits - 1));
                let max = (1i128 << (bits - 1)) - 1;
                (format!("{min}.0"), format!("{max}.0"))
            }
            CTypeKind::PtrWidth { signed: true } => {
                drop(types);
                // Use INTPTR_MIN/MAX for portable saturation bounds
                (
                    "(double)INTPTR_MIN".to_string(),
                    "(double)INTPTR_MAX".to_string(),
                )
            }
            _ => {
                drop(types);
                return self.cast(val, dest_ty);
            }
        };
        // NaN check: x != x is true iff x is NaN (works for all float types)
        let isnan = CExpr::binop(CExpr::var(v.clone()), CBinOp::Ne, CExpr::var(v.clone()));
        let zero = CExpr::cast(dt.clone(), CExpr::lit("0"));
        let cast_v = CExpr::cast(dt.clone(), CExpr::var(v.clone()));
        let cast_max = CExpr::cast(dt.clone(), CExpr::lit(max_val.clone()));
        // v > max_val ? (dt)max_val : (dt)v
        let inner = CExpr::ternary(
            CExpr::binop(CExpr::var(v.clone()), CBinOp::Gt, CExpr::lit(max_val)),
            cast_max,
            cast_v,
        );
        // v < min_val ? (dt)min_val : inner
        let cast_min = CExpr::cast(dt, CExpr::lit(min_val.clone()));
        let mid = CExpr::ternary(
            CExpr::binop(CExpr::var(v.clone()), CBinOp::Lt, CExpr::lit(min_val)),
            cast_min,
            inner,
        );
        // isnan(v) ? (dt)0 : mid
        self.new_temp_with_stmt(dest_ty, CExpr::ternary(isnan, zero, mid))
    }
    fn fptoui(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        // Float to unsigned int: cast to signed dest first, then reinterpret.
        // For values that fit in the signed range this is fine; for values
        // near the unsigned max, C's float->signed conversion is UB, but
        // Rust only emits fptoui_sat for those cases.  We cast through
        // the unsigned type so the C compiler sees the intended semantics.
        if let Some(unsigned_dest) = self.unsigned_type(dest_ty) {
            let v = self.cx.render_value(val);
            let ut = self.cx.render_type(unsigned_dest);
            let dt = self.cx.render_type(dest_ty);
            self.new_temp_with_stmt(dest_ty, CExpr::cast(dt, CExpr::cast(ut, CExpr::var(v))))
        } else {
            self.cast(val, dest_ty)
        }
    }
    fn fptosi(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        self.cast(val, dest_ty)
    }
    fn uitofp(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        // Unsigned int to float: cast source to unsigned first so that
        // the C compiler treats the value as unsigned (otherwise a large
        // unsigned value stored in a signed int would convert to a
        // negative float).
        let src_ty = self.cx.values.borrow().get_type(val);
        if let Some(unsigned_src) = self.unsigned_type(src_ty) {
            let v = self.cx.render_value(val);
            let us = self.cx.render_type(unsigned_src);
            let dt = self.cx.render_type(dest_ty);
            self.new_temp_with_stmt(dest_ty, CExpr::cast(dt, CExpr::cast(us, CExpr::var(v))))
        } else {
            self.cast(val, dest_ty)
        }
    }
    fn sitofp(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        self.cast(val, dest_ty)
    }
    fn fptrunc(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        self.cast(val, dest_ty)
    }
    fn fpext(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        self.cast(val, dest_ty)
    }
    fn ptrtoint(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        self.cast(val, dest_ty)
    }
    fn inttoptr(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        self.cast(val, dest_ty)
    }
    fn bitcast(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        // bitcast preserves bit pattern; use memcpy for type punning
        let src_ty = self.cx.values.borrow().get_type(val);
        if src_ty == dest_ty {
            return val;
        }

        // For pointer-to-pointer casts, a C cast is fine
        let src_is_ptr = matches!(self.cx.types.borrow().get(src_ty), CTypeKind::Ptr);
        let dst_is_ptr = matches!(self.cx.types.borrow().get(dest_ty), CTypeKind::Ptr);
        if src_is_ptr && dst_is_ptr {
            return self.cast(val, dest_ty);
        }

        // Same-width integer types (e.g. int64_t <-> uint64_t, intptr_t <-> uintptr_t):
        // a C cast preserves the bit pattern without needing memcpy.
        let both_int_same_width = {
            let types = self.cx.types.borrow();
            match (types.get(src_ty), types.get(dest_ty)) {
                (CTypeKind::Int { bits: a, .. }, CTypeKind::Int { bits: b, .. }) => *a == *b,
                (CTypeKind::PtrWidth { .. }, CTypeKind::PtrWidth { .. }) => true,
                (CTypeKind::PtrWidth { .. }, CTypeKind::Int { bits, .. })
                | (CTypeKind::Int { bits, .. }, CTypeKind::PtrWidth { .. }) => {
                    *bits == self.cx.tcx.data_layout.pointer_size().bits() as u32
                }
                _ => false,
            }
        };
        if both_int_same_width {
            return self.cast(val, dest_ty);
        }

        // Use memcpy to reinterpret bits (safe under strict aliasing).
        // Store val to a temp first so we always have an lvalue for &
        // (constants like 4ULL aren't lvalues).
        let result = self.cx.new_temp(dest_ty);
        let result_name = self.cx.render_value(result);
        let dt = self.cx.render_type(dest_ty);
        let v = self.cx.render_value(val);
        let st = self.cx.render_type(src_ty);
        let src_temp = self.cx.new_temp(src_ty);
        let src_name = self.cx.render_value(src_temp);
        {
            let mut module = self.cx.module.borrow_mut();
            if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                func.add_local_decl(format!("{dt} {result_name};"));
                func.add_local_decl(format!("{st} {src_name};"));
            }
        }
        self.emit(CStmt::assign(CExpr::var(&src_name), CExpr::var(v)));
        self.emit(CStmt::expr(CExpr::call(
            CExpr::var("memcpy"),
            vec![
                CExpr::addr_of(CExpr::var(&result_name)),
                CExpr::addr_of(CExpr::var(&src_name)),
                CExpr::sizeof_expr(CExpr::var(&result_name)),
            ],
        )));
        result
    }
    fn intcast(&mut self, val: ValueRef, dest_ty: TypeRef, is_signed: bool) -> ValueRef {
        if is_signed {
            // Sign extension: C promotes signed values with sign extension.
            // If the source type is unsigned (e.g. uintptr_t), cast to the
            // same-width signed type first so C sign-extends correctly.
            let src_ty = self.cx.values.borrow().get_type(val);
            let src_is_unsigned = {
                let types = self.cx.types.borrow();
                match types.get(src_ty) {
                    CTypeKind::Int { signed: false, .. }
                    | CTypeKind::PtrWidth { signed: false } => true,
                    _ => false,
                }
            };
            if src_is_unsigned {
                let signed_src = {
                    let types = self.cx.types.borrow();
                    match types.get(src_ty) {
                        CTypeKind::Int { bits, .. } => {
                            let bits = *bits;
                            drop(types);
                            self.cx.intern_type(CTypeKind::Int { bits, signed: true })
                        }
                        CTypeKind::PtrWidth { .. } => {
                            drop(types);
                            self.cx.intern_type(CTypeKind::PtrWidth { signed: true })
                        }
                        _ => {
                            drop(types);
                            src_ty
                        }
                    }
                };
                let v = self.cx.render_value(val);
                let st = self.cx.render_type(signed_src);
                let dt = self.cx.render_type(dest_ty);
                self.new_temp_with_stmt(
                    dest_ty,
                    CExpr::cast(dt, CExpr::cast(st, CExpr::var(v))),
                )
            } else {
                self.cast(val, dest_ty)
            }
        } else {
            // Zero extension: zext casts to unsigned first to prevent
            // sign extension when the source type is signed.
            self.zext(val, dest_ty)
        }
    }
    fn pointercast(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        self.cast(val, dest_ty)
    }

    fn icmp(&mut self, op: IntPredicate, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        let bool_ty = self.cx.intern_type(CTypeKind::Bool);
        let l = self.cx.render_value(lhs);
        let r = self.cx.render_value(rhs);

        let c_op = match op {
            IntPredicate::IntEQ => CBinOp::Eq,
            IntPredicate::IntNE => CBinOp::Ne,
            IntPredicate::IntUGT | IntPredicate::IntSGT => CBinOp::Gt,
            IntPredicate::IntUGE | IntPredicate::IntSGE => CBinOp::Ge,
            IntPredicate::IntULT | IntPredicate::IntSLT => CBinOp::Lt,
            IntPredicate::IntULE | IntPredicate::IntSLE => CBinOp::Le,
        };

        // In LLVM IR, integer constants are typeless -- signedness comes from
        // the comparison opcode. In C, signedness comes from the operand types.
        // We must explicitly cast both operands to the same type to prevent
        // C's implicit promotion rules (e.g. int32_t vs uint64_t literal)
        // from changing comparison semantics.
        let lhs_ty = self.cx.values.borrow().get_type(lhs);
        let rhs_ty = self.cx.values.borrow().get_type(rhs);
        let is_unsigned = matches!(
            op,
            IntPredicate::IntUGT
                | IntPredicate::IntUGE
                | IntPredicate::IntULT
                | IntPredicate::IntULE
        );
        let is_signed = matches!(
            op,
            IntPredicate::IntSGT
                | IntPredicate::IntSGE
                | IntPredicate::IntSLT
                | IntPredicate::IntSLE
        );

        // Check if either operand is a pointer-width type.
        let either_ptrwidth = {
            let types = self.cx.types.borrow();
            matches!(types.get(lhs_ty), CTypeKind::PtrWidth { .. })
                || matches!(types.get(rhs_ty), CTypeKind::PtrWidth { .. })
        };

        if either_ptrwidth {
            // When either operand is intptr_t/uintptr_t, use pointer-width
            // cast type to keep the comparison portable.
            let cast_ty = self.cx.intern_type(CTypeKind::PtrWidth {
                signed: !is_unsigned,
            });
            let ct = self.cx.render_type(cast_ty);
            return self.new_temp_with_stmt(
                bool_ty,
                CExpr::binop(
                    CExpr::cast(&ct, CExpr::var(&l)),
                    c_op,
                    CExpr::cast(ct, CExpr::var(&r)),
                ),
            );
        }

        // Get the bit width of a type (for choosing the wider type).
        let int_width = |ty: TypeRef| -> u32 {
            let types = self.cx.types.borrow();
            match types.get(ty) {
                CTypeKind::Int { bits, .. } => *bits,
                CTypeKind::Bool => 1,
                CTypeKind::Ptr => 64,
                _ => 0,
            }
        };
        // Use the wider of the two operand types to avoid truncation.
        let lhs_bits = int_width(lhs_ty);
        let rhs_bits = int_width(rhs_ty);
        let cast_bits = lhs_bits.max(rhs_bits);

        if is_unsigned {
            let cast_ty = self.cx.intern_type(CTypeKind::Int {
                bits: cast_bits,
                signed: false,
            });
            let ut = self.cx.render_type(cast_ty);
            return self.new_temp_with_stmt(
                bool_ty,
                CExpr::binop(
                    CExpr::cast(&ut, CExpr::var(&l)),
                    c_op,
                    CExpr::cast(ut, CExpr::var(&r)),
                ),
            );
        } else if is_signed || cast_bits > 0 {
            // For signed and equality comparisons, cast to a signed type
            // wide enough for both operands.
            let cast_ty = self.cx.intern_type(CTypeKind::Int {
                bits: cast_bits,
                signed: true,
            });
            let st = self.cx.render_type(cast_ty);
            return self.new_temp_with_stmt(
                bool_ty,
                CExpr::binop(
                    CExpr::cast(&st, CExpr::var(&l)),
                    c_op,
                    CExpr::cast(st, CExpr::var(&r)),
                ),
            );
        }
        self.new_temp_with_stmt(bool_ty, CExpr::binop(CExpr::var(l), c_op, CExpr::var(r)))
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        let bool_ty = self.cx.intern_type(CTypeKind::Bool);
        let l = self.cx.render_value(lhs);
        let r = self.cx.render_value(rhs);
        // NaN check: x != x is true iff x is NaN (works for all float types)
        let isnan_l = CExpr::binop(CExpr::var(l.clone()), CBinOp::Ne, CExpr::var(l.clone()));
        let isnan_r = CExpr::binop(CExpr::var(r.clone()), CBinOp::Ne, CExpr::var(r.clone()));
        let c_op = match op {
            RealPredicate::RealOEQ | RealPredicate::RealUEQ => CBinOp::Eq,
            RealPredicate::RealONE | RealPredicate::RealUNE => CBinOp::Ne,
            RealPredicate::RealOGT | RealPredicate::RealUGT => CBinOp::Gt,
            RealPredicate::RealOGE | RealPredicate::RealUGE => CBinOp::Ge,
            RealPredicate::RealOLT | RealPredicate::RealULT => CBinOp::Lt,
            RealPredicate::RealOLE | RealPredicate::RealULE => CBinOp::Le,
            RealPredicate::RealPredicateFalse => {
                return self.cx.const_bool(false);
            }
            RealPredicate::RealPredicateTrue => {
                return self.cx.const_bool(true);
            }
            RealPredicate::RealORD => {
                // Ordered: both are not NaN
                return self.new_temp_with_stmt(
                    bool_ty,
                    CExpr::binop(
                        CExpr::unary(CUnaryOp::LogNot, isnan_l),
                        CBinOp::LogAnd,
                        CExpr::unary(CUnaryOp::LogNot, isnan_r),
                    ),
                );
            }
            RealPredicate::RealUNO => {
                // Unordered: either is NaN
                return self
                    .new_temp_with_stmt(bool_ty, CExpr::binop(isnan_l, CBinOp::LogOr, isnan_r));
            }
        };
        self.new_temp_with_stmt(bool_ty, CExpr::binop(CExpr::var(l), c_op, CExpr::var(r)))
    }

    fn memcpy(
        &mut self,
        dst: ValueRef,
        _dst_align: Align,
        src: ValueRef,
        _src_align: Align,
        size: ValueRef,
        _flags: MemFlags,
        _tt: Option<FncTree>,
    ) {
        let d = self.cx.render_value(dst);
        let s = self.cx.render_value(src);
        let sz = self.cx.render_value(size);
        self.emit(CStmt::expr(CExpr::call(
            CExpr::var("memcpy"),
            vec![CExpr::var(d), CExpr::var(s), CExpr::var(sz)],
        )));
    }

    fn memmove(
        &mut self,
        dst: ValueRef,
        _dst_align: Align,
        src: ValueRef,
        _src_align: Align,
        size: ValueRef,
        _flags: MemFlags,
    ) {
        let d = self.cx.render_value(dst);
        let s = self.cx.render_value(src);
        let sz = self.cx.render_value(size);
        self.emit(CStmt::expr(CExpr::call(
            CExpr::var("memmove"),
            vec![CExpr::var(d), CExpr::var(s), CExpr::var(sz)],
        )));
    }

    fn memset(
        &mut self,
        ptr: ValueRef,
        fill_byte: ValueRef,
        size: ValueRef,
        _align: Align,
        _flags: MemFlags,
    ) {
        let p = self.cx.render_value(ptr);
        let b = self.cx.render_value(fill_byte);
        let sz = self.cx.render_value(size);
        self.emit(CStmt::expr(CExpr::call(
            CExpr::var("memset"),
            vec![CExpr::var(p), CExpr::var(b), CExpr::var(sz)],
        )));
    }

    fn select(&mut self, cond: ValueRef, then_val: ValueRef, else_val: ValueRef) -> ValueRef {
        let ty = self.cx.values.borrow().get_type(then_val);
        let c = self.cx.render_value(cond);
        let t = self.cx.render_value(then_val);
        let e = self.cx.render_value(else_val);
        self.new_temp_with_stmt(
            ty,
            CExpr::ternary(CExpr::var(c), CExpr::var(t), CExpr::var(e)),
        )
    }

    fn va_arg(&mut self, list: ValueRef, ty: TypeRef) -> ValueRef {
        let l = self.cx.render_value(list);
        let t = self.cx.render_type(ty);
        self.new_temp_with_stmt(
            ty,
            CExpr::call(CExpr::var("va_arg"), vec![CExpr::var(l), CExpr::raw(t)]),
        )
    }

    fn extract_element(&mut self, vec: ValueRef, idx: ValueRef) -> ValueRef {
        let ty = self.element_type(self.cx.values.borrow().get_type(vec));
        let v = self.cx.render_value(vec);
        let i = self.cx.render_value(idx);
        self.new_temp_with_stmt(ty, CExpr::index(CExpr::var(v), CExpr::var(i)))
    }

    fn vector_splat(&mut self, num_elts: usize, elt: ValueRef) -> ValueRef {
        let elem_ty = self.cx.values.borrow().get_type(elt);
        let vec_ty = self.cx.intern_type(CTypeKind::Vector {
            element: elem_ty,
            len: num_elts as u64,
        });
        let e = self.cx.render_value(elt);
        self.new_temp_with_stmt(
            vec_ty,
            CExpr::InitList((0..num_elts).map(|_| CExpr::var(e.clone())).collect()),
        )
    }

    fn extract_value(&mut self, agg_val: ValueRef, idx: u64) -> ValueRef {
        let agg_ty = self.cx.values.borrow().get_type(agg_val);
        let field_ty = {
            let types = self.cx.types.borrow();
            match types.get(agg_ty) {
                CTypeKind::Struct { fields, .. } => fields[idx as usize],
                _ => agg_ty, // fallback
            }
        };
        let v = self.cx.render_value(agg_val);
        self.new_temp_with_stmt(field_ty, CExpr::field(CExpr::var(v), format!("f{idx}")))
    }

    fn insert_value(&mut self, agg_val: ValueRef, elt: ValueRef, idx: u64) -> ValueRef {
        let agg_ty = self.cx.values.borrow().get_type(agg_val);
        let result = self.cx.new_temp(agg_ty);
        let result_name = self.cx.render_value(result);
        let e = self.cx.render_value(elt);
        let t = self.cx.render_type(agg_ty);

        {
            let mut module = self.cx.module.borrow_mut();
            if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                func.add_local_decl(format!("{t} {result_name};"));
            }
        }
        // Check if the source is undef/poison -- use memset instead of assignment
        let is_undef = matches!(
            self.cx.values.borrow().get(agg_val).kind,
            CValueKind::Undef | CValueKind::Poison
        );
        if is_undef {
            self.emit(CStmt::expr(CExpr::call(
                CExpr::var("memset"),
                vec![
                    CExpr::addr_of(CExpr::var(&result_name)),
                    CExpr::lit("0"),
                    CExpr::sizeof_expr(CExpr::var(&result_name)),
                ],
            )));
        } else {
            let agg = self.cx.render_value(agg_val);
            self.emit(CStmt::assign(CExpr::var(&result_name), CExpr::var(agg)));
        }
        self.emit(CStmt::assign(
            CExpr::field(CExpr::var(&result_name), format!("f{idx}")),
            CExpr::var(e),
        ));
        result
    }

    fn set_personality_fn(&mut self, _personality: ValueRef) {
        // No-op: C doesn't have personality functions
    }

    fn cleanup_landing_pad(&mut self, _pers_fn: ValueRef) -> (ValueRef, ValueRef) {
        // Return the exception pointer stored by the invoke handler's
        // setjmp catch path, plus a dummy selector (always 0 for Rust).
        let ptr_ty = self.cx.type_ptr();
        let exn = self
            .cx
            .intern_value(CValueKind::InlineExpr("__exn_ptr".to_string()), ptr_ty);
        (exn, self.cx.const_i32(0))
    }

    fn filter_landing_pad(&mut self, _pers_fn: ValueRef) {
        // No-op
    }

    fn resume(&mut self, exn0: ValueRef, _exn1: ValueRef) {
        // Propagate the exception to the next handler in the unwind chain
        // via longjmp.  If the chain is empty (no enclosing catch_unwind),
        // abort -- matching the standard Rust behavior for uncaught panics.
        let e = self.cx.render_value(exn0);
        self.emit(CStmt::raw(format!(
            "if (__rustc_unwind_chain) {{ \
                __rustc_unwind_chain->exception_ptr = (void *){e}; \
                __rustc_longjmp(__rustc_unwind_chain->buf, 1); \
            }} else {{ abort(); }}"
        )));
        let bb = self.current_bb;
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.set_terminated(bb);
        }
    }

    fn cleanup_pad(&mut self, _parent: Option<ValueRef>, _args: &[ValueRef]) -> CFunclet {
        CFunclet
    }

    fn cleanup_ret(&mut self, _funclet: &CFunclet, unwind: Option<BasicBlockId>) {
        if let Some(target) = unwind {
            let label = self.block_label(target);
            self.emit(CStmt::goto(label));
        } else {
            self.emit(CStmt::raw(
                "if (__rustc_unwind_chain) { \
                    __rustc_unwind_chain->exception_ptr = __exn_ptr; \
                    __rustc_longjmp(__rustc_unwind_chain->buf, 1); \
                } else { abort(); }"
                    .to_string(),
            ));
        }
        let bb = self.current_bb;
        let mut module = self.cx.module.borrow_mut();
        if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
            func.set_terminated(bb);
        }
    }

    fn catch_pad(&mut self, _parent: ValueRef, _args: &[ValueRef]) -> CFunclet {
        CFunclet
    }

    fn catch_switch(
        &mut self,
        _parent: Option<ValueRef>,
        _unwind: Option<BasicBlockId>,
        _handlers: &[BasicBlockId],
    ) -> ValueRef {
        self.cx.const_null(self.cx.type_ptr())
    }

    fn atomic_cmpxchg(
        &mut self,
        dst: ValueRef,
        cmp: ValueRef,
        src: ValueRef,
        _order: rustc_middle::ty::AtomicOrdering,
        _failure_order: rustc_middle::ty::AtomicOrdering,
        _weak: bool,
    ) -> (ValueRef, ValueRef) {
        let ty = self.cx.values.borrow().get_type(cmp);
        let bool_ty = self.cx.intern_type(CTypeKind::Bool);
        let d = self.cx.render_value(dst);
        let c = self.cx.render_value(cmp);
        let s = self.cx.render_value(src);
        let t = self.cx.render_type(ty);

        let old = self.new_temp_with_stmt(ty, CExpr::var(c));
        let old_name = self.cx.render_value(old);
        let success = self.new_temp_with_stmt(
            bool_ty,
            CExpr::call(
                CExpr::var("atomic_compare_exchange_strong"),
                vec![
                    CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d)),
                    CExpr::addr_of(CExpr::var(&old_name)),
                    CExpr::var(s),
                ],
            ),
        );
        (old, success)
    }

    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: ValueRef,
        src: ValueRef,
        _order: rustc_middle::ty::AtomicOrdering,
        _ret_ptr: bool,
    ) -> ValueRef {
        let ty = self.cx.values.borrow().get_type(src);
        let d = self.cx.render_value(dst);
        let s = self.cx.render_value(src);
        let t = self.cx.render_type(ty);

        match op {
            AtomicRmwBinOp::AtomicXchg => {
                let atomic_ptr = CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d));
                self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("atomic_exchange"),
                        vec![atomic_ptr, CExpr::var(&s)],
                    ),
                )
            }
            AtomicRmwBinOp::AtomicAdd => {
                let atomic_ptr = CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d));
                self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("atomic_fetch_add"),
                        vec![atomic_ptr, CExpr::var(&s)],
                    ),
                )
            }
            AtomicRmwBinOp::AtomicSub => {
                let atomic_ptr = CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d));
                self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("atomic_fetch_sub"),
                        vec![atomic_ptr, CExpr::var(&s)],
                    ),
                )
            }
            AtomicRmwBinOp::AtomicAnd => {
                let atomic_ptr = CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d));
                self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("atomic_fetch_and"),
                        vec![atomic_ptr, CExpr::var(&s)],
                    ),
                )
            }
            AtomicRmwBinOp::AtomicOr => {
                let atomic_ptr = CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d));
                self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("atomic_fetch_or"),
                        vec![atomic_ptr, CExpr::var(&s)],
                    ),
                )
            }
            AtomicRmwBinOp::AtomicXor => {
                let atomic_ptr = CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d));
                self.new_temp_with_stmt(
                    ty,
                    CExpr::call(
                        CExpr::var("atomic_fetch_xor"),
                        vec![atomic_ptr, CExpr::var(&s)],
                    ),
                )
            }
            AtomicRmwBinOp::AtomicNand => {
                // NAND = ~(old & val), implemented via CAS loop
                let old = self.cx.new_temp(ty);
                let old_name = self.cx.render_value(old);
                let desired = self.cx.new_temp(ty);
                let desired_name = self.cx.render_value(desired);
                {
                    let mut module = self.cx.module.borrow_mut();
                    if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                        func.add_local_decl(format!("{t} {old_name};"));
                        func.add_local_decl(format!("{t} {desired_name};"));
                    }
                }
                self.emit(CStmt::assign(
                    CExpr::var(&old_name),
                    CExpr::call(
                        CExpr::var("atomic_load"),
                        vec![CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d))],
                    ),
                ));
                self.emit(CStmt::DoWhile {
                    body: vec![CStmt::assign(
                        CExpr::var(&desired_name),
                        CExpr::unary(
                            CUnaryOp::BitNot,
                            CExpr::paren(CExpr::binop(
                                CExpr::var(&old_name),
                                CBinOp::BitAnd,
                                CExpr::var(&s),
                            )),
                        ),
                    )],
                    cond: CExpr::unary(
                        CUnaryOp::LogNot,
                        CExpr::call(
                            CExpr::var("atomic_compare_exchange_weak"),
                            vec![
                                CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d)),
                                CExpr::addr_of(CExpr::var(&old_name)),
                                CExpr::var(&desired_name),
                            ],
                        ),
                    ),
                });
                old
            }
            AtomicRmwBinOp::AtomicMin
            | AtomicRmwBinOp::AtomicMax
            | AtomicRmwBinOp::AtomicUMin
            | AtomicRmwBinOp::AtomicUMax => {
                // Min/Max via CAS loop
                let old = self.cx.new_temp(ty);
                let old_name = self.cx.render_value(old);
                let desired = self.cx.new_temp(ty);
                let desired_name = self.cx.render_value(desired);
                {
                    let mut module = self.cx.module.borrow_mut();
                    if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                        func.add_local_decl(format!("{t} {old_name};"));
                        func.add_local_decl(format!("{t} {desired_name};"));
                    }
                }
                let cmp = match op {
                    AtomicRmwBinOp::AtomicMin => format!("{old_name} < {s} ? {old_name} : {s}"),
                    AtomicRmwBinOp::AtomicMax => format!("{old_name} > {s} ? {old_name} : {s}"),
                    AtomicRmwBinOp::AtomicUMin => {
                        if let Some(ut) = self.unsigned_type(ty) {
                            let ut_str = self.cx.render_type(ut);
                            format!(
                                "({t})(({ut_str}){old_name} < ({ut_str}){s} ? ({ut_str}){old_name} : ({ut_str}){s})"
                            )
                        } else {
                            format!("{old_name} < {s} ? {old_name} : {s}")
                        }
                    }
                    AtomicRmwBinOp::AtomicUMax => {
                        if let Some(ut) = self.unsigned_type(ty) {
                            let ut_str = self.cx.render_type(ut);
                            format!(
                                "({t})(({ut_str}){old_name} > ({ut_str}){s} ? ({ut_str}){old_name} : ({ut_str}){s})"
                            )
                        } else {
                            format!("{old_name} > {s} ? {old_name} : {s}")
                        }
                    }
                    _ => unreachable!(),
                };
                self.emit(CStmt::assign(
                    CExpr::var(&old_name),
                    CExpr::call(
                        CExpr::var("atomic_load"),
                        vec![CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d))],
                    ),
                ));
                self.emit(CStmt::DoWhile {
                    body: vec![CStmt::assign(CExpr::var(&desired_name), CExpr::raw(cmp))],
                    cond: CExpr::unary(
                        CUnaryOp::LogNot,
                        CExpr::call(
                            CExpr::var("atomic_compare_exchange_weak"),
                            vec![
                                CExpr::cast(format!("_Atomic({t}) *"), CExpr::var(&d)),
                                CExpr::addr_of(CExpr::var(&old_name)),
                                CExpr::var(&desired_name),
                            ],
                        ),
                    ),
                });
                old
            }
        }
    }

    fn atomic_fence(
        &mut self,
        order: rustc_middle::ty::AtomicOrdering,
        scope: SynchronizationScope,
    ) {
        let c_order = Self::atomic_ordering_to_c(order);
        let fence_fn = match scope {
            SynchronizationScope::SingleThread => "atomic_signal_fence",
            SynchronizationScope::CrossThread => "atomic_thread_fence",
        };
        self.emit(CStmt::expr(CExpr::call(
            CExpr::var(fence_fn),
            vec![CExpr::var(c_order)],
        )));
    }

    fn set_invariant_load(&mut self, _load: ValueRef) {
        // No-op
    }

    fn lifetime_start(&mut self, _ptr: ValueRef, _size: Size) {
        // No-op
    }

    fn lifetime_end(&mut self, _ptr: ValueRef, _size: Size) {
        // No-op
    }

    fn call(
        &mut self,
        llty: TypeRef,
        _caller_attrs: Option<&CodegenFnAttrs>,
        _fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        fn_val: ValueRef,
        args: &[ValueRef],
        _funclet: Option<&CFunclet>,
        _callee_instance: Option<Instance<'tcx>>,
    ) -> ValueRef {
        // codegen_ssa passes the sret output pointer as the first argument
        // for indirect returns.  Since our C functions return the struct by
        // value (the C compiler handles sret natively for ALL ABIs), we
        // strip the sret pointer and capture the return value into it.
        let is_indirect_ret = _fn_abi.map(|abi| abi.ret.is_indirect()).unwrap_or(false);
        let (sret_ptr, actual_args) = if is_indirect_ret && !args.is_empty() {
            (Some(args[0]), &args[1..])
        } else {
            (None, args)
        };

        let args_exprs: Vec<_> = actual_args
            .iter()
            .map(|a| CExpr::var(self.cx.render_value(*a)))
            .collect();

        // Determine return type and function name
        let (ret_ty, f) = {
            let values = self.cx.values.borrow();
            if let Some(sig) = values.get_fn_sig(fn_val) {
                let types = self.cx.types.borrow();
                let ret = match types.get(sig) {
                    CTypeKind::Function { ret, .. } => *ret,
                    _ => self.cx.intern_type(CTypeKind::Void),
                };
                (ret, self.cx.render_value(fn_val))
            } else {
                let f_raw = self.cx.render_value(fn_val);
                let types = self.cx.types.borrow();
                let ret = match types.get(llty) {
                    CTypeKind::Function { ret, .. } => *ret,
                    _ => self.cx.intern_type(CTypeKind::Void),
                };
                drop(types);
                let fn_ptr_ty = self.cx.render_type(llty);
                (ret, format!("(({fn_ptr_ty}){f_raw})"))
            }
        };

        {
            let is_void = matches!(self.cx.types.borrow().get(ret_ty), CTypeKind::Void);
            if is_void {
                self.emit(CStmt::expr(CExpr::call(CExpr::var(&f), args_exprs)));
                // For indirect return, codegen_ssa never uses the call
                // result (it reads from the output pointer instead), but
                // it does check the type.  Return an undef of the expected
                // layout type so that type_kind() sees Integer/Struct
                // instead of Pointer.
                if is_indirect_ret {
                    let real_ret_ty = _fn_abi
                        .map(|abi| crate::type_of::layout_to_c_type(self.cx, abi.ret.layout))
                        .unwrap_or_else(|| self.cx.type_ptr());
                    self.cx.const_undef(real_ret_ty)
                } else {
                    self.cx.const_null(self.cx.type_ptr())
                }
            } else {
                let result =
                    self.new_temp_with_stmt(ret_ty, CExpr::call(CExpr::var(&f), args_exprs));
                // For C-ABI indirect return, copy the return value into
                // the sret buffer and return the sret pointer.
                if let Some(sret) = sret_ptr {
                    let r = self.cx.render_value(result);
                    let s = self.cx.render_value(sret);
                    let t = self.cx.render_type(ret_ty);
                    self.emit(CStmt::assign(
                        CExpr::deref(format!("{t} *"), CExpr::var(s)),
                        CExpr::var(r),
                    ));
                    return sret;
                }
                // If the ABI returns a boolean as int8_t, convert back
                // to _Bool so that `not` (and other boolean ops) see a
                // _Bool value and use logical `!` instead of bitwise `~`.
                if let Some(abi) = _fn_abi {
                    if let rustc_abi::BackendRepr::Scalar(s) = abi.ret.layout.backend_repr {
                        if s.is_bool() {
                            let bool_ty = self.cx.intern_type(CTypeKind::Bool);
                            let v = self.cx.render_value(result);
                            return self.new_temp_with_stmt(
                                bool_ty,
                                CExpr::cast("_Bool".to_string(), CExpr::paren(CExpr::var(v))),
                            );
                        }
                    }
                }
                result
            }
        }
    }

    fn tail_call(
        &mut self,
        llty: TypeRef,
        caller_attrs: Option<&CodegenFnAttrs>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        llfn: ValueRef,
        args: &[ValueRef],
        funclet: Option<&CFunclet>,
        callee_instance: Option<Instance<'tcx>>,
    ) {
        let result = self.call(
            llty,
            caller_attrs,
            Some(fn_abi),
            llfn,
            args,
            funclet,
            callee_instance,
        );
        // Tail call: return the result immediately.
        // If the current function uses indirect return (sret), call()
        // already stored the callee's result into _retbuf via the sret
        // pointer.  ret_void() emits `return _retbuf;` which is correct.
        let has_retbuf = {
            let module = self.cx.module.borrow();
            module
                .open_functions
                .get(&self.current_fn)
                .and_then(|f| f.retbuf_name.clone())
                .is_some()
        };
        if has_retbuf {
            self.ret_void();
        } else {
            let ret_ty = {
                let values = self.cx.values.borrow();
                if let Some(sig) = values.get_fn_sig(llfn) {
                    let types = self.cx.types.borrow();
                    match types.get(sig) {
                        CTypeKind::Function { ret, .. } => *ret,
                        _ => self.cx.intern_type(CTypeKind::Void),
                    }
                } else {
                    self.cx.intern_type(CTypeKind::Void)
                }
            };
            let is_void = matches!(self.cx.types.borrow().get(ret_ty), CTypeKind::Void);
            if is_void {
                self.ret_void();
            } else {
                self.ret(result);
            }
        }
    }

    fn zext(&mut self, val: ValueRef, dest_ty: TypeRef) -> ValueRef {
        // Zero extension: cast source to unsigned first to prevent sign extension
        let src_ty = self.cx.values.borrow().get_type(val);
        if let Some(unsigned_src) = self.unsigned_type(src_ty) {
            let v = self.cx.render_value(val);
            let us = self.cx.render_type(unsigned_src);
            let dt = self.cx.render_type(dest_ty);
            self.new_temp_with_stmt(dest_ty, CExpr::cast(dt, CExpr::cast(us, CExpr::var(v))))
        } else {
            self.cast(val, dest_ty)
        }
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, _llret: ValueRef) {
        // No-op
    }
}

// --- ArgAbiBuilderMethods ---

impl<'a, 'tcx> ArgAbiBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, ValueRef>,
    ) {
        match arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Pair(_, _) => {
                let a = self.get_param(*idx);
                *idx += 1;
                let b = self.get_param(*idx);
                *idx += 1;
                self.store(a, dst.val.llval, dst.val.align);
                // Compute offset like LLVM: a.size aligned to b.align
                let (sa, sb) = match dst.layout.backend_repr {
                    rustc_abi::BackendRepr::ScalarPair(sa, sb) => (sa, sb),
                    _ => panic!("store_fn_arg Pair on non-ScalarPair"),
                };
                let b_offset = sa.size(self).align_to(sb.align(self).abi);
                let b_ptr = if b_offset.bytes() > 0 {
                    self.ptradd(dst.val.llval, self.const_usize(b_offset.bytes()))
                } else {
                    dst.val.llval
                };
                self.store(b, b_ptr, dst.val.align);
            }
            PassMode::Indirect {
                attrs: _,
                meta_attrs: _,
                on_stack: _,
            } => {
                let val = self.get_param(*idx);
                *idx += 1;
                let size = self.const_usize(dst.layout.size.bytes());
                self.memcpy(
                    dst.val.llval,
                    dst.val.align,
                    val,
                    dst.val.align,
                    size,
                    MemFlags::empty(),
                    None,
                );
            }
            PassMode::Cast {
                ref cast,
                pad_i32: _,
            } => {
                let val = self.get_param(*idx);
                *idx += 1;
                // The cast type may be larger than the Rust layout (rounded
                // up to register size). Store to a scratch buffer first,
                // then memcpy only the layout-sized bytes to the destination.
                let cast = cast.clone();
                let scratch_size = cast.size(self);
                let scratch_align = cast.align(self);
                let copy_bytes =
                    cmp::min(cast.unaligned_size(self).bytes(), dst.layout.size.bytes());
                let scratch = self.alloca(scratch_size, scratch_align);
                rustc_codegen_ssa::mir::store_cast(self, &cast, val, scratch, scratch_align);
                let size_val = self.const_usize(copy_bytes);
                self.memcpy(
                    dst.val.llval,
                    dst.val.align,
                    scratch,
                    scratch_align,
                    size_val,
                    MemFlags::empty(),
                    None,
                );
            }
            _ => {
                let val = self.get_param(*idx);
                *idx += 1;
                self.store(val, dst.val.llval, dst.val.align);
            }
        }
    }

    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: ValueRef,
        dst: PlaceRef<'tcx, ValueRef>,
    ) {
        match &arg_abi.mode {
            PassMode::Cast { cast, pad_i32: _ } => {
                // The cast type may be larger than the Rust layout (rounded
                // up to register size). Store to a scratch buffer first,
                // then memcpy only the layout-sized bytes to the destination.
                let scratch_size = cast.size(self);
                let scratch_align = cast.align(self);
                let copy_bytes = cmp::min(
                    cast.unaligned_size(self).bytes(),
                    arg_abi.layout.size.bytes(),
                );
                let scratch = self.alloca(scratch_size, scratch_align);
                rustc_codegen_ssa::mir::store_cast(self, cast, val, scratch, scratch_align);
                let size_val = self.const_usize(copy_bytes);
                self.memcpy(
                    dst.val.llval,
                    dst.val.align,
                    scratch,
                    scratch_align,
                    size_val,
                    MemFlags::empty(),
                    None,
                );
            }
            PassMode::Indirect {
                attrs,
                meta_attrs: None,
                ..
            } => {
                let align = attrs.pointee_align.unwrap_or(arg_abi.layout.align.abi);
                OperandValue::Ref(PlaceValue::new_sized(val, align)).store(self, dst);
            }
            _ => {
                self.store(val, dst.val.llval, dst.val.align);
            }
        }
    }
}

// --- AbiBuilderMethods ---

impl<'a, 'tcx> AbiBuilderMethods for Builder<'a, 'tcx> {
    fn get_param(&mut self, index: usize) -> ValueRef {
        let has_indirect_ret = {
            let module = self.cx.module.borrow();
            module
                .open_functions
                .get(&self.current_fn)
                .map_or(false, |f| f.has_indirect_ret)
        };

        if has_indirect_ret && index == 0 {
            // codegen_ssa wants a pointer to write the return value to.
            // Allocate a local buffer; ret_void() will emit
            // `return _retbuf;` so the C compiler copies the value to
            // the caller via the platform sret convention.
            let existing = {
                let module = self.cx.module.borrow();
                module
                    .open_functions
                    .get(&self.current_fn)
                    .and_then(|f| f.retbuf_name.clone())
            };
            if let Some(arr_name) = existing {
                return self.cx.intern_value(
                    CValueKind::InlineExpr(format!("(void *)&{arr_name}")),
                    self.cx.type_ptr(),
                );
            }
            let ret_ty = {
                let module = self.cx.module.borrow();
                module
                    .open_functions
                    .get(&self.current_fn)
                    .and_then(|f| f.ret_data_type)
                    .unwrap_or(self.cx.type_ptr())
            };
            let t = self.cx.render_type_decl(ret_ty, "_retbuf");
            {
                let mut module = self.cx.module.borrow_mut();
                if let Some(func) = module.open_functions.get_mut(&self.current_fn) {
                    func.add_local_decl(format!("_Alignas(16) {t};"));
                    func.retbuf_name = Some("_retbuf".to_string());
                }
            }
            self.cx.intern_value(
                CValueKind::InlineExpr("(void *)&_retbuf".to_string()),
                self.cx.type_ptr(),
            )
        } else {
            // codegen_ssa's llarg_idx starts at 1 when has_indirect_ret
            // (to skip the sret slot).  The C function has NO explicit
            // sret parameter, so subtract 1 to get the correct C
            // parameter index.
            let c_index = if has_indirect_ret { index - 1 } else { index };
            let param_ty = {
                let module = self.cx.module.borrow();
                let func = module.open_functions.get(&self.current_fn);
                func.and_then(|f| f.params.get(c_index).map(|(ty, _)| *ty))
                    .unwrap_or(self.cx.type_ptr())
            };
            self.cx
                .intern_value(CValueKind::Param { index: c_index }, param_ty)
        }
    }
}

// --- StaticBuilderMethods ---

impl<'a, 'tcx> StaticBuilderMethods for Builder<'a, 'tcx> {
    fn get_static(&mut self, def_id: rustc_hir::def_id::DefId) -> ValueRef {
        if let Some(&v) = self.cx.statics_cache.borrow().get(&def_id) {
            return v;
        }
        let instance = Instance::mono(self.cx.tcx, def_id);
        let sym = self.cx.tcx.symbol_name(instance).name.to_string();
        let c_name = CodegenCx::sanitize_name(&sym);
        let ptr_ty = self.cx.type_ptr();

        // Emit extern declaration for external statics
        self.cx.emit_extern_static_decl(&c_name, &sym, def_id);

        let val = if self.cx.is_extern_weak(def_id) {
            // extern_weak: declared as a weak function in C.
            // codegen_ssa expects get_static to return a pointer to memory
            // holding the function pointer. Create a local that stores the
            // function pointer so the caller can load from it.
            self.cx.intern_value(
                CValueKind::InlineExpr(format!("(void *)&(void *){{{c_name}}}")),
                ptr_ty,
            )
        } else {
            self.cx.intern_value(
                CValueKind::Global {
                    name: format!("&{c_name}"),
                },
                ptr_ty,
            )
        };
        self.cx.statics_cache.borrow_mut().insert(def_id, val);
        val
    }
}
