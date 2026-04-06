/// Debug info and inline assembly stubs for the C codegen backend.
///
/// The C backend does not emit DWARF or other debug info; all methods
/// are no-ops. Inline assembly is similarly unsupported and emitted
/// as a comment.
use rustc_codegen_ssa::traits::*;
use rustc_middle::mir;
use rustc_middle::ty::{ExistentialTraitRef, Instance, Ty};
use rustc_span::{SourceFile, Span, Symbol};
use rustc_target::callconv::FnAbi;

use crate::context::{CodegenCx, DebugLoc, DebugScope, DebugVar};
use crate::values::ValueRef;

// --- DebugInfoCodegenMethods (stubs) ---

impl<'tcx> DebugInfoCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn create_vtable_debuginfo(
        &self,
        _ty: Ty<'tcx>,
        _trait_ref: Option<ExistentialTraitRef<'tcx>>,
        _vtable: ValueRef,
    ) {
    }

    fn create_function_debug_context(
        &self,
        _instance: Instance<'tcx>,
        _fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        _llfn: ValueRef,
        _mir: &mir::Body<'tcx>,
    ) -> Option<rustc_codegen_ssa::mir::debuginfo::FunctionDebugContext<'tcx, DebugScope, DebugLoc>>
    {
        None
    }

    fn dbg_scope_fn(
        &self,
        _instance: Instance<'tcx>,
        _fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        _maybe_definition_llfn: Option<ValueRef>,
    ) -> DebugScope {
        DebugScope
    }

    fn dbg_loc(&self, _scope: DebugScope, _inlined_at: Option<DebugLoc>, _span: Span) -> DebugLoc {
        DebugLoc
    }

    fn extend_scope_to_file(&self, _scope_metadata: DebugScope, _file: &SourceFile) -> DebugScope {
        DebugScope
    }

    fn debuginfo_finalize(&self) {}

    fn create_dbg_var(
        &self,
        _variable_name: Symbol,
        _variable_type: Ty<'tcx>,
        _scope_metadata: DebugScope,
        _variable_kind: rustc_codegen_ssa::mir::debuginfo::VariableKind,
        _span: Span,
    ) -> DebugVar {
        DebugVar
    }
}

// --- AsmCodegenMethods (stubs) ---

impl<'tcx> AsmCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn codegen_global_asm(
        &mut self,
        _template: &[rustc_ast::InlineAsmTemplatePiece],
        _operands: &[GlobalAsmOperandRef<'tcx>],
        _options: rustc_ast::InlineAsmOptions,
        _line_spans: &[Span],
    ) {
        self.tcx
            .dcx()
            .warn("global_asm! is not supported by the C backend");
    }

    fn mangled_name(&self, instance: Instance<'tcx>) -> String {
        self.tcx.symbol_name(instance).name.to_string()
    }
}
