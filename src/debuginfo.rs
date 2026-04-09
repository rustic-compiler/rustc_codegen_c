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
        // Build the assembly text, substituting operand placeholders.
        // SymFn operands become the mangled symbol name; SymStatic
        // become the static's symbol name.  If any operand type
        // cannot be handled, fall back to the warning.
        let mut operand_strs: Vec<Option<String>> = Vec::new();
        for op in _operands {
            match op {
                GlobalAsmOperandRef::SymFn { instance } => {
                    operand_strs.push(Some(self.tcx.symbol_name(*instance).name.to_string()));
                }
                GlobalAsmOperandRef::SymStatic { def_id } => {
                    let instance = Instance::mono(self.tcx, *def_id);
                    operand_strs.push(Some(self.tcx.symbol_name(instance).name.to_string()));
                }
                _ => {
                    operand_strs.push(None);
                }
            }
        }

        let mut asm_text = String::new();
        let mut ok = true;
        for piece in _template {
            match piece {
                rustc_ast::InlineAsmTemplatePiece::String(s) => {
                    asm_text.push_str(s);
                }
                rustc_ast::InlineAsmTemplatePiece::Placeholder { operand_idx, .. } => {
                    if let Some(Some(name)) = operand_strs.get(*operand_idx) {
                        asm_text.push_str(name);
                    } else {
                        ok = false;
                        break;
                    }
                }
            }
        }

        if ok && !asm_text.trim().is_empty() {
            // On x86/x86_64, Rust asm defaults to Intel syntax.
            // GCC's __asm__() uses AT&T by default, so wrap with
            // .intel_syntax/.att_syntax unless ATT_SYNTAX is requested.
            let is_x86 = matches!(
                self.tcx.sess.target.arch,
                rustc_target::spec::Arch::X86 | rustc_target::spec::Arch::X86_64
            );
            let use_intel = is_x86 && !_options.contains(rustc_ast::InlineAsmOptions::ATT_SYNTAX);
            // Convert Rust-style `//` comments to GAS `#` comments.
            // GAS doesn't recognize `//` as a comment delimiter.
            let asm_text = asm_text
                .lines()
                .map(|line| {
                    if let Some(idx) = line.find("//") {
                        format!("{}#{}", &line[..idx], &line[idx + 2..])
                    } else {
                        line.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

            let mut full_asm = String::new();
            if use_intel {
                full_asm.push_str(".intel_syntax noprefix\n");
            }
            full_asm.push_str(&asm_text);
            if use_intel {
                full_asm.push_str("\n.att_syntax\n");
            }

            // Escape for C string literal and emit as __asm__()
            let escaped = full_asm
                .lines()
                .map(|line| line.replace('\\', "\\\\").replace('"', "\\\""))
                .collect::<Vec<_>>()
                .join("\\n\"\n\"");
            self.module
                .borrow_mut()
                .function_defs
                .push(format!("__asm__(\n\"{escaped}\\n\"\n);\n"));
            return;
        }

        self.tcx
            .dcx()
            .warn("global_asm! is not supported by the C backend");
    }

    fn mangled_name(&self, instance: Instance<'tcx>) -> String {
        self.tcx.symbol_name(instance).name.to_string()
    }
}
