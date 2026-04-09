/// The codegen context: holds all state needed during code generation
/// of a single codegen unit.
use std::cell::RefCell;

use rustc_codegen_ssa::traits::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::mono::{CodegenUnit, Visibility};
use rustc_middle::ty::layout::{
    FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTypingEnv, LayoutError, LayoutOfHelpers,
};
use rustc_middle::ty::{self, ExistentialTraitRef, Instance, Ty, TyCtxt};
use rustc_session::Session;
use rustc_span::Span;

use crate::module::{BasicBlockId, CModule};
use crate::types::{CTypeKind, TypeRef};
use crate::values::{CValueKind, ValueRef};

/// The codegen context for a single codegen unit.
pub(crate) struct CodegenCx<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub codegen_unit: &'tcx CodegenUnit<'tcx>,
    /// Type store (separate RefCell to avoid borrow conflicts).
    pub types: RefCell<crate::types::TypeStore>,
    /// Value store (separate RefCell to avoid borrow conflicts).
    pub values: RefCell<crate::values::ValueStore>,
    /// The C module being constructed (functions, declarations, etc.).
    pub module: RefCell<CModule>,
    /// Cache: Instance -> function ValueRef.
    pub instances: RefCell<FxHashMap<Instance<'tcx>, ValueRef>>,
    /// Cache: vtable keys -> ValueRef.
    pub vtables_cache:
        RefCell<FxHashMap<(Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>), ValueRef>>,
    /// Cache: DefId -> static global ValueRef.
    pub statics_cache: RefCell<FxHashMap<DefId, ValueRef>>,
    /// Personality function for exception handling.
    eh_personality: RefCell<Option<ValueRef>>,
}

impl<'tcx> CodegenCx<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, codegen_unit: &'tcx CodegenUnit<'tcx>, name: &str) -> Self {
        let mut types = crate::types::TypeStore::new();
        types.intern(CTypeKind::Void);
        Self {
            tcx,
            codegen_unit,
            types: RefCell::new(types),
            values: RefCell::new(crate::values::ValueStore::new()),
            module: RefCell::new(CModule::new(name.to_string())),
            instances: RefCell::new(FxHashMap::default()),
            vtables_cache: RefCell::new(FxHashMap::default()),
            statics_cache: RefCell::new(FxHashMap::default()),
            eh_personality: RefCell::new(None),
        }
    }

    /// Intern a type into the type store.
    pub fn type_usize(&self) -> TypeRef {
        self.intern_type(CTypeKind::PtrWidth { signed: false })
    }

    pub fn intern_type(&self, kind: CTypeKind) -> TypeRef {
        self.types.borrow_mut().intern(kind)
    }

    /// Allocate a value in the value store.
    pub fn intern_value(&self, kind: CValueKind, ty: TypeRef) -> ValueRef {
        self.values.borrow_mut().alloc(kind, ty)
    }

    /// Create a new temporary value.
    pub fn new_temp(&self, ty: TypeRef) -> ValueRef {
        self.values.borrow_mut().next_temp(ty)
    }

    /// Render a value to its C expression string.
    pub fn render_value(&self, v: ValueRef) -> String {
        self.values.borrow().render(v)
    }

    /// Render a type to its C type string.
    pub fn render_type(&self, ty: TypeRef) -> String {
        self.types.borrow().render(ty)
    }

    /// Render a type declaration.
    pub fn render_type_decl(&self, ty: TypeRef, name: &str) -> String {
        self.types.borrow().render_decl(ty, name)
    }

    /// Sanitize a symbol name for use as a C identifier.
    pub fn sanitize_name(name: &str) -> String {
        name.chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect()
    }

    /// Generate a GCC `__asm__("original")` label suffix if the original
    /// symbol name differs from the sanitized C identifier.  This ensures
    /// the object file contains the exact Rust-mangled symbol name that
    /// the linker version script expects.
    pub fn asm_label(original: &str, sanitized: &str) -> String {
        if original == sanitized {
            String::new()
        } else {
            format!(" __asm__(\"{}\")", original)
        }
    }

    /// Check if a static has extern_weak linkage.
    pub fn is_extern_weak(&self, def_id: DefId) -> bool {
        let attrs = self.tcx.codegen_fn_attrs(def_id);
        matches!(
            attrs.import_linkage,
            Some(rustc_hir::attrs::Linkage::ExternalWeak)
        )
    }

    /// Generate extern declaration for a static symbol and add it to the module.
    /// `original_name` is the unsanitized Rust symbol name for `__asm__` labels.
    /// For cross-CGU statics, uses `extern uint8_t NAME[]`.
    /// For same-CGU statics, pre-evaluates the allocation to determine the type:
    /// - No relocations: `extern uint8_t NAME[]` (compatible with uint8_t[N] definition)
    /// - Has relocations: defines named struct and uses `extern struct _gs_NAME NAME;`
    pub fn emit_extern_static_decl(&self, c_name: &str, original_name: &str, def_id: DefId) {
        let is_weak = self.is_extern_weak(def_id);
        let tls = if self.tcx.is_thread_local_static(def_id) {
            "_Thread_local "
        } else {
            ""
        };
        let asm = Self::asm_label(original_name, c_name);
        if is_weak {
            self.module.borrow_mut().add_global_decl(
                c_name,
                format!("#pragma weak {c_name}\nvoid {c_name}(void){asm};"),
            );
            return;
        }

        // Check if this is a same-CGU static that will have relocations.
        // Guard against foreign items (extern statics with no body):
        // eval_static_initializer would ICE on those.
        let is_same_cgu = self.module.borrow().declared_globals.contains(c_name);
        let has_relocs = is_same_cgu
            && !self.tcx.is_foreign_item(def_id)
            && self
                .tcx
                .eval_static_initializer(def_id)
                .is_ok_and(|alloc| !alloc.inner().provenance().ptrs().is_empty());

        if has_relocs {
            // Same-CGU static with relocations: define the struct type and
            // use it for the forward declaration. This must match what
            // codegen_static will produce.
            self.emit_static_struct_fwd_decl(c_name, original_name, def_id, &tls);
        } else {
            self.module
                .borrow_mut()
                .add_global_decl(c_name, format!("extern {tls}uint8_t {c_name}[]{asm};"));
        }
    }

    /// Emit a named struct type definition and forward variable declaration
    /// for a static with relocations. Used by both emit_extern_static_decl
    /// and codegen_static.
    pub(crate) fn emit_static_struct_fwd_decl(
        &self,
        c_name: &str,
        original_name: &str,
        def_id: DefId,
        tls: &str,
    ) {
        let pointer_size = self.tcx.data_layout.pointer_size().bytes() as usize;
        let alloc = match self.tcx.eval_static_initializer(def_id) {
            Ok(a) => a,
            Err(_) => return,
        };
        let init = alloc.inner();
        let provenance = init.provenance();
        let alloc_len = init.len();

        // Build struct field types
        let mut fields = Vec::new();
        let mut pos = 0usize;
        let mut field_idx = 0usize;

        for &(reloc_offset, _) in provenance.ptrs().iter() {
            let reloc_off = reloc_offset.bytes() as usize;
            if reloc_off > pos {
                let pad_len = reloc_off - pos;
                fields.push(format!("unsigned char f{field_idx}[{pad_len}]"));
                field_idx += 1;
            }
            fields.push(format!("void *f{field_idx}"));
            field_idx += 1;
            pos = reloc_off + pointer_size;
        }
        if pos < alloc_len {
            let trail_len = alloc_len - pos;
            fields.push(format!("unsigned char f{field_idx}[{trail_len}]"));
        }

        let struct_name = format!("_gs_{c_name}");
        let fields_str = fields
            .iter()
            .map(|f| format!("{f};"))
            .collect::<Vec<_>>()
            .join(" ");

        let mut module = self.module.borrow_mut();
        // Only add if not already added (e.g., by codegen_static)
        if module.declared_extern_globals.contains(c_name) {
            return;
        }
        module.struct_defs.push(format!(
            "#pragma pack(push, 1)\nstruct {struct_name} {{ {fields_str} }};\n#pragma pack(pop)"
        ));
        let asm = Self::asm_label(original_name, c_name);
        module.add_global_decl(
            c_name,
            format!("extern {tls}struct {struct_name} {c_name}{asm};"),
        );
    }
}

// -- Required trait impls for the codegen framework --

impl<'tcx> BackendTypes for CodegenCx<'tcx> {
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

/// Dummy funclet (unused, only needed for Windows SEH).
pub(crate) struct CFunclet;

/// Dummy debug scope.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DebugScope;

/// Dummy debug location.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DebugLoc;

/// Dummy debug variable.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DebugVar;

// --- HasTyCtxt, HasParamEnv, etc. ---

impl<'tcx> rustc_middle::ty::layout::HasTyCtxt<'tcx> for CodegenCx<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> HasTypingEnv<'tcx> for CodegenCx<'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv::fully_monomorphized()
    }
}

impl<'tcx> rustc_abi::HasDataLayout for CodegenCx<'tcx> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        &self.tcx.data_layout
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for CodegenCx<'tcx> {
    fn handle_layout_err(&self, err: LayoutError<'tcx>, span: Span, ty: Ty<'tcx>) -> ! {
        if let LayoutError::SizeOverflow(_) | LayoutError::ReferencesError(_) = err {
            self.tcx.dcx().span_fatal(span, format!("{err}"))
        } else {
            self.tcx
                .dcx()
                .span_fatal(span, format!("failed to get layout for `{ty}`: {err}"))
        }
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for CodegenCx<'tcx> {
    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> ! {
        match err {
            FnAbiError::Layout(layout_err) => self.tcx.dcx().span_fatal(
                span,
                format!("fn_abi_of failed: {layout_err:?} for {fn_abi_request:?}"),
            ),
        }
    }
}

// --- MiscCodegenMethods ---

impl<'tcx> MiscCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<ExistentialTraitRef<'tcx>>), ValueRef>> {
        &self.vtables_cache
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> ValueRef {
        if let Some(&v) = self.instances.borrow().get(&instance) {
            return v;
        }
        let sym = self.tcx.symbol_name(instance).name.to_string();
        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
        let fn_ty = crate::type_of::fn_abi_to_c_type(self, fn_abi);
        let c_name = Self::sanitize_name(&sym);
        let ptr_ty = self.type_ptr();
        let val = self.intern_value(
            CValueKind::Function {
                name: c_name.clone(),
                sig: fn_ty,
            },
            ptr_ty,
        );
        self.instances.borrow_mut().insert(instance, val);

        // Emit forward declaration.
        let ret_ty = crate::type_of::fn_abi_ret_type(self, fn_abi);
        let ret_str = self.render_type(ret_ty);
        let asm = Self::asm_label(&sym, &c_name);
        // Emit full typed parameter declarations for all functions.
        // This ensures the C compiler knows the correct ABI for each
        // parameter (e.g., float in FP registers vs integer in GP registers).
        let (params, is_variadic) = {
            let types = self.types.borrow();
            match types.get(fn_ty) {
                CTypeKind::Function { args, variadic, .. } => (
                    args.iter()
                        .map(|a| self.render_type(*a))
                        .collect::<Vec<_>>(),
                    *variadic,
                ),
                _ => (vec![], false),
            }
        };
        let mut params_str = if params.is_empty() {
            "void".to_string()
        } else {
            params.join(", ")
        };
        if is_variadic {
            if params.is_empty() {
                params_str = "...".to_string();
            } else {
                params_str.push_str(", ...");
            }
        }
        let decl = format!("{ret_str} {c_name}({params_str}){asm};");
        {
            let mut module = self.module.borrow_mut();
            if module.declared_fns.insert(c_name.clone()) {
                module.function_decls.push(decl);
            }
        }

        val
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> ValueRef {
        self.get_fn(instance)
    }

    fn eh_personality(&self) -> ValueRef {
        if let Some(v) = *self.eh_personality.borrow() {
            return v;
        }
        let sig = self.intern_type(CTypeKind::Function {
            ret: self.intern_type(CTypeKind::Int {
                bits: 32,
                signed: true,
            }),
            args: vec![],
            variadic: false,
        });
        let ptr_ty = self.type_ptr();
        let val = self.intern_value(
            CValueKind::Function {
                name: "__rust_eh_personality".into(),
                sig,
            },
            ptr_ty,
        );
        *self.eh_personality.borrow_mut() = Some(val);
        val
    }

    fn sess(&self) -> &Session {
        self.tcx.sess
    }

    fn set_frame_pointer_type(&self, _llfn: ValueRef) {
        // No-op for C backend
    }

    fn apply_target_cpu_attr(&self, _llfn: ValueRef) {
        // No-op for C backend
    }

    fn declare_c_main(&self, _fn_type: TypeRef) -> Option<ValueRef> {
        let ret_ty = self.intern_type(CTypeKind::Int {
            bits: 32,
            signed: true,
        });
        let sig = self.intern_type(CTypeKind::Function {
            ret: ret_ty,
            args: vec![],
            variadic: false,
        });
        let ptr_ty = self.type_ptr();
        let val = self.intern_value(
            CValueKind::Function {
                name: "main".into(),
                sig,
            },
            ptr_ty,
        );

        // Register main in open_functions so codegen can generate its body.
        // Use the standard C main signature: int main(int, char **).
        // Clang enforces this strictly.
        let i32_ty = self.type_i32();
        let params = vec![(i32_ty, "_arg0".into()), (ptr_ty, "_arg1".into())];
        let mut func_def = crate::module::CModule::new_function_def("main".into(), ret_ty, params);
        func_def
            .param_type_overrides
            .insert(1, "char **".to_string());
        self.module
            .borrow_mut()
            .open_functions
            .insert("main".into(), func_def);

        Some(val)
    }
}

// --- PreDefineCodegenMethods ---

impl<'tcx> PreDefineCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn predefine_static(
        &mut self,
        def_id: DefId,
        _linkage: rustc_hir::attrs::Linkage,
        _visibility: Visibility,
        symbol_name: &str,
    ) {
        let c_name = Self::sanitize_name(symbol_name);
        let ptr_ty = self.type_ptr();
        let val = self.intern_value(
            CValueKind::Global {
                name: format!("&{c_name}"),
            },
            ptr_ty,
        );
        self.statics_cache.borrow_mut().insert(def_id, val);
        // Pre-register to prevent conflicting extern declarations
        self.module.borrow_mut().declared_globals.insert(c_name);
    }

    fn predefine_fn(
        &mut self,
        instance: Instance<'tcx>,
        _linkage: rustc_hir::attrs::Linkage,
        _visibility: Visibility,
        symbol_name: &str,
    ) {
        let c_name = Self::sanitize_name(symbol_name);
        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
        let fn_ty = crate::type_of::fn_abi_to_c_type(self, fn_abi);
        let ptr_ty = self.type_ptr();
        let val = self.intern_value(
            CValueKind::Function {
                name: c_name.clone(),
                sig: fn_ty,
            },
            ptr_ty,
        );
        self.instances.borrow_mut().insert(instance, val);

        // Create function forward declaration
        let ret_ty = crate::type_of::fn_abi_ret_type(self, fn_abi);

        let mut param_types: Vec<(TypeRef, String)> = Vec::new();
        let mut arg_idx = 0usize;
        let mut on_stack_params = std::collections::BTreeSet::new();

        // No explicit sret parameter -- the C compiler handles sret
        // natively for ALL ABIs.  codegen_ssa still passes a sret pointer
        // as args[0]; get_param() and call() handle the mapping.

        for arg in fn_abi.args.iter() {
            match arg.mode {
                rustc_target::callconv::PassMode::Ignore => continue,
                rustc_target::callconv::PassMode::Pair(_, _) => {
                    // ScalarPair: decompose into two separate params.
                    // Use scalar_field_to_c_type (same as fn_abi_to_c_type)
                    // so the declaration matches the internal function type.
                    if let rustc_abi::BackendRepr::ScalarPair(a, b) = arg.layout.backend_repr {
                        let a_ty = crate::type_of::scalar_field_to_c_type(self, a, arg.layout, 0);
                        let b_ty = crate::type_of::scalar_field_to_c_type(self, b, arg.layout, 1);
                        param_types.push((a_ty, format!("_arg{arg_idx}")));
                        arg_idx += 1;
                        param_types.push((b_ty, format!("_arg{arg_idx}")));
                        arg_idx += 1;
                    } else {
                        let ty = crate::type_of::fn_abi_arg_type(self, arg);
                        param_types.push((ty, format!("_arg{arg_idx}")));
                        arg_idx += 1;
                    }
                }
                rustc_target::callconv::PassMode::Indirect {
                    meta_attrs: Some(_),
                    ..
                } => {
                    // Unsized indirect: pointer + metadata (two params)
                    let ptr_ty = self.type_ptr();
                    param_types.push((ptr_ty, format!("_arg{arg_idx}")));
                    arg_idx += 1;
                    param_types.push((ptr_ty, format!("_arg{arg_idx}")));
                    arg_idx += 1;
                }
                rustc_target::callconv::PassMode::Indirect {
                    on_stack: true,
                    meta_attrs: None,
                    ..
                } => {
                    let ty = crate::type_of::fn_abi_arg_type(self, arg);
                    on_stack_params.insert(arg_idx);
                    param_types.push((ty, format!("_arg{arg_idx}")));
                    arg_idx += 1;
                }
                _ => {
                    let ty = crate::type_of::fn_abi_arg_type(self, arg);
                    param_types.push((ty, format!("_arg{arg_idx}")));
                    arg_idx += 1;
                }
            }
        }

        let params_str: Vec<_> = param_types
            .iter()
            .map(|(ty, _)| self.render_type(*ty))
            .collect();
        let is_variadic = fn_abi.c_variadic;
        let mut params_joined = if params_str.is_empty() {
            if is_variadic {
                "".to_string()
            } else {
                "void".to_string()
            }
        } else {
            params_str.join(", ")
        };
        if is_variadic {
            if !params_joined.is_empty() {
                params_joined.push_str(", ");
            }
            params_joined.push_str("...");
        }
        let ret_str = self.render_type(ret_ty);
        let is_internal = matches!(
            _linkage,
            rustc_hir::attrs::Linkage::Internal | rustc_hir::attrs::Linkage::AvailableExternally
        );
        let linkage_prefix = if is_internal { "static " } else { "" };
        let static_kw = linkage_prefix;
        let asm = Self::asm_label(symbol_name, &c_name);
        let decl = format!("{static_kw}{ret_str} {c_name}({params_joined}){asm};");
        {
            let mut module = self.module.borrow_mut();
            if module.declared_fns.insert(c_name.clone()) {
                module.function_decls.push(decl);
            }
        }

        // Register FunctionDef in open_functions so append_block works
        let mut func_def = CModule::new_function_def(c_name.clone(), ret_ty, param_types);
        func_def.linkage_prefix = linkage_prefix.to_string();
        func_def.is_variadic = is_variadic;
        func_def.on_stack_params = on_stack_params;
        let is_indirect = matches!(
            fn_abi.ret.mode,
            rustc_target::callconv::PassMode::Indirect { .. }
        );
        func_def.has_indirect_ret = is_indirect;
        if is_indirect {
            func_def.ret_data_type =
                Some(crate::type_of::layout_to_c_type(self, fn_abi.ret.layout));
        }
        self.module
            .borrow_mut()
            .open_functions
            .insert(c_name, func_def);
    }
}
