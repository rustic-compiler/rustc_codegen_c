//! Rust-to-C transpiler codegen backend.
//!
//! This backend generates C source code from Rust's MIR, then compiles
//! the C code with a system C compiler to produce object files.

#![allow(dead_code, unreachable_pub)]
#![feature(extern_types)]
#![feature(impl_trait_in_assoc_type)]
#![feature(try_blocks)]

use std::any::Any;
use std::path::PathBuf;
use std::time::Instant;

use rustc_ast::expand::allocator::AllocatorMethod;
use rustc_codegen_ssa::back::lto::{SerializedModule, ThinModule};
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLtoInput, ModuleConfig, TargetMachineFactoryFn,
};
use rustc_codegen_ssa::base::maybe_create_entry_wrapper;
use rustc_codegen_ssa::mono_item::MonoItemExt;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{CodegenResults, CompiledModule, ModuleCodegen, ModuleKind, TargetConfig};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::DiagCtxtHandle;
use rustc_middle::dep_graph::{self, WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{OutputFilenames, PrintRequest};
use rustc_span::Symbol;

mod allocator;
mod builder;
mod builtins;
mod consts;
mod context;
mod debuginfo;
mod intrinsic;
mod module;
mod type_of;
mod types;
mod values;
mod write;

use builder::Builder;
use context::CodegenCx;
use module::{CModule, CModuleBuffer};

// =====================================================================
// The main codegen backend
// =====================================================================

#[derive(Clone)]
pub struct CCodegenBackend(());

impl CCodegenBackend {
    pub fn new() -> Box<dyn CodegenBackend> {
        Box::new(CCodegenBackend(()))
    }
}

/// Entry point for loading this backend from the sysroot as a dylib.
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    CCodegenBackend::new()
}

impl CodegenBackend for CCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        ""
    }

    fn name(&self) -> &'static str {
        "c"
    }

    #[allow(rustc::potential_query_instability)]
    fn target_config(&self, sess: &Session) -> TargetConfig {
        // Collect baseline features. The target spec uses LLVM feature names
        // (e.g. "+v8a") which don't map directly to Rust feature names.
        // Instead, start from ABI-required features and the target spec's
        // enabled features, expanding implied features.
        //
        // IMPORTANT: The C codegen cannot correctly translate SIMD/vector
        // intrinsics (NEON, SVE, SSE, AVX, etc.) to C. While it emits
        // GCC vector extension types, the NEON intrinsic functions
        // (vdupq_n_u8, vceqq_u8, vshrn_n_u16, movemask, etc.) produce
        // incorrect results when compiled through C. This causes subtle
        // bugs: e.g. memchr's NEON path fails for haystack positions >= 16,
        // breaking the fluent-syntax parser in proc-macros.
        //
        // We therefore exclude SIMD features so that crates using
        // `#[cfg(target_feature = "neon")]` (like memchr) fall back to
        // their scalar implementations, which the C codegen handles
        // correctly. The system C compiler still uses SIMD for its own
        // optimizations (e.g. memcpy, string ops), so performance is
        // acceptable.
        const SIMD_FEATURES: &[&str] = &[
            "neon", "sve", "sve2", "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "avx",
            "avx2", "avx512f", "simd128",
        ];

        let mut base_features = rustc_data_structures::fx::FxHashSet::default();

        // ABI-required features (e.g. "neon" on aarch64)
        let abi = sess.target.abi_required_features();
        for &feat in abi.required {
            base_features.extend(sess.target.implied_target_features(feat));
        }

        // Features from the target spec and -Ctarget-feature
        for source in [&*sess.target.features, &*sess.opts.cg.target_feature] {
            for feat in source.split(',') {
                let feat = feat.trim();
                if let Some(name) = feat.strip_prefix('+') {
                    base_features.extend(sess.target.implied_target_features(name));
                }
            }
        }

        // Remove SIMD features the C codegen cannot handle.
        // During bootstrap (RUSTC_STAGE is set), exclude ALL SIMD features
        // including ABI-required ones like neon, because the compiler's own
        // dependencies (memchr via fluent-syntax) use #[cfg(target_feature)]
        // to select SIMD paths that codegen_c translates incorrectly.
        // Outside bootstrap, keep ABI-required features (e.g. neon on
        // aarch64) since they are baseline and crates like ring assert
        // their presence at compile time.
        let is_bootstrap = std::env::var_os("RUSTC_STAGE").is_some();
        for &simd in SIMD_FEATURES {
            if is_bootstrap || !abi.required.contains(&simd) {
                base_features.remove(simd);
            }
        }

        let (target_features, unstable_target_features) =
            rustc_codegen_ssa::target_features::cfg_target_feature::<1>(
                sess,
                |_feature| rustc_data_structures::smallvec::SmallVec::new(),
                |feature| base_features.contains(feature),
            );

        TargetConfig {
            target_features,
            unstable_target_features,
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Box<dyn Any> {
        Box::new(rustc_codegen_ssa::base::codegen_crate(
            CCodegenBackend(()),
            tcx,
            "generic".to_string(),
        ))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        _outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<CCodegenBackend>>()
            .expect("Expected CCodegenBackend's OngoingCodegen, found Box<Any>")
            .join(sess)
    }

    fn print(&self, _req: &PrintRequest, _out: &mut String, _sess: &Session) {}
}

// =====================================================================
// ExtraBackendMethods
// =====================================================================

impl ExtraBackendMethods for CCodegenBackend {
    fn codegen_allocator<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        module_name: &str,
        methods: &[AllocatorMethod],
    ) -> CModule {
        let mut module = CModule::new(module_name.to_string());
        allocator::codegen(_tcx, &mut module, module_name, methods);
        module
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<CModule>, u64) {
        compile_codegen_unit(tcx, cgu_name)
    }

    fn target_machine_factory(
        &self,
        _sess: &Session,
        _opt_level: rustc_session::config::OptLevel,
        _target_features: &[String],
    ) -> TargetMachineFactoryFn<Self> {
        std::sync::Arc::new(|_config| Ok(()))
    }

    fn supports_parallel(&self) -> bool {
        true
    }
}

/// Compile a single codegen unit to a CModule.
fn compile_codegen_unit(tcx: TyCtxt<'_>, cgu_name: Symbol) -> (ModuleCodegen<CModule>, u64) {
    let start_time = Instant::now();

    let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
    let (module, _) = tcx.dep_graph.with_task(
        dep_node,
        tcx,
        cgu_name,
        module_codegen,
        Some(dep_graph::hash_result),
    );
    let cost = start_time.elapsed().as_nanos() as u64;

    fn module_codegen(tcx: TyCtxt<'_>, cgu_name: Symbol) -> ModuleCodegen<CModule> {
        let cgu = tcx.codegen_unit(cgu_name);

        let mut cx = CodegenCx::new(tcx, cgu, cgu_name.as_str());

        // Predefine all mono items (forward declarations)
        let mono_items = cx.codegen_unit.items_in_deterministic_order(cx.tcx);
        for &(mono_item, data) in &mono_items {
            mono_item.predefine::<Builder<'_, '_>>(
                &mut cx,
                cgu_name.as_str(),
                data.linkage,
                data.visibility,
            );
        }

        // Define all mono items (generate code)
        for &(mono_item, item_data) in &mono_items {
            mono_item.define::<Builder<'_, '_>>(&mut cx, cgu_name.as_str(), item_data);
        }

        // Create entry wrapper (main)
        maybe_create_entry_wrapper::<Builder<'_, '_>>(&cx, cx.codegen_unit);

        // Finalize any open functions
        let open_fn_names: Vec<_> = cx.module.borrow().open_functions.keys().cloned().collect();
        {
            let types = cx.types.borrow();
            for name in open_fn_names {
                cx.module.borrow_mut().finalize_function(&name, &types);
            }
        }

        let mut module = cx.module.into_inner();
        module.types = cx.types.into_inner();
        module.values = cx.values.into_inner();
        ModuleCodegen {
            name: cgu_name.to_string(),
            module_llvm: module,
            kind: ModuleKind::Regular,
            thin_lto_buffer: None,
        }
    }

    (module, cost)
}

// =====================================================================
// WriteBackendMethods
// =====================================================================

/// Thin buffer: holds serialized C source for thin LTO pass-through.
pub struct CThinBuffer(Vec<u8>);

impl rustc_codegen_ssa::traits::ThinBufferMethods for CThinBuffer {
    fn data(&self) -> &[u8] {
        &self.0
    }
}

impl WriteBackendMethods for CCodegenBackend {
    type Module = CModule;
    type TargetMachine = ();
    type TargetMachineError = String;
    type ModuleBuffer = CModuleBuffer;
    type ThinData = ();
    type ThinBuffer = CThinBuffer;

    fn run_and_optimize_fat_lto(
        _cgcx: &CodegenContext<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<FatLtoInput<Self>>,
    ) -> ModuleCodegen<Self::Module> {
        // Fat LTO: just use the first module
        modules
            .into_iter()
            .next()
            .map(|input| match input {
                FatLtoInput::Serialized { .. } => {
                    panic!("serialized LTO not supported by C backend")
                }
                FatLtoInput::InMemory(m) => m,
            })
            .expect("no modules for fat LTO")
    }

    fn run_thin_lto(
        _cgcx: &CodegenContext<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>) {
        // Pass through: wrap modules in ThinModule for optimize_thin
        use std::ffi::CString;
        let names: Vec<CString> = modules
            .iter()
            .map(|(n, _)| CString::new(n.as_str()).unwrap())
            .collect();
        let buffers: Vec<CThinBuffer> = modules.into_iter().map(|(_, buf)| buf).collect();
        let shared = std::sync::Arc::new(rustc_codegen_ssa::back::lto::ThinShared {
            data: (),
            thin_buffers: buffers,
            serialized_modules: vec![],
            module_names: names,
        });
        let thin_modules: Vec<_> = (0..shared.module_names.len())
            .map(|idx| ThinModule {
                shared: shared.clone(),
                idx,
            })
            .collect();
        (
            thin_modules,
            cached_modules.into_iter().map(|(_, wp)| wp).collect(),
        )
    }

    fn print_pass_timings(&self) {
        // No-op
    }

    fn print_statistics(&self) {
        // No-op
    }

    fn optimize(
        _cgcx: &CodegenContext<Self>,
        _dcx: DiagCtxtHandle<'_>,
        _module: &mut ModuleCodegen<Self::Module>,
        _config: &ModuleConfig,
    ) {
        // The C compiler handles optimization; nothing to do here.
    }

    fn optimize_thin(
        _cgcx: &CodegenContext<Self>,
        thin: ThinModule<Self>,
    ) -> ModuleCodegen<Self::Module> {
        // For the C backend, "thin LTO" is a pass-through: reconstruct
        // the CModule from the serialized C source and compile it.
        let name = thin.name().to_string();
        let mut module = CModule::new(name.clone());
        // The actual C source was serialized in prepare_thin; we store
        // it in the CModule for write::codegen to use.
        module.precompiled_source =
            Some(String::from_utf8(thin.data().to_vec()).unwrap_or_default());
        ModuleCodegen {
            name,
            module_llvm: module,
            kind: ModuleKind::Regular,
            thin_lto_buffer: None,
        }
    }

    fn codegen(
        cgcx: &CodegenContext<Self>,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> CompiledModule {
        write::codegen(cgcx, module, config)
    }

    fn prepare_thin(module: ModuleCodegen<Self::Module>) -> (String, Self::ThinBuffer) {
        // Serialize the module so optimize_thin can reconstruct it
        let source = module.module_llvm.to_c_source();
        (module.name, CThinBuffer(source.into_bytes()))
    }

    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        let name = module.name.clone();
        let source = module.module_llvm.to_c_source();
        let buffer = CModuleBuffer::new(&source);
        (name, buffer)
    }
}
