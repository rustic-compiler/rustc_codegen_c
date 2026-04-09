/// C type representation for the codegen backend.
///
/// Types are interned via `TypeRef` indices into a `TypeStore`.
/// Each type maps to a C type string when rendered.
use rustc_abi::AddressSpace;
use rustc_abi::Reg;
use rustc_codegen_ssa::common::TypeKind;
use rustc_codegen_ssa::traits::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::Ty;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_target::callconv::{CastTarget, FnAbi};

use crate::context::CodegenCx;

/// An interned reference to a C type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypeRef(pub u32);

/// The kind of C type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CTypeKind {
    Void,
    Bool,
    /// Integer type: (bits, signed).
    Int {
        bits: u32,
        signed: bool,
    },
    /// Floating-point type: bits (16, 32, 64, 128).
    Float {
        bits: u32,
    },
    /// Opaque pointer (`void *`).
    Ptr,
    /// Array type.
    Array {
        element: TypeRef,
        len: u64,
    },
    /// Struct type with optional name for named structs.
    Struct {
        fields: Vec<TypeRef>,
        packed: bool,
        name: Option<String>,
    },
    /// Function signature type.
    Function {
        ret: TypeRef,
        args: Vec<TypeRef>,
        variadic: bool,
    },
    /// Vector (SIMD) type.
    Vector {
        element: TypeRef,
        len: u64,
    },
    /// Pointer-width integer (renders as intptr_t / uintptr_t).
    /// Portable across architectures with the same or different pointer
    /// widths -- the C compiler picks the correct size.
    PtrWidth {
        signed: bool,
    },
}

/// Storage for interned types.
#[derive(Debug)]
pub struct TypeStore {
    types: Vec<CTypeKind>,
    index: FxHashMap<CTypeKind, TypeRef>,
}

impl TypeStore {
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            index: FxHashMap::default(),
        }
    }

    pub fn intern(&mut self, kind: CTypeKind) -> TypeRef {
        if let Some(&existing) = self.index.get(&kind) {
            return existing;
        }
        let idx = self.types.len();
        let type_ref = TypeRef(idx as u32);
        self.types.push(kind.clone());
        self.index.insert(kind, type_ref);
        type_ref
    }

    pub fn get(&self, ty: TypeRef) -> &CTypeKind {
        &self.types[ty.0 as usize]
    }

    /// Compute the minimum natural alignment (in bytes) that the C compiler
    /// would assign to a type.  This is used to avoid emitting `_Alignas(N)`
    /// with N smaller than the type's natural alignment, which Clang rejects.
    pub fn natural_align_bytes(&self, ty: TypeRef, pointer_bytes: u64) -> u64 {
        match self.get(ty) {
            CTypeKind::Void => 1,
            CTypeKind::Bool => 1,
            CTypeKind::Int { bits, .. } => {
                let c_bits = match bits {
                    0..=8 => 8,
                    9..=16 => 16,
                    17..=32 => 32,
                    33..=64 => 64,
                    _ => 128,
                };
                (c_bits / 8).max(1)
            }
            CTypeKind::Float { bits } => (*bits as u64 / 8).max(1),
            CTypeKind::Ptr | CTypeKind::PtrWidth { .. } => pointer_bytes,
            CTypeKind::Array { element, .. } => self.natural_align_bytes(*element, pointer_bytes),
            CTypeKind::Vector { element, len } => {
                // GCC/Clang vector types have alignment equal to their total
                // byte size (rounded up to the next power of two).
                let elem_bytes = self.natural_align_bytes(*element, pointer_bytes);
                let total = elem_bytes * len;
                total.next_power_of_two()
            }
            CTypeKind::Struct { fields, packed, .. } => {
                if *packed {
                    1
                } else {
                    fields
                        .iter()
                        .map(|f| self.natural_align_bytes(*f, pointer_bytes))
                        .max()
                        .unwrap_or(1)
                }
            }
            CTypeKind::Function { .. } => pointer_bytes,
        }
    }

    /// Render all unnamed struct type definitions for the C source preamble.
    pub fn render_struct_defs(&self) -> Vec<String> {
        let mut defs = Vec::new();
        for (i, kind) in self.types.iter().enumerate() {
            if let CTypeKind::Struct {
                fields,
                packed,
                name: None,
            } = kind
            {
                let fields_str: Vec<_> = fields
                    .iter()
                    .enumerate()
                    .map(|(j, f)| format!("  {};", self.render_decl(*f, &format!("f{j}"))))
                    .collect();
                let packed_attr = if *packed {
                    "__attribute__((packed)) "
                } else {
                    ""
                };
                defs.push(format!(
                    "struct {packed_attr}_S{i} {{\n{}\n}};",
                    fields_str.join("\n")
                ));
            }
        }
        defs
    }

    /// Render a type as a C type string (for variable declarations, the
    /// variable name is appended after this string).
    pub fn render(&self, ty: TypeRef) -> String {
        match self.get(ty) {
            CTypeKind::Void => "void".into(),
            CTypeKind::Bool => "_Bool".into(),
            CTypeKind::Int { bits: 1, .. } => "_Bool".into(),
            CTypeKind::Int { bits, signed } => {
                // C only has standard integer widths. Round up
                // non-standard widths (e.g. 24 from i24 scalars) to
                // the next available C type.
                let c_bits = match bits {
                    0..=8 => 8,
                    9..=16 => 16,
                    17..=32 => 32,
                    33..=64 => 64,
                    _ => 128,
                };
                if *signed {
                    format!("int{c_bits}_t")
                } else {
                    format!("uint{c_bits}_t")
                }
            }
            CTypeKind::Float { bits: 16 } => "_Float16".into(),
            CTypeKind::Float { bits: 32 } => "float".into(),
            CTypeKind::Float { bits: 64 } => "double".into(),
            CTypeKind::Float { bits: 128 } => "_Float128".into(),
            CTypeKind::Float { bits } => panic!("unsupported float width: {bits}"),
            CTypeKind::Ptr => "void *".into(),
            CTypeKind::Array { element, len } => {
                // For declarations we use a typedef; for inline use, we
                // render as a pointer cast. This is simplified.
                format!("{}[{}]", self.render(*element), len)
            }
            CTypeKind::Struct {
                name: Some(name), ..
            } => format!("struct {name}"),
            CTypeKind::Struct { name: None, .. } => {
                // Use TypeRef index as a stable, unique name
                format!("struct _S{}", ty.0)
            }
            CTypeKind::Function {
                ret,
                args,
                variadic,
            } => {
                let args_str: Vec<_> = args.iter().map(|a| self.render(*a)).collect();
                let mut args_joined = args_str.join(", ");
                if args.is_empty() {
                    args_joined = "void".into();
                }
                if *variadic {
                    args_joined.push_str(", ...");
                }
                format!("{} (*)({})", self.render(*ret), args_joined)
            }
            CTypeKind::Vector { element, len } => {
                // GCC vector extension.
                // GCC doesn't allow pointer types as vector elements;
                // use uintptr_t (same size) instead.
                let elem_str = match self.get(*element) {
                    CTypeKind::Ptr | CTypeKind::PtrWidth { .. } => "uintptr_t".to_string(),
                    _ => self.render(*element),
                };
                format!("{elem_str} __attribute__((vector_size({len} * sizeof({elem_str}))))")
            }
            CTypeKind::PtrWidth { signed } => {
                if *signed {
                    "intptr_t".into()
                } else {
                    "uintptr_t".into()
                }
            }
        }
    }

    /// Render a type for a variable declaration (name inserted properly for
    /// arrays and function pointers where C syntax requires it).
    pub fn render_decl(&self, ty: TypeRef, name: &str) -> String {
        match self.get(ty) {
            CTypeKind::Array { element, len } => {
                format!("{} {name}[{len}]", self.render(*element))
            }
            CTypeKind::Function {
                ret,
                args,
                variadic,
            } => {
                let args_str: Vec<_> = args.iter().map(|a| self.render(*a)).collect();
                let mut args_joined = args_str.join(", ");
                if args.is_empty() {
                    args_joined = "void".into();
                }
                if *variadic {
                    args_joined.push_str(", ...");
                }
                format!("{} (*{name})({})", self.render(*ret), args_joined)
            }
            _ => format!("{} {name}", self.render(ty)),
        }
    }
}

// -- Trait implementations for CodegenCx --

impl<'tcx> BaseTypeCodegenMethods for CodegenCx<'tcx> {
    fn type_i8(&self) -> TypeRef {
        self.intern_type(CTypeKind::Int {
            bits: 8,
            signed: true,
        })
    }
    fn type_i16(&self) -> TypeRef {
        self.intern_type(CTypeKind::Int {
            bits: 16,
            signed: true,
        })
    }
    fn type_i32(&self) -> TypeRef {
        self.intern_type(CTypeKind::Int {
            bits: 32,
            signed: true,
        })
    }
    fn type_i64(&self) -> TypeRef {
        self.intern_type(CTypeKind::Int {
            bits: 64,
            signed: true,
        })
    }
    fn type_i128(&self) -> TypeRef {
        self.intern_type(CTypeKind::Int {
            bits: 128,
            signed: true,
        })
    }
    fn type_isize(&self) -> TypeRef {
        self.intern_type(CTypeKind::PtrWidth { signed: true })
    }

    fn type_f16(&self) -> TypeRef {
        self.intern_type(CTypeKind::Float { bits: 16 })
    }
    fn type_f32(&self) -> TypeRef {
        self.intern_type(CTypeKind::Float { bits: 32 })
    }
    fn type_f64(&self) -> TypeRef {
        self.intern_type(CTypeKind::Float { bits: 64 })
    }
    fn type_f128(&self) -> TypeRef {
        self.intern_type(CTypeKind::Float { bits: 128 })
    }

    fn type_array(&self, ty: TypeRef, len: u64) -> TypeRef {
        self.intern_type(CTypeKind::Array { element: ty, len })
    }

    fn type_func(&self, args: &[TypeRef], ret: TypeRef) -> TypeRef {
        self.intern_type(CTypeKind::Function {
            ret,
            args: args.to_vec(),
            variadic: false,
        })
    }

    fn type_kind(&self, ty: TypeRef) -> TypeKind {
        let store = self.types.borrow();
        match store.get(ty) {
            CTypeKind::Void => TypeKind::Void,
            CTypeKind::Bool => TypeKind::Integer,
            CTypeKind::Int { bits: 1, .. } => TypeKind::Integer,
            CTypeKind::Int { .. } => TypeKind::Integer,
            CTypeKind::Float { bits: 16 } => TypeKind::Half,
            CTypeKind::Float { bits: 32 } => TypeKind::Float,
            CTypeKind::Float { bits: 64 } => TypeKind::Double,
            CTypeKind::Float { bits: 128 } => TypeKind::FP128,
            CTypeKind::Float { .. } => TypeKind::Float,
            CTypeKind::Ptr => TypeKind::Pointer,
            CTypeKind::Array { .. } => TypeKind::Array,
            CTypeKind::Struct { .. } => TypeKind::Struct,
            CTypeKind::Function { .. } => TypeKind::Function,
            CTypeKind::Vector { .. } => TypeKind::ScalableVector,
            CTypeKind::PtrWidth { .. } => TypeKind::Integer,
        }
    }

    fn type_ptr(&self) -> TypeRef {
        self.intern_type(CTypeKind::Ptr)
    }

    fn type_ptr_ext(&self, _address_space: AddressSpace) -> TypeRef {
        // C doesn't have address spaces; just use void*
        self.type_ptr()
    }

    fn element_type(&self, ty: TypeRef) -> TypeRef {
        let store = self.types.borrow();
        match store.get(ty) {
            CTypeKind::Array { element, .. } => *element,
            CTypeKind::Vector { element, .. } => *element,
            _ => panic!("element_type on non-sequential type"),
        }
    }

    fn vector_length(&self, ty: TypeRef) -> usize {
        let store = self.types.borrow();
        match store.get(ty) {
            CTypeKind::Vector { len, .. } => *len as usize,
            _ => panic!("vector_length on non-vector type"),
        }
    }

    fn float_width(&self, ty: TypeRef) -> usize {
        let store = self.types.borrow();
        match store.get(ty) {
            CTypeKind::Float { bits } => *bits as usize,
            _ => panic!("float_width on non-float type"),
        }
    }

    fn int_width(&self, ty: TypeRef) -> u64 {
        let store = self.types.borrow();
        match store.get(ty) {
            CTypeKind::Bool => 1,
            CTypeKind::Int { bits, .. } => *bits as u64,
            CTypeKind::PtrWidth { .. } => self.tcx.data_layout.pointer_size().bits(),
            // Pointer-typed values can appear in integer contexts (e.g., shifts)
            CTypeKind::Ptr => self.tcx.data_layout.pointer_size().bits(),
            other => panic!(
                "int_width on non-integer type: {:?} (TypeRef({}))",
                other, ty.0
            ),
        }
    }

    fn val_ty(&self, v: crate::values::ValueRef) -> TypeRef {
        self.values.borrow().get_type(v)
    }
}

impl<'tcx> LayoutTypeCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> TypeRef {
        crate::type_of::layout_to_c_type(self, layout)
    }

    fn cast_backend_type(&self, ty: &CastTarget) -> TypeRef {
        crate::type_of::cast_target_to_c_type(self, ty)
    }

    fn fn_decl_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> TypeRef {
        crate::type_of::fn_abi_to_c_type(self, fn_abi)
    }

    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> TypeRef {
        // Function pointer type is the same as function declaration type in our repr
        self.fn_decl_backend_type(fn_abi)
    }

    fn reg_backend_type(&self, ty: &Reg) -> TypeRef {
        let bits = ty.size.bits();
        match ty.kind {
            rustc_abi::RegKind::Integer => self.intern_type(CTypeKind::Int {
                bits: bits as u32,
                signed: true,
            }),
            rustc_abi::RegKind::Float => self.intern_type(CTypeKind::Float { bits: bits as u32 }),
            rustc_abi::RegKind::Vector => {
                let elem = self.intern_type(CTypeKind::Int {
                    bits: 8,
                    signed: true,
                });
                self.intern_type(CTypeKind::Vector {
                    element: elem,
                    len: bits / 8,
                })
            }
        }
    }

    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> TypeRef {
        self.backend_type(layout)
    }

    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        match layout.backend_repr {
            rustc_abi::BackendRepr::Scalar(_) | rustc_abi::BackendRepr::SimdVector { .. } => true,
            rustc_abi::BackendRepr::ScalarPair(..) => false,
            _ => false,
        }
    }

    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool {
        matches!(layout.backend_repr, rustc_abi::BackendRepr::ScalarPair(..))
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: TyAndLayout<'tcx>,
        index: usize,
        _immediate: bool,
    ) -> TypeRef {
        let (a, b) = match layout.backend_repr {
            rustc_abi::BackendRepr::ScalarPair(a, b) => (a, b),
            _ => panic!("scalar_pair_element_backend_type on non-scalar-pair"),
        };
        let scalar = if index == 0 { a } else { b };
        crate::type_of::scalar_field_to_c_type(self, scalar, layout, index)
    }
}

impl<'tcx> TypeMembershipCodegenMethods<'tcx> for CodegenCx<'tcx> {
    // Use default no-op implementations
}
