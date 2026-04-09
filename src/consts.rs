/// Constant and static codegen: generates C representations for
/// Rust constants, string literals, and static variables.
use rustc_abi::Size;
use rustc_codegen_ssa::traits::*;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret;
use rustc_middle::ty::Instance;
use rustc_middle::ty::layout::HasTypingEnv;

use crate::context::CodegenCx;
use crate::types::{CTypeKind, TypeRef};
use crate::values::{CValueKind, ValueRef};

/// Sign-extend a u128 bit pattern from `bits` width to i128.
fn sign_extend_to_i128(val: u128, bits: u32) -> i128 {
    if bits >= 128 {
        val as i128
    } else {
        let shift = 128 - bits;
        ((val as i128) << shift) >> shift
    }
}

// --- ConstCodegenMethods ---

impl<'tcx> ConstCodegenMethods for CodegenCx<'tcx> {
    fn const_null(&self, t: TypeRef) -> ValueRef {
        self.intern_value(CValueKind::NullPtr, t)
    }
    fn const_undef(&self, t: TypeRef) -> ValueRef {
        self.intern_value(CValueKind::Undef, t)
    }
    fn const_poison(&self, t: TypeRef) -> ValueRef {
        self.intern_value(CValueKind::Poison, t)
    }
    fn const_bool(&self, val: bool) -> ValueRef {
        let ty = self.intern_type(CTypeKind::Bool);
        self.intern_value(CValueKind::BoolConst(val), ty)
    }
    fn const_i8(&self, i: i8) -> ValueRef {
        let ty = self.type_i8();
        self.intern_value(CValueKind::IntConst(i as i128), ty)
    }
    fn const_i16(&self, i: i16) -> ValueRef {
        let ty = self.type_i16();
        self.intern_value(CValueKind::IntConst(i as i128), ty)
    }
    fn const_i32(&self, i: i32) -> ValueRef {
        let ty = self.type_i32();
        self.intern_value(CValueKind::IntConst(i as i128), ty)
    }
    fn const_int(&self, t: TypeRef, i: i64) -> ValueRef {
        self.intern_value(CValueKind::IntConst(i as i128), t)
    }
    fn const_u8(&self, i: u8) -> ValueRef {
        let ty = self.type_i8();
        self.intern_value(CValueKind::UintConst(i as u128), ty)
    }
    fn const_u32(&self, i: u32) -> ValueRef {
        let ty = self.type_i32();
        self.intern_value(CValueKind::UintConst(i as u128), ty)
    }
    fn const_u64(&self, i: u64) -> ValueRef {
        let ty = self.type_i64();
        self.intern_value(CValueKind::UintConst(i as u128), ty)
    }
    fn const_u128(&self, i: u128) -> ValueRef {
        let ty = self.type_i128();
        self.intern_value(CValueKind::UintConst(i), ty)
    }
    fn const_usize(&self, i: u64) -> ValueRef {
        let ty = self.type_usize();
        self.intern_value(CValueKind::UintConst(i as u128), ty)
    }
    fn const_uint(&self, t: TypeRef, i: u64) -> ValueRef {
        // If the target type is signed, store as IntConst with proper
        // sign-extension so C renders a signed literal (preventing
        // zero-extension when casting to wider types).
        let signed_bits = {
            let types = self.types.borrow();
            match types.get(t) {
                CTypeKind::Int { bits, signed: true } => Some(*bits),
                CTypeKind::PtrWidth { signed: true } => {
                    Some(self.tcx.data_layout.pointer_size().bits() as u32)
                }
                _ => None,
            }
        };
        if let Some(bits) = signed_bits {
            let sext = sign_extend_to_i128(i as u128, bits);
            self.intern_value(CValueKind::IntConst(sext), t)
        } else {
            self.intern_value(CValueKind::UintConst(i as u128), t)
        }
    }
    fn const_uint_big(&self, t: TypeRef, u: u128) -> ValueRef {
        let signed_bits = {
            let types = self.types.borrow();
            match types.get(t) {
                CTypeKind::Int { bits, signed: true } => Some(*bits),
                CTypeKind::PtrWidth { signed: true } => {
                    Some(self.tcx.data_layout.pointer_size().bits() as u32)
                }
                _ => None,
            }
        };
        if let Some(bits) = signed_bits {
            let sext = sign_extend_to_i128(u, bits);
            self.intern_value(CValueKind::IntConst(sext), t)
        } else {
            self.intern_value(CValueKind::UintConst(u), t)
        }
    }
    fn const_real(&self, t: TypeRef, val: f64) -> ValueRef {
        self.intern_value(CValueKind::FloatConst(val), t)
    }

    fn const_str(&self, s: &str) -> (ValueRef, ValueRef) {
        let ptr_ty = self.type_ptr();
        let ptr_val = self
            .values
            .borrow_mut()
            .next_string_literal(s.to_string(), ptr_ty);
        let name = self.values.borrow().render(ptr_val);

        // Use byte array form to correctly handle embedded null bytes
        let hex: Vec<_> = s.bytes().map(|b| format!("0x{b:02x}")).collect();
        let decl = format!(
            "static const unsigned char {name}[] = {{ {} }};",
            hex.join(", ")
        );
        self.module.borrow_mut().data_sections.push(decl);

        let len_val = self.const_usize(s.len() as u64);
        (ptr_val, len_val)
    }

    fn const_struct(&self, elts: &[ValueRef], _packed: bool) -> ValueRef {
        // Determine the struct type from element types
        let field_types: Vec<_> = elts
            .iter()
            .map(|e| self.values.borrow().get_type(*e))
            .collect();
        let ty = self.intern_type(CTypeKind::Struct {
            fields: field_types,
            packed: _packed,
            name: None,
        });
        self.intern_value(
            CValueKind::StructConst {
                fields: elts.to_vec(),
            },
            ty,
        )
    }

    fn const_vector(&self, elts: &[ValueRef]) -> ValueRef {
        let elem_ty = if elts.is_empty() {
            self.type_i8()
        } else {
            self.values.borrow().get_type(elts[0])
        };
        let ty = self.intern_type(CTypeKind::Vector {
            element: elem_ty,
            len: elts.len() as u64,
        });
        let type_str = self.render_type(ty);
        self.intern_value(
            CValueKind::VectorConst {
                elements: elts.to_vec(),
                type_str,
            },
            ty,
        )
    }

    fn const_to_opt_uint(&self, v: ValueRef) -> Option<u64> {
        self.values.borrow().as_u64(v)
    }

    fn const_to_opt_u128(&self, v: ValueRef, sign_ext: bool) -> Option<u128> {
        self.values.borrow().as_u128(v, sign_ext)
    }

    fn scalar_to_backend(
        &self,
        cv: rustc_middle::mir::interpret::Scalar,
        layout: rustc_abi::Scalar,
        llty: TypeRef,
    ) -> ValueRef {
        use rustc_middle::mir::interpret::Scalar as InterpScalar;
        match cv {
            InterpScalar::Int(int) => {
                let data = int.to_bits(layout.size(self));
                // When the C type is a pointer (e.g., BackendRepr::Scalar(Pointer)
                // for types like Alignment), cast through uintptr_t to avoid
                // -Wint-conversion and strict-aliasing UB in generated C code.
                let is_ptr = matches!(self.types.borrow().get(llty), CTypeKind::Ptr);
                if is_ptr {
                    let expr = if data <= u64::MAX as u128 {
                        format!("(void *)(uintptr_t){data}ULL")
                    } else {
                        let lo = (data & u64::MAX as u128) as u64;
                        let hi = (data >> 64) as u64;
                        format!("(void *)(uintptr_t)((unsigned __int128){hi}ULL << 64 | {lo}ULL)")
                    };
                    self.intern_value(CValueKind::InlineExpr(expr), llty)
                } else if matches!(layout.primitive(), rustc_abi::Primitive::Float(f) if f.size().bits() == 32)
                {
                    // f32 scalar: reinterpret the raw bits as an f32, then
                    // promote to f64 for the FloatConst representation.
                    let f = f32::from_bits(data as u32);
                    self.intern_value(CValueKind::FloatConst(f as f64), llty)
                } else if matches!(layout.primitive(), rustc_abi::Primitive::Float(f) if f.size().bits() == 64)
                {
                    // f64 scalar: reinterpret the raw bits as an f64.
                    let f = f64::from_bits(data as u64);
                    self.intern_value(CValueKind::FloatConst(f), llty)
                } else if matches!(layout.primitive(), rustc_abi::Primitive::Float(_)) {
                    // f16 / f128: emit as integer bits with a cast, since C may
                    // not have a literal syntax for these types.
                    self.intern_value(CValueKind::UintConst(data), llty)
                } else if matches!(layout.primitive(), rustc_abi::Primitive::Int(_, true)) {
                    // Signed integer scalar: sign-extend the raw bit pattern
                    // so the C literal has the correct signed value.
                    // Without this, e.g. i32(-65) stored as 4 bytes would emit
                    // "4294967231ULL" instead of "-65LL".
                    let bits = layout.size(self).bits();
                    let signed_val = if bits < 128 {
                        let shift = 128 - bits;
                        ((data as i128) << shift) >> shift
                    } else {
                        data as i128
                    };
                    self.intern_value(CValueKind::IntConst(signed_val), llty)
                } else {
                    self.intern_value(CValueKind::UintConst(data), llty)
                }
            }
            InterpScalar::Ptr(ptr, _size) => {
                let (prov, offset) = ptr.into_raw_parts();
                let offset_bytes = offset.bytes();
                let ptr_val = match self.tcx.global_alloc(prov.alloc_id()) {
                    interpret::GlobalAlloc::Function { instance, .. } => self.get_fn_addr(instance),
                    interpret::GlobalAlloc::Static(def_id) => {
                        let sym = self.tcx.symbol_name(Instance::mono(self.tcx, def_id)).name;
                        let c_name = Self::sanitize_name(sym);
                        self.emit_extern_static_decl(&c_name, sym, def_id);
                        // For extern_weak (declared as weak functions in C),
                        // use a compound literal to store the function pointer
                        // in memory, so that *(void**)ptr gives the pointer
                        // value (not the function's code bytes).
                        let addr = if self.is_extern_weak(def_id) {
                            format!("&(void *){{(void *){c_name}}}")
                        } else {
                            format!("&{c_name}")
                        };
                        let val =
                            self.intern_value(CValueKind::Global { name: addr }, self.type_ptr());
                        if offset_bytes != 0 {
                            self.intern_value(
                                CValueKind::PtrOffset {
                                    base: val,
                                    offset: offset_bytes,
                                },
                                self.type_ptr(),
                            )
                        } else {
                            val
                        }
                    }
                    interpret::GlobalAlloc::Memory(alloc) => {
                        // Use const_data_from_alloc which handles relocations
                        let val = self.const_data_from_alloc(alloc);

                        if offset_bytes != 0 {
                            self.intern_value(
                                CValueKind::PtrOffset {
                                    base: val,
                                    offset: offset_bytes,
                                },
                                self.type_ptr(),
                            )
                        } else {
                            val
                        }
                    }
                    interpret::GlobalAlloc::VTable(ty, dyn_ty) => {
                        let alloc = self
                            .tcx
                            .global_alloc(self.tcx.vtable_allocation((
                                ty,
                                dyn_ty.principal().map(|principal| {
                                    self.tcx.instantiate_bound_regions_with_erased(principal)
                                }),
                            )))
                            .unwrap_memory();
                        let val = self.const_data_from_alloc(alloc);

                        if offset_bytes != 0 {
                            self.intern_value(
                                CValueKind::PtrOffset {
                                    base: val,
                                    offset: offset_bytes,
                                },
                                self.type_ptr(),
                            )
                        } else {
                            val
                        }
                    }
                    interpret::GlobalAlloc::TypeId { .. } => {
                        // The TypeId hash is encoded in the pointer offset.
                        // Drop the provenance and use the offset as the value,
                        // matching LLVM's ConstIntToPtr behavior.
                        let expr = format!("(void *)(uintptr_t){offset_bytes}ULL");
                        return self.intern_value(CValueKind::InlineExpr(expr), llty);
                    }
                };
                // Ensure the returned value matches the expected llty.
                // The pointer value may need to be re-typed (e.g., when the
                // caller expects an integer type for a pointer-sized scalar).
                let ptr_ty = self.type_ptr();
                if llty != ptr_ty {
                    let expr = self.render_value(ptr_val);
                    let t = self.render_type(llty);
                    self.intern_value(CValueKind::InlineExpr(format!("({t}){expr}")), llty)
                } else {
                    ptr_val
                }
            }
        }
    }

    fn const_data_from_alloc(
        &self,
        alloc: rustc_middle::mir::interpret::ConstAllocation<'_>,
    ) -> ValueRef {
        let init = alloc.inner();
        let ptr_ty = self.type_ptr();
        let provenance = init.provenance();
        let has_relocs = !provenance.ptrs().is_empty();
        let val = if has_relocs {
            // For struct-type allocations, use Global with &name reference
            let id = self.values.borrow().byte_string_counter();
            let bs_name = format!("_bytes{id}");
            let _v = self
                .values
                .borrow_mut()
                .next_byte_string(Vec::new(), ptr_ty);
            // Override the rendered value to use address-of for struct type
            self.intern_value(
                CValueKind::Global {
                    name: format!("&{bs_name}"),
                },
                ptr_ty,
            )
        } else {
            self.values
                .borrow_mut()
                .next_byte_string(Vec::new(), ptr_ty)
        };
        // Get the raw name for declarations
        let raw_id = self.values.borrow().byte_string_counter() - 1;
        let name = format!("_bytes{raw_id}");

        let pointer_size = self.tcx.data_layout.pointer_size().bytes() as usize;
        let alloc_len = init.len();

        if !has_relocs {
            // No relocations -- emit as simple byte array
            let data = init.get_bytes_unchecked(interpret::alloc_range(Size::ZERO, init.size()));
            let hex: Vec<_> = data.iter().map(|b| format!("0x{b:02x}")).collect();
            let align = init.align.bytes();
            let align_attr = if align > 1 {
                format!("_Alignas({align}) ")
            } else {
                String::new()
            };
            let decl = format!(
                "{align_attr}static const unsigned char {name}[] = {{ {} }};",
                hex.join(", ")
            );
            self.module.borrow_mut().data_sections.push(decl);
        } else {
            // Has relocations. Emit as a struct with byte-array padding
            // between pointers, using static initialization (no constructor).
            let mut fields = Vec::new();
            let mut inits = Vec::new();
            let mut pos = 0usize;
            let mut field_idx = 0usize;

            for &(reloc_offset, prov) in provenance.ptrs().iter() {
                let reloc_off = reloc_offset.bytes() as usize;

                // Padding bytes before this pointer
                if reloc_off > pos {
                    let pad_len = reloc_off - pos;
                    let pad_bytes =
                        init.inspect_with_uninit_and_ptr_outside_interpreter(pos..reloc_off);
                    let hex: Vec<_> = pad_bytes.iter().map(|b| format!("0x{b:02x}")).collect();
                    fields.push(format!("  unsigned char f{field_idx}[{pad_len}];"));
                    inits.push(format!("  .f{field_idx} = {{ {} }}", hex.join(", ")));
                    field_idx += 1;
                }

                // Pointer field
                let ptr_bytes = init.inspect_with_uninit_and_ptr_outside_interpreter(
                    reloc_off..(reloc_off + pointer_size),
                );
                let ptr_offset = match pointer_size {
                    4 => u32::from_le_bytes(ptr_bytes.try_into().unwrap_or([0u8; 4])) as u64,
                    8 => u64::from_le_bytes(ptr_bytes.try_into().unwrap_or([0u8; 8])),
                    _ => panic!("unsupported pointer size: {pointer_size}"),
                };
                let target = match self.tcx.global_alloc(prov.alloc_id()) {
                    interpret::GlobalAlloc::Function { instance, .. } => {
                        let sym = self.tcx.symbol_name(instance).name.to_string();
                        let c_name = Self::sanitize_name(&sym);
                        let _ = self.get_fn(instance);
                        if ptr_offset != 0 {
                            format!("(void *)((char *)(void (*)(void)){c_name} + {ptr_offset})")
                        } else {
                            format!("(void *)(void (*)(void)){c_name}")
                        }
                    }
                    interpret::GlobalAlloc::Static(def_id) => {
                        let sym = self.tcx.symbol_name(Instance::mono(self.tcx, def_id)).name;
                        let c_name = Self::sanitize_name(sym);
                        self.emit_extern_static_decl(&c_name, sym, def_id);
                        if self.is_extern_weak(def_id) {
                            // extern_weak: use function pointer cast
                            format!("(void *){c_name}")
                        } else if ptr_offset != 0 {
                            format!("(void *)((char *)&{c_name} + {ptr_offset})")
                        } else {
                            format!("(void *)&{c_name}")
                        }
                    }
                    interpret::GlobalAlloc::Memory(inner_alloc) => {
                        let inner_val = self.const_data_from_alloc(inner_alloc);
                        let inner_name = self.render_value(inner_val);
                        if ptr_offset != 0 {
                            format!("(void *)((char *){inner_name} + {ptr_offset})")
                        } else {
                            format!("(void *){inner_name}")
                        }
                    }
                    interpret::GlobalAlloc::VTable(ty, dyn_ty) => {
                        let alloc = self
                            .tcx
                            .global_alloc(self.tcx.vtable_allocation((
                                ty,
                                dyn_ty.principal().map(|principal| {
                                    self.tcx.instantiate_bound_regions_with_erased(principal)
                                }),
                            )))
                            .unwrap_memory();
                        let inner_val = self.const_data_from_alloc(alloc);
                        let inner_name = self.render_value(inner_val);
                        if ptr_offset != 0 {
                            format!("(void *)((char *){inner_name} + {ptr_offset})")
                        } else {
                            format!("(void *){inner_name}")
                        }
                    }
                    interpret::GlobalAlloc::TypeId { .. } => {
                        // TypeId hash is encoded in the pointer offset
                        format!("(void *)(uintptr_t){ptr_offset}ULL")
                    }
                };
                fields.push(format!("  void *f{field_idx};"));
                inits.push(format!("  .f{field_idx} = {target}"));
                field_idx += 1;
                pos = reloc_off + pointer_size;
            }

            // Trailing bytes
            if pos < alloc_len {
                let trail_len = alloc_len - pos;
                let trail_bytes =
                    init.inspect_with_uninit_and_ptr_outside_interpreter(pos..alloc_len);
                let hex: Vec<_> = trail_bytes.iter().map(|b| format!("0x{b:02x}")).collect();
                fields.push(format!("  unsigned char f{field_idx}[{trail_len}];"));
                inits.push(format!("  .f{field_idx} = {{ {} }}", hex.join(", ")));
            }

            let decl = format!(
                "#pragma pack(push, 1)\nstatic const struct {{ {fields} }} {name} = {{ {inits} }};\n#pragma pack(pop)",
                fields = fields.join(" "),
                inits = inits.join(", "),
            );
            self.module.borrow_mut().data_sections.push(decl);
        }

        val
    }

    fn const_ptr_byte_offset(&self, val: ValueRef, offset: rustc_abi::Size) -> ValueRef {
        if offset.bytes() == 0 {
            return val;
        }
        let ty = self.type_ptr();
        self.intern_value(
            CValueKind::PtrOffset {
                base: val,
                offset: offset.bytes(),
            },
            ty,
        )
    }
}

// --- StaticCodegenMethods ---

impl<'tcx> StaticCodegenMethods for CodegenCx<'tcx> {
    fn static_addr_of(
        &self,
        cv: ValueRef,
        _align: rustc_abi::Align,
        _kind: Option<&str>,
    ) -> ValueRef {
        // The value is already allocated; just return it
        cv
    }

    fn codegen_static(&mut self, def_id: DefId) {
        let instance = Instance::mono(self.tcx, def_id);
        let sym = self.tcx.symbol_name(instance).name.to_string();
        let c_name = Self::sanitize_name(&sym);
        let asm = Self::asm_label(&sym, &c_name);

        let tls = if self.tcx.is_thread_local_static(def_id) {
            "_Thread_local "
        } else {
            ""
        };
        let pointer_size = self.tcx.data_layout.pointer_size().bytes() as usize;

        // Try to evaluate the static's initial value
        let decl = if let Ok(alloc) = self.tcx.eval_static_initializer(def_id) {
            let init = alloc.inner();
            let provenance = init.provenance();
            let has_relocs = !provenance.ptrs().is_empty();
            let alloc_len = init.len();

            // Use the allocation's actual alignment (not just pointer size).
            // Types with #[repr(align(N))] need N-byte alignment.
            let actual_align = init.align.bytes().max(pointer_size as u64);
            if alloc_len == 0 {
                // Zero-size static
                format!("{tls}uint8_t {c_name}[0]{asm};")
            } else if !has_relocs {
                // No relocations: emit as byte array with initializer
                let data =
                    init.get_bytes_unchecked(interpret::alloc_range(Size::ZERO, init.size()));
                let hex: Vec<_> = data.iter().map(|b| format!("0x{b:02x}")).collect();
                format!(
                    "_Alignas({actual_align}) {tls}uint8_t {c_name}[{alloc_len}]{asm} = {{ {} }};",
                    hex.join(", ")
                )
            } else {
                // Has relocations: emit as named struct with pointer fields.
                // The named struct allows forward declarations (extern struct _gs_X NAME;)
                // that are compatible with the definition.
                let struct_name = format!("_gs_{c_name}");
                let mut fields = Vec::new();
                let mut inits = Vec::new();
                let mut pos = 0usize;
                let mut field_idx = 0usize;

                for &(reloc_offset, prov) in provenance.ptrs().iter() {
                    let reloc_off = reloc_offset.bytes() as usize;

                    // Padding bytes before this pointer
                    if reloc_off > pos {
                        let pad_len = reloc_off - pos;
                        let pad_bytes =
                            init.inspect_with_uninit_and_ptr_outside_interpreter(pos..reloc_off);
                        let hex: Vec<_> = pad_bytes.iter().map(|b| format!("0x{b:02x}")).collect();
                        fields.push(format!("unsigned char f{field_idx}[{pad_len}]"));
                        inits.push(format!(".f{field_idx} = {{ {} }}", hex.join(", ")));
                        field_idx += 1;
                    }

                    // Pointer field
                    let ptr_bytes = init.inspect_with_uninit_and_ptr_outside_interpreter(
                        reloc_off..(reloc_off + pointer_size),
                    );
                    let ptr_offset = match pointer_size {
                        4 => u32::from_le_bytes(ptr_bytes.try_into().unwrap_or([0u8; 4])) as u64,
                        8 => u64::from_le_bytes(ptr_bytes.try_into().unwrap_or([0u8; 8])),
                        _ => panic!("unsupported pointer size: {pointer_size}"),
                    };
                    let target = match self.tcx.global_alloc(prov.alloc_id()) {
                        interpret::GlobalAlloc::Function { instance, .. } => {
                            let sym = self.tcx.symbol_name(instance).name.to_string();
                            let fn_name = Self::sanitize_name(&sym);
                            let _ = self.get_fn(instance);
                            if ptr_offset != 0 {
                                format!(
                                    "(void *)((char *)(void (*)(void)){fn_name} + {ptr_offset})"
                                )
                            } else {
                                format!("(void *)(void (*)(void)){fn_name}")
                            }
                        }
                        interpret::GlobalAlloc::Static(sid) => {
                            let sym = self.tcx.symbol_name(Instance::mono(self.tcx, sid)).name;
                            let sname = Self::sanitize_name(sym);
                            self.emit_extern_static_decl(&sname, sym, sid);
                            if self.is_extern_weak(sid) {
                                format!("(void *){sname}")
                            } else if ptr_offset != 0 {
                                format!("(void *)((char *)&{sname} + {ptr_offset})")
                            } else {
                                format!("(void *)&{sname}")
                            }
                        }
                        interpret::GlobalAlloc::Memory(inner_alloc) => {
                            let inner_val = self.const_data_from_alloc(inner_alloc);
                            let inner_name = self.render_value(inner_val);
                            if ptr_offset != 0 {
                                format!("(void *)((char *){inner_name} + {ptr_offset})")
                            } else {
                                format!("(void *){inner_name}")
                            }
                        }
                        interpret::GlobalAlloc::VTable(ty, dyn_ty) => {
                            let alloc = self
                                .tcx
                                .global_alloc(self.tcx.vtable_allocation((
                                    ty,
                                    dyn_ty.principal().map(|principal| {
                                        self.tcx.instantiate_bound_regions_with_erased(principal)
                                    }),
                                )))
                                .unwrap_memory();
                            let inner_val = self.const_data_from_alloc(alloc);
                            let inner_name = self.render_value(inner_val);
                            if ptr_offset != 0 {
                                format!("(void *)((char *){inner_name} + {ptr_offset})")
                            } else {
                                format!("(void *){inner_name}")
                            }
                        }
                        interpret::GlobalAlloc::TypeId { .. } => {
                            format!("(void *)(uintptr_t){ptr_offset}ULL")
                        }
                    };
                    fields.push(format!("void *f{field_idx}"));
                    inits.push(format!(".f{field_idx} = {target}"));
                    field_idx += 1;
                    pos = reloc_off + pointer_size;
                }

                // Trailing bytes
                if pos < alloc_len {
                    let trail_len = alloc_len - pos;
                    let trail_bytes =
                        init.inspect_with_uninit_and_ptr_outside_interpreter(pos..alloc_len);
                    let hex: Vec<_> = trail_bytes.iter().map(|b| format!("0x{b:02x}")).collect();
                    fields.push(format!("unsigned char f{field_idx}[{trail_len}]"));
                    inits.push(format!(".f{field_idx} = {{ {} }}", hex.join(", ")));
                }

                let inits_str = inits.join(", ");

                // Emit named struct type definition and forward variable declaration
                // (if not already emitted by emit_static_struct_fwd_decl).
                {
                    let module = self.module.borrow();
                    if !module.declared_extern_globals.contains(&c_name) {
                        drop(module);
                        self.emit_static_struct_fwd_decl(&c_name, &sym, def_id, &tls);
                    }
                }

                format!("{tls}struct {struct_name} {c_name}{asm} = {{ {inits_str} }};")
            }
        } else {
            // Can't evaluate: emit uninitialized declaration
            let ty = self.tcx.type_of(def_id).instantiate_identity();
            let layout = self
                .tcx
                .layout_of(self.typing_env().as_query_input(ty))
                .unwrap();
            let c_ty = crate::type_of::layout_to_c_type(self, layout);
            let type_decl = self.render_type_decl(c_ty, &c_name);
            format!("{tls}{type_decl}{asm};")
        };

        // Add link_section attribute if specified (e.g. .init_array for
        // pre-main constructors).
        let attrs = self.tcx.codegen_fn_attrs(def_id);
        // Section placement: no C11 equivalent exists.
        // Keep __attribute__((section(...))) as it is essential for correct
        // behavior (e.g., .init_array for pre-main constructors).
        let decl = if let Some(section) = attrs.link_section {
            let sect = section.as_str();
            format!("__attribute__((section(\"{sect}\"))) {decl}")
        } else {
            decl
        };

        // Add alignment attribute if the static has a specified alignment
        // override (e.g. #[rustc_align_static(N)]).
        let decl = if let Some(align) = attrs.alignment {
            format!("_Alignas({}) {decl}", align.bytes())
        } else {
            decl
        };

        // #[used] / #[used(compiler)] -- prevent linker GC from discarding.
        // No C11 equivalent exists; keep __attribute__((used)) as it is
        // essential for correct linker behavior.
        use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
        let decl = if attrs.flags.contains(CodegenFnAttrFlags::USED_LINKER)
            || attrs.flags.contains(CodegenFnAttrFlags::USED_COMPILER)
        {
            format!("__attribute__((used)) {decl}")
        } else {
            decl
        };

        // Push the definition and register in declared_globals to prevent
        // conflicting extern declarations from being added later.
        {
            let mut module = self.module.borrow_mut();
            module.declared_globals.insert(c_name.clone());
            module.global_decls.push(decl);
        }
        let ptr_ty = self.type_ptr();
        let val = self.intern_value(
            CValueKind::Global {
                name: format!("&{c_name}"),
            },
            ptr_ty,
        );
        self.statics_cache.borrow_mut().insert(def_id, val);
    }
}
