/// The C module: collects all generated C code for a single codegen unit.
///
/// After codegen completes, the module is serialized to a `.c` file and
/// compiled with a system C compiler to produce an object file.
use std::collections::BTreeMap;
use std::fmt::Write;

use crate::c_ast::{CStmt, PrettyPrinter};
use crate::types::{TypeRef, TypeStore};
use crate::values::ValueStore;

/// A basic block within a function.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BasicBlockId(pub u32);

/// Data for a basic block.
#[derive(Debug)]
pub struct BasicBlockData {
    pub label: String,
    pub statements: Vec<CStmt>,
    /// Whether this block has a terminator (return, goto, etc.)
    pub terminated: bool,
}

/// A function definition being built.
#[derive(Debug)]
pub struct FunctionDef {
    pub name: String,
    pub return_type: TypeRef,
    pub params: Vec<(TypeRef, String)>,
    pub blocks: BTreeMap<u32, BasicBlockData>,
    pub block_counter: u32,
    /// Declarations of local temporary variables (type, name).
    pub local_decls: Vec<String>,
    /// Linkage prefix for this function (e.g., "static ", "__attribute__((weak)) ").
    pub linkage_prefix: String,
    /// Whether this function uses indirect return (codegen_ssa passes out
    /// pointer as first arg, but our C signature doesn't include it).
    pub has_indirect_ret: bool,
    /// The actual data type for indirect returns (the type of the value
    /// written through the out pointer, as opposed to `return_type` which
    /// is Void for indirect return functions).
    pub ret_data_type: Option<TypeRef>,
    /// Name of the return buffer variable for indirect returns.
    pub retbuf_name: Option<String>,
    /// Counter for unique invoke context variable names.
    pub invoke_counter: u32,
    /// Override the C type string for specific parameters (by index).
    /// Used for `main(int, char **)` where the second param must be
    /// `char **` rather than the generic `void *`.
    pub param_type_overrides: BTreeMap<usize, String>,
    /// Whether this function is C-variadic.
    pub is_variadic: bool,
}

impl FunctionDef {
    pub fn new(name: String, return_type: TypeRef, params: Vec<(TypeRef, String)>) -> Self {
        Self {
            name,
            return_type,
            params,
            blocks: BTreeMap::new(),
            block_counter: 0,
            local_decls: Vec::new(),
            linkage_prefix: "__attribute__((weak)) ".to_string(),
            has_indirect_ret: false,
            ret_data_type: None,
            retbuf_name: None,
            invoke_counter: 0,
            param_type_overrides: BTreeMap::new(),
            is_variadic: false,
        }
    }

    pub fn new_block(&mut self, label: String) -> BasicBlockId {
        let id = self.block_counter;
        self.block_counter += 1;
        self.blocks.insert(
            id,
            BasicBlockData {
                label,
                statements: Vec::new(),
                terminated: false,
            },
        );
        BasicBlockId(id)
    }

    pub fn emit(&mut self, bb: BasicBlockId, stmt: CStmt) {
        if let Some(block) = self.blocks.get_mut(&bb.0) {
            block.statements.push(stmt);
        }
    }

    pub fn set_terminated(&mut self, bb: BasicBlockId) {
        if let Some(block) = self.blocks.get_mut(&bb.0) {
            block.terminated = true;
        }
    }

    pub fn add_local_decl(&mut self, decl: String) {
        self.local_decls.push(decl);
    }

    /// Render the function definition as C code.
    pub fn render(&self, types: &TypeStore) -> String {
        let mut s = String::new();

        // Signature
        let params_str: Vec<_> = self
            .params
            .iter()
            .enumerate()
            .map(|(i, (ty, name))| {
                if let Some(override_ty) = self.param_type_overrides.get(&i) {
                    format!("{override_ty} {name}")
                } else {
                    types.render_decl(*ty, name)
                }
            })
            .collect();
        let mut params_joined = if params_str.is_empty() {
            if self.is_variadic { "".to_string() } else { "void".to_string() }
        } else {
            params_str.join(", ")
        };
        if self.is_variadic {
            if !params_joined.is_empty() {
                params_joined.push_str(", ");
            }
            params_joined.push_str("...");
        }

        let _ = writeln!(
            s,
            "{}{} {}({}) {{",
            self.linkage_prefix,
            types.render(self.return_type),
            self.name,
            params_joined
        );

        // Local variable declarations
        for decl in &self.local_decls {
            let _ = writeln!(s, "  {decl}");
        }
        if !self.local_decls.is_empty() {
            s.push('\n');
        }

        // Basic blocks
        for (_, block) in &self.blocks {
            let _ = writeln!(s, "{}:", block.label);
            let pp = PrettyPrinter::with_indent(&block.statements, 1);
            s.push_str(&pp.to_string());
            // If not terminated, fall through (add explicit label comment)
            if !block.terminated {
                let _ = writeln!(s, "  ; /* fallthrough */");
            }
        }

        s.push_str("}\n");
        s
    }
}

/// The C module output.
pub struct CModule {
    pub name: String,
    /// Type store -- populated after codegen from CodegenCx.
    pub types: TypeStore,
    /// Value store -- populated after codegen from CodegenCx.
    pub values: ValueStore,

    /// Forward struct declarations.
    pub struct_defs: Vec<String>,
    /// Extern global variable declarations (rendered early, before data sections).
    pub global_extern_decls: Vec<String>,
    /// Global variable definitions (rendered after data sections).
    pub global_decls: Vec<String>,
    /// Function forward declarations.
    pub function_decls: Vec<String>,
    /// Set of declared function names (to avoid duplicates).
    pub declared_fns: std::collections::BTreeSet<String>,
    /// Set of defined global variable names (to suppress extern declarations
    /// that would conflict with the definition's type).
    pub declared_globals: std::collections::BTreeSet<String>,
    /// Set of extern-declared global variable names (to avoid duplicates
    /// in `global_extern_decls`).
    pub declared_extern_globals: std::collections::BTreeSet<String>,
    /// Completed function definitions.
    pub function_defs: Vec<String>,
    /// Currently open function definitions (being built by codegen_mir).
    pub open_functions: BTreeMap<String, FunctionDef>,
    /// Reverse map: BasicBlockId -> function name for O(1) lookup.
    pub block_to_function: BTreeMap<u32, String>,
    /// Byte string data sections.
    pub data_sections: Vec<String>,
    /// Constructor functions (for patching relocations in const data).
    pub constructor_defs: Vec<String>,
    /// Pre-compiled C source (used for thin LTO pass-through).
    pub precompiled_source: Option<String>,
}

// SAFETY: CModule is only accessed through RefCell<CModule> in CodegenCx,
// which enforces single-threaded borrow semantics. All fields (String, Vec,
// BTreeMap, TypeStore, ValueStore) are composed of Send + Sync types. The
// auto-impl is blocked by the lack of auto-trait on `CModule` itself (due to
// it being used as WriteBackendMethods::Module, which requires explicit impls).
unsafe impl Send for CModule {}
unsafe impl Sync for CModule {}

impl CModule {
    pub fn new(name: String) -> Self {
        Self {
            name,
            types: TypeStore::new(),
            values: ValueStore::new(),
            struct_defs: Vec::new(),
            global_extern_decls: Vec::new(),
            global_decls: Vec::new(),
            function_decls: Vec::new(),
            declared_fns: [
                // Pre-populate with functions declared in the preamble
                // to prevent get_fn from emitting conflicting declarations.
                "memcpy", "memset", "memmove", "abort", "__rust_try",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            declared_globals: std::collections::BTreeSet::new(),
            declared_extern_globals: std::collections::BTreeSet::new(),
            function_defs: Vec::new(),
            open_functions: BTreeMap::new(),
            block_to_function: BTreeMap::new(),
            data_sections: Vec::new(),
            constructor_defs: Vec::new(),
            precompiled_source: None,
        }
    }

    /// Add an extern global declaration, deduplicating by name.
    /// These are rendered early (before data sections and static definitions).
    pub fn add_global_decl(&mut self, name: &str, decl: String) {
        if self.declared_extern_globals.insert(name.to_string()) {
            self.global_extern_decls.push(decl);
        }
    }

    /// Create a new FunctionDef (convenience for use from context.rs).
    pub fn new_function_def(
        name: String,
        return_type: TypeRef,
        params: Vec<(TypeRef, String)>,
    ) -> FunctionDef {
        FunctionDef::new(name, return_type, params)
    }

    /// Serialize the module to a complete C source file.
    pub fn to_c_source(&self) -> String {
        // If we have pre-compiled source (from thin LTO pass-through), use it
        if let Some(ref src) = self.precompiled_source {
            return src.clone();
        }
        let mut s = String::new();

        // Type-only headers. We avoid function-declaring headers (string.h,
        // stdlib.h, POSIX headers) because our generated forward declarations
        // may have slightly different signatures (e.g. int64_t vs size_t) and
        // would conflict.
        s.push_str("/* Generated by rustc_codegen_c */\n");
        s.push_str("#include <stdint.h>\n");
        s.push_str("#include <stdbool.h>\n");
        s.push_str("#include <stddef.h>\n");
        s.push_str("#include <math.h>\n");
        s.push_str("#include <stdarg.h>\n");
        s.push_str("#include <stdatomic.h>\n");
        s.push_str("void *memcpy(void *, const void *, size_t);\n");
        s.push_str("void *memset(void *, int, size_t);\n");
        s.push_str("void *memmove(void *, const void *, size_t);\n");
        s.push_str("int memcmp(const void *, const void *, size_t);\n");
        s.push_str("void abort(void);\n");
        // setjmp/longjmp-based unwind context for invoke/resume/catch_unwind.
        // Uses standard C setjmp/longjmp for full architecture portability.
        s.push_str("#include <setjmp.h>\n");
        s.push_str("#define __rustc_setjmp setjmp\n");
        s.push_str("#define __rustc_longjmp longjmp\n");
        s.push_str("struct __rustc_unwind_context {\n");
        s.push_str("  jmp_buf buf;\n");
        s.push_str("  void *exception_ptr;\n");
        s.push_str("  struct __rustc_unwind_context *prev;\n");
        s.push_str("};\n");
        // Weak definition (not extern): ensures binary crates that link
        // against a dylib can resolve this symbol from their own object
        // files.  Rust's linker export list only includes Rust-mangled
        // symbols, so a definition in the allocator module inside a .so
        // becomes local and invisible to the binary.  A weak definition
        // in every module lets the linker merge them into one per link
        // unit.
        s.push_str(
            "__attribute__((weak)) __thread struct __rustc_unwind_context *__rustc_unwind_chain;\n",
        );
        // Weak definition of __rust_try so that binary crates linking
        // dynamically against std can resolve this symbol from their own
        // object files. The allocator module emits a strong definition that
        // takes precedence when statically linked.
        s.push_str("__attribute__((weak)) int __rust_try(void (*try_fn)(void *), void *data, void (*catch_fn)(void *, void *)) {\n");
        s.push_str("  struct __rustc_unwind_context __ctx;\n");
        s.push_str("  __ctx.prev = __rustc_unwind_chain;\n");
        s.push_str("  __ctx.exception_ptr = (void *)0;\n");
        s.push_str("  __rustc_unwind_chain = &__ctx;\n");
        s.push_str("  if (__rustc_setjmp(__ctx.buf) == 0) {\n");
        s.push_str("    try_fn(data);\n");
        s.push_str("    __rustc_unwind_chain = __ctx.prev;\n");
        s.push_str("    return 0;\n");
        s.push_str("  } else {\n");
        s.push_str("    void *__exn = __ctx.exception_ptr;\n");
        s.push_str("    __rustc_unwind_chain = __ctx.prev;\n");
        s.push_str("    catch_fn(data, __exn);\n");
        s.push_str("    return 1;\n");
        s.push_str("  }\n");
        s.push_str("}\n\n");

        // 128-bit integer support (GCC/Clang extension)
        s.push_str("#ifdef __SIZEOF_INT128__\n");
        s.push_str("typedef __int128 int128_t;\n");
        s.push_str("typedef unsigned __int128 uint128_t;\n");
        s.push_str("#else\n");
        s.push_str("#error \"128-bit integer support requires __int128 (GCC/Clang)\"\n");
        s.push_str("#endif\n\n");

        // _Float16 / __float128 portability.
        // Clang may support these as builtin types without defining the
        // corresponding __FLT*_MAX__ macros. Use __is_identifier() (a
        // clang builtin) to check whether the name is already a keyword.
        s.push_str("#ifndef __FLT16_MAX__\n");
        s.push_str("#ifdef __clang__\n");
        s.push_str("#if __is_identifier(_Float16)\n");
        s.push_str("typedef unsigned short _Float16; /* fallback: no hardware f16 */\n");
        s.push_str("#endif\n");
        s.push_str("#else\n");
        s.push_str("typedef unsigned short _Float16; /* fallback: no hardware f16 */\n");
        s.push_str("#endif\n");
        s.push_str("#endif\n");
        s.push_str("#ifndef __FLT128_MAX__\n");
        s.push_str("#ifdef __clang__\n");
        s.push_str("#if __is_identifier(_Float128)\n");
        s.push_str("typedef long double _Float128; /* fallback: reduced precision */\n");
        s.push_str("#endif\n");
        s.push_str("#else\n");
        s.push_str("/* GCC: _Float128 is a builtin on x86_64; on other arches, fall back */\n");
        s.push_str("#ifndef __SIZEOF_FLOAT128__\n");
        s.push_str("#if !defined(__x86_64__) && !defined(__i386__)\n");
        s.push_str("typedef long double _Float128; /* fallback: reduced precision */\n");
        s.push_str("#endif\n");
        s.push_str("#endif\n");
        s.push_str("#endif\n");
        s.push_str("#endif\n\n");

        // MSVC-compatible fallbacks for GCC/Clang builtins
        s.push_str("#ifdef _MSC_VER\n");
        s.push_str("#include <intrin.h>\n");
        s.push_str("#define __builtin_unreachable() __assume(0)\n");
        s.push_str("#define __builtin_expect(expr, val) (expr)\n");
        s.push_str("#define __builtin_isnan(x) _isnan(x)\n");
        s.push_str("#endif\n\n");

        // Struct type definitions (auto-generated from TypeStore)
        for def in self.types.render_struct_defs() {
            s.push_str(&def);
            s.push('\n');
        }
        for def in &self.struct_defs {
            s.push_str(def);
            s.push('\n');
        }
        if !self.struct_defs.is_empty() {
            s.push('\n');
        }

        // Extern global declarations (before data sections and static definitions)
        for decl in &self.global_extern_decls {
            s.push_str(decl);
            s.push('\n');
        }
        if !self.global_extern_decls.is_empty() {
            s.push('\n');
        }

        // Function forward declarations.
        // Skip declarations whose signature conflicts with a definition in
        // this module (can happen for intrinsic fallback bodies where
        // fn_abi differs between the intrinsic and its fallback).
        let defined_sigs: std::collections::BTreeMap<&str, usize> = self
            .function_defs
            .iter()
            .filter_map(|def| {
                let paren = def.find('(')?;
                let before = &def[..paren];
                let name_start = before.rfind(|c: char| c.is_whitespace())? + 1;
                let name = &before[name_start..];
                let after_paren = &def[paren..];
                let close = after_paren.find(')')?;
                let params = &after_paren[1..close];
                let count = if params.trim() == "void" || params.trim().is_empty() {
                    0
                } else {
                    params.split(',').count()
                };
                Some((name, count))
            })
            .collect();
        for decl in &self.function_decls {
            let skip = (|| {
                let paren = decl.find('(')?;
                let before = &decl[..paren];
                let name_start = before.rfind(|c: char| c.is_whitespace())? + 1;
                let name = &before[name_start..];
                let after_paren = &decl[paren..];
                let close = after_paren.find(')')?;
                let params = &after_paren[1..close];
                let decl_count = if params.trim() == "void" || params.trim().is_empty() {
                    0
                } else {
                    params.split(',').count()
                };
                if let Some(&def_count) = defined_sigs.get(name) {
                    if def_count != decl_count {
                        return Some(true); // conflicting signature
                    }
                }
                Some(false)
            })()
            .unwrap_or(false);
            if !skip {
                s.push_str(decl);
                s.push('\n');
            }
        }
        if !self.function_decls.is_empty() {
            s.push('\n');
        }

        // Data sections (before global decls so _bytes constants are available
        // for static initializers)
        for data in &self.data_sections {
            s.push_str(data);
            s.push('\n');
        }
        if !self.data_sections.is_empty() {
            s.push('\n');
        }

        // Global declarations (statics -- may reference _bytes constants above)
        for decl in &self.global_decls {
            s.push_str(decl);
            s.push('\n');
        }
        if !self.global_decls.is_empty() {
            s.push('\n');
        }

        // Function definitions
        for def in &self.function_defs {
            s.push_str(def);
            s.push('\n');
        }

        // Constructor functions (emitted after all declarations to resolve symbols)
        for def in &self.constructor_defs {
            s.push_str(def);
            s.push('\n');
        }

        s
    }

    /// Finalize an open function and move it to completed definitions.
    pub fn finalize_function(&mut self, name: &str, types: &TypeStore) {
        if let Some(func) = self.open_functions.remove(name) {
            let rendered = func.render(types);
            self.function_defs.push(rendered);
        }
    }
}

/// Buffer for serialized modules (for LTO, which we don't support meaningfully).
pub struct CModuleBuffer {
    data: Vec<u8>,
}

impl CModuleBuffer {
    pub fn new(source: &str) -> Self {
        Self {
            data: source.as_bytes().to_vec(),
        }
    }
}

impl rustc_codegen_ssa::traits::ModuleBufferMethods for CModuleBuffer {
    fn data(&self) -> &[u8] {
        &self.data
    }
}
