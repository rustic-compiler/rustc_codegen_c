/// C value representation for the codegen backend.
///
/// Values are interned via `ValueRef` indices into a `ValueStore`.
use crate::types::TypeRef;

/// An interned reference to a C value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValueRef(pub u32);

/// The kind of C value.
#[derive(Clone, Debug)]
pub enum CValueKind {
    /// Function parameter: `_arg{index}`.
    Param { index: usize },
    /// Local temporary variable: `_t{id}`.
    Temp { id: u32 },
    /// Integer constant.
    IntConst(i128),
    /// Unsigned integer constant.
    UintConst(u128),
    /// Float constant.
    FloatConst(f64),
    /// Boolean constant.
    BoolConst(bool),
    /// Null pointer.
    NullPtr,
    /// Undefined value.
    Undef,
    /// Poison (treated as undef).
    Poison,
    /// A reference to a global variable.
    Global { name: String },
    /// A reference to a function. `sig` stores the function signature type.
    Function { name: String, sig: TypeRef },
    /// A struct constant with element values.
    StructConst { fields: Vec<ValueRef> },
    /// A vector/array constant (rendered as a compound literal).
    VectorConst {
        elements: Vec<ValueRef>,
        type_str: String,
    },
    /// A byte string constant (for `static_addr_of`).
    ByteString { data: Vec<u8>, id: u32 },
    /// A string literal: (data_ptr, len).
    StringLiteral { data: String, id: u32 },
    /// The result of a const_ptr_byte_offset.
    PtrOffset { base: ValueRef, offset: u64 },
    /// Inline C expression (for computed expressions that aren't assigned to a temp).
    InlineExpr(String),
}

/// Information stored per value.
#[derive(Clone, Debug)]
pub struct CValueInfo {
    pub kind: CValueKind,
    pub ty: TypeRef,
}

/// Storage for interned values.
#[derive(Debug)]
pub struct ValueStore {
    values: Vec<CValueInfo>,
    temp_counter: u32,
    byte_string_counter: u32,
    string_literal_counter: u32,
}

impl ValueStore {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            temp_counter: 0,
            byte_string_counter: 0,
            string_literal_counter: 0,
        }
    }

    /// Allocate a new value. Does not deduplicate; each call creates a distinct ValueRef.
    pub fn alloc(&mut self, kind: CValueKind, ty: TypeRef) -> ValueRef {
        let idx = self.values.len();
        self.values.push(CValueInfo { kind, ty });
        ValueRef(idx as u32)
    }

    pub fn next_temp(&mut self, ty: TypeRef) -> ValueRef {
        let id = self.temp_counter;
        self.temp_counter += 1;
        self.alloc(CValueKind::Temp { id }, ty)
    }

    pub fn next_byte_string(&mut self, data: Vec<u8>, ty: TypeRef) -> ValueRef {
        let id = self.byte_string_counter;
        self.byte_string_counter += 1;
        self.alloc(CValueKind::ByteString { data, id }, ty)
    }

    pub fn next_string_literal(&mut self, data: String, ty: TypeRef) -> ValueRef {
        let id = self.string_literal_counter;
        self.string_literal_counter += 1;
        self.alloc(CValueKind::StringLiteral { data, id }, ty)
    }

    pub fn byte_string_counter(&self) -> u32 {
        self.byte_string_counter
    }

    pub fn get(&self, v: ValueRef) -> &CValueInfo {
        &self.values[v.0 as usize]
    }

    pub fn get_type(&self, v: ValueRef) -> TypeRef {
        self.values[v.0 as usize].ty
    }

    /// Get the function signature type if this value is a function reference.
    pub fn get_fn_sig(&self, v: ValueRef) -> Option<TypeRef> {
        match &self.values[v.0 as usize].kind {
            CValueKind::Function { sig, .. } => Some(*sig),
            _ => None,
        }
    }

    /// Render a value as a C expression.
    pub fn render(&self, v: ValueRef) -> String {
        let info = &self.values[v.0 as usize];
        match &info.kind {
            CValueKind::Param { index } => format!("_arg{index}"),
            CValueKind::Temp { id } => format!("_t{id}"),
            CValueKind::IntConst(i) => {
                if *i > i64::MIN as i128 && *i <= i64::MAX as i128 {
                    // Fits in long long. Excludes i64::MIN because the
                    // literal 9223372036854775808LL overflows long long in
                    // clang (the minus sign is applied after parsing).
                    format!("{i}LL")
                } else if *i == i64::MIN as i128 {
                    "(-9223372036854775807LL - 1LL)".to_string()
                } else {
                    // Split into hi/lo 64-bit parts for 128-bit values
                    let bits = *i as u128;
                    let lo = (bits & u64::MAX as u128) as u64;
                    let hi = (bits >> 64) as u64;
                    format!("((int128_t)((uint128_t){hi}ULL << 64 | {lo}ULL))")
                }
            }
            CValueKind::UintConst(u) => {
                if *u <= u64::MAX as u128 {
                    format!("{u}ULL")
                } else {
                    // Split into hi/lo 64-bit parts for 128-bit values
                    let lo = (*u & u64::MAX as u128) as u64;
                    let hi = (*u >> 64) as u64;
                    format!("((uint128_t){hi}ULL << 64 | {lo}ULL)")
                }
            }
            CValueKind::FloatConst(f) => {
                if f.is_nan() {
                    "NAN".into()
                } else if f.is_infinite() {
                    if *f > 0.0 { "INFINITY" } else { "(-INFINITY)" }.into()
                } else {
                    // Ensure float literals always have a decimal point so C
                    // interprets them as floating-point, not integer.
                    let s = format!("{f}");
                    if s.contains('.') || s.contains('e') || s.contains('E') {
                        s
                    } else {
                        format!("{s}.0")
                    }
                }
            }
            CValueKind::BoolConst(b) => if *b { "1" } else { "0" }.into(),
            CValueKind::NullPtr => "0".into(),
            CValueKind::Undef | CValueKind::Poison => "0 /* undef */".into(),
            CValueKind::Global { name } => name.clone(),
            CValueKind::Function { name, .. } => name.clone(),
            CValueKind::StructConst { fields } => {
                let elts: Vec<_> = fields.iter().map(|f| self.render(*f)).collect();
                format!("{{ {} }}", elts.join(", "))
            }
            CValueKind::VectorConst { elements, type_str } => {
                let elts: Vec<_> = elements.iter().map(|e| self.render(*e)).collect();
                // Struct-based vector: ({type}){{ .v = { e1, e2, ... } }}
                format!("({type_str}){{ .v = {{ {} }} }}", elts.join(", "))
            }
            CValueKind::ByteString { id, .. } => format!("_bytes{id}"),
            CValueKind::StringLiteral { id, .. } => format!("_str{id}"),
            CValueKind::PtrOffset { base, offset } => {
                format!("((char *){} + {})", self.render(*base), offset)
            }
            CValueKind::InlineExpr(expr) => expr.clone(),
        }
    }

    /// Try to extract a u64 constant from a value.
    pub fn as_u64(&self, v: ValueRef) -> Option<u64> {
        match &self.values[v.0 as usize].kind {
            CValueKind::UintConst(u) => u64::try_from(*u).ok(),
            CValueKind::IntConst(i) if *i >= 0 => u64::try_from(*i).ok(),
            CValueKind::BoolConst(b) => Some(*b as u64),
            _ => None,
        }
    }

    /// Try to extract a u128 constant from a value.
    pub fn as_u128(&self, v: ValueRef, sign_ext: bool) -> Option<u128> {
        match &self.values[v.0 as usize].kind {
            CValueKind::UintConst(u) => Some(*u),
            CValueKind::IntConst(i) => {
                if sign_ext {
                    Some(*i as u128)
                } else if *i >= 0 {
                    Some(*i as u128)
                } else {
                    None
                }
            }
            CValueKind::BoolConst(b) => Some(*b as u128),
            _ => None,
        }
    }
}
