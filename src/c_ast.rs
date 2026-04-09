/// C Abstract Syntax Tree types for the codegen backend.
///
/// Instead of generating C code as strings directly, the codegen backend
/// constructs AST nodes which are then rendered to C source code via
/// [`PrettyPrinter`].
use std::fmt::{self, Write};

/// Binary operator in C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Shl,
    Shr,
    BitAnd,
    BitOr,
    BitXor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LogAnd,
    LogOr,
}

impl fmt::Display for CBinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Rem => "%",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::BitAnd => "&",
            Self::BitOr => "|",
            Self::BitXor => "^",
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::LogAnd => "&&",
            Self::LogOr => "||",
        })
    }
}

/// Unary operator in C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CUnaryOp {
    /// `-`
    Neg,
    /// `~`
    BitNot,
    /// `!`
    LogNot,
}

impl fmt::Display for CUnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Neg => "-",
            Self::BitNot => "~",
            Self::LogNot => "!",
        })
    }
}

/// A C expression AST node.
#[derive(Clone, Debug)]
pub enum CExpr {
    /// Variable or identifier: `name`
    Var(String),
    /// Pre-rendered literal: `42LL`, `0ULL`, `NAN`, etc.
    Lit(String),
    /// Binary operation: `lhs op rhs`
    BinOp(Box<CExpr>, CBinOp, Box<CExpr>),
    /// Unary prefix operation: `op(expr)`
    UnaryOp(CUnaryOp, Box<CExpr>),
    /// Type cast: `(ty)expr`
    Cast(String, Box<CExpr>),
    /// Pointer dereference with cast: `*(cast_ty)ptr`
    Deref(String, Box<CExpr>),
    /// Address-of: `&expr`
    AddrOf(Box<CExpr>),
    /// Function/macro call: `func(args...)`
    Call(Box<CExpr>, Vec<CExpr>),
    /// Struct field access: `expr.field`
    Field(Box<CExpr>, String),
    /// Array subscript: `expr[idx]`
    Index(Box<CExpr>, Box<CExpr>),
    /// Ternary conditional: `(cond) ? (then) : (else)`
    Ternary(Box<CExpr>, Box<CExpr>, Box<CExpr>),
    /// `sizeof(expr)`
    Sizeof(Box<CExpr>),
    /// `sizeof(type)`
    SizeofType(String),
    /// Compound literal: `(type){ e1, e2, ... }`
    CompoundLiteral(String, Vec<CExpr>),
    /// Parenthesized expression: `(expr)`
    Paren(Box<CExpr>),
    /// Raw C expression string (fallback for complex cases).
    Raw(String),
}

// -- Convenience constructors --

impl CExpr {
    pub fn var(s: impl Into<String>) -> Self {
        Self::Var(s.into())
    }

    pub fn lit(s: impl Into<String>) -> Self {
        Self::Lit(s.into())
    }

    pub fn raw(s: impl Into<String>) -> Self {
        Self::Raw(s.into())
    }

    pub fn binop(lhs: Self, op: CBinOp, rhs: Self) -> Self {
        Self::BinOp(Box::new(lhs), op, Box::new(rhs))
    }

    pub fn unary(op: CUnaryOp, expr: Self) -> Self {
        Self::UnaryOp(op, Box::new(expr))
    }

    pub fn cast(ty: impl Into<String>, expr: Self) -> Self {
        Self::Cast(ty.into(), Box::new(expr))
    }

    pub fn deref(cast_ty: impl Into<String>, ptr: Self) -> Self {
        Self::Deref(cast_ty.into(), Box::new(ptr))
    }

    pub fn addr_of(expr: Self) -> Self {
        Self::AddrOf(Box::new(expr))
    }

    pub fn call(func: Self, args: Vec<Self>) -> Self {
        Self::Call(Box::new(func), args)
    }

    pub fn field(expr: Self, name: impl Into<String>) -> Self {
        Self::Field(Box::new(expr), name.into())
    }

    pub fn index(expr: Self, idx: Self) -> Self {
        Self::Index(Box::new(expr), Box::new(idx))
    }

    pub fn ternary(cond: Self, then_expr: Self, else_expr: Self) -> Self {
        Self::Ternary(Box::new(cond), Box::new(then_expr), Box::new(else_expr))
    }

    pub fn sizeof_expr(expr: Self) -> Self {
        Self::Sizeof(Box::new(expr))
    }

    pub fn sizeof_ty(ty: impl Into<String>) -> Self {
        Self::SizeofType(ty.into())
    }

    pub fn paren(expr: Self) -> Self {
        Self::Paren(Box::new(expr))
    }
}

impl fmt::Display for CExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(name) => write!(f, "{name}"),
            Self::Lit(s) => write!(f, "{s}"),
            Self::BinOp(lhs, op, rhs) => write!(f, "{lhs} {op} {rhs}"),
            Self::UnaryOp(op, expr) => write!(f, "{op}({expr})"),
            Self::Cast(ty, expr) => write!(f, "({ty}){expr}"),
            Self::Deref(cast_ty, ptr) => write!(f, "*({cast_ty}){ptr}"),
            Self::AddrOf(expr) => write!(f, "&{expr}"),
            Self::Call(func, args) => {
                write!(f, "{func}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::Field(expr, name) => write!(f, "{expr}.{name}"),
            Self::Index(expr, idx) => write!(f, "{expr}[{idx}]"),
            Self::Ternary(cond, then_expr, else_expr) => {
                write!(f, "({cond}) ? ({then_expr}) : ({else_expr})")
            }
            Self::Sizeof(expr) => write!(f, "sizeof({expr})"),
            Self::SizeofType(ty) => write!(f, "sizeof({ty})"),
            Self::CompoundLiteral(ty, elts) => {
                write!(f, "({ty}){{ ")?;
                for (i, elt) in elts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{elt}")?;
                }
                write!(f, " }}")
            }
            Self::Paren(expr) => write!(f, "({expr})"),
            Self::Raw(s) => write!(f, "{s}"),
        }
    }
}

/// A C statement AST node.
#[derive(Clone, Debug)]
pub enum CStmt {
    /// Assignment: `target = value;`
    Assign(CExpr, CExpr),
    /// Expression statement: `expr;`
    Expr(CExpr),
    /// Return: `return expr;` or `return;`
    Return(Option<CExpr>),
    /// Goto: `goto label;`
    Goto(String),
    /// Conditional goto: `if (cond) goto then_label; else goto else_label;`
    CondGoto {
        cond: CExpr,
        then_label: String,
        else_label: String,
    },
    /// Switch with goto cases.
    Switch {
        expr: CExpr,
        cases: Vec<(CExpr, String)>,
        default: String,
    },
    /// `__builtin_unreachable();`
    Unreachable,
    /// `do { body } while (cond);`
    DoWhile { body: Vec<CStmt>, cond: CExpr },
    /// Fallthrough marker: `; /* fallthrough */`
    Fallthrough,
    /// Raw C statement string (fallback for complex constructs).
    Raw(String),
}

// -- Convenience constructors --

impl CStmt {
    pub fn assign(target: CExpr, value: CExpr) -> Self {
        Self::Assign(target, value)
    }

    pub fn expr(e: CExpr) -> Self {
        Self::Expr(e)
    }

    pub fn ret(v: Option<CExpr>) -> Self {
        Self::Return(v)
    }

    pub fn ret_val(v: CExpr) -> Self {
        Self::Return(Some(v))
    }

    pub fn ret_void() -> Self {
        Self::Return(None)
    }

    pub fn goto(label: impl Into<String>) -> Self {
        Self::Goto(label.into())
    }

    pub fn raw(s: impl Into<String>) -> Self {
        Self::Raw(s.into())
    }
}

// =====================================================================
// PrettyPrinter
// =====================================================================

/// Trait for AST nodes that can be formatted by [`PrettyPrinter`].
pub trait CFormat {
    fn fmt_c(&self, pp: &mut PrettyPrinter);
}

/// Pretty-prints C AST nodes with proper indentation.
///
/// # Examples
///
/// ```ignore
/// let s = PrettyPrinter::new(&stmt).to_string();
/// let s = PrettyPrinter::with_indent(&stmts, 1).to_string();
/// ```
pub struct PrettyPrinter {
    buf: String,
    indent: usize,
}

impl PrettyPrinter {
    /// Format an AST node starting at indent level 0.
    pub fn new(node: &impl CFormat) -> Self {
        Self::with_indent(node, 0)
    }

    /// Format an AST node starting at the given indent level.
    pub fn with_indent(node: &impl CFormat, indent: usize) -> Self {
        let mut pp = Self {
            buf: String::new(),
            indent,
        };
        node.fmt_c(&mut pp);
        pp
    }

    /// Return the formatted output, consuming the printer.
    pub fn finish(self) -> String {
        self.buf
    }

    // -- low-level helpers --

    fn push_indent(&mut self) {
        for _ in 0..self.indent {
            self.buf.push_str("  ");
        }
    }

    fn push_newline(&mut self) {
        self.buf.push('\n');
    }

    /// Write an indented line (indent + text + newline).
    fn line(&mut self, text: &str) {
        self.push_indent();
        self.buf.push_str(text);
        self.push_newline();
    }

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        self.indent = self.indent.saturating_sub(1);
    }

    // -- public formatting entry points --

    /// Format an expression inline (no indentation / newline).
    pub fn fmt_expr(&mut self, expr: &CExpr) {
        // Expressions are single-line; delegate to Display.
        let _ = write!(self.buf, "{expr}");
    }

    /// Format a statement with indentation and trailing newline.
    pub fn fmt_stmt(&mut self, stmt: &CStmt) {
        match stmt {
            CStmt::Assign(target, value) => {
                self.push_indent();
                let _ = write!(self.buf, "{target} = {value};");
                self.push_newline();
            }
            CStmt::Expr(expr) => {
                self.push_indent();
                let _ = write!(self.buf, "{expr};");
                self.push_newline();
            }
            CStmt::Return(None) => self.line("return;"),
            CStmt::Return(Some(expr)) => {
                self.push_indent();
                let _ = write!(self.buf, "return {expr};");
                self.push_newline();
            }
            CStmt::Goto(label) => {
                self.push_indent();
                let _ = write!(self.buf, "goto {label};");
                self.push_newline();
            }
            CStmt::CondGoto {
                cond,
                then_label,
                else_label,
            } => {
                self.push_indent();
                let _ = write!(
                    self.buf,
                    "if ({cond}) goto {then_label}; else goto {else_label};"
                );
                self.push_newline();
            }
            CStmt::Switch {
                expr,
                cases,
                default,
            } => {
                self.push_indent();
                let _ = write!(self.buf, "switch ({expr}) {{");
                self.push_newline();
                self.indent();
                for (val, label) in cases {
                    self.push_indent();
                    let _ = write!(self.buf, "case {val}: goto {label};");
                    self.push_newline();
                }
                self.push_indent();
                let _ = write!(self.buf, "default: goto {default};");
                self.push_newline();
                self.dedent();
                self.line("}");
            }
            CStmt::Unreachable => self.line("__builtin_unreachable();"),
            CStmt::DoWhile { body, cond } => {
                self.line("do {");
                self.indent();
                for s in body {
                    self.fmt_stmt(s);
                }
                self.dedent();
                self.push_indent();
                let _ = write!(self.buf, "}} while ({cond});");
                self.push_newline();
            }
            CStmt::Fallthrough => self.line("; /* fallthrough */"),
            CStmt::Raw(s) => {
                for raw_line in s.lines() {
                    self.push_indent();
                    self.buf.push_str(raw_line);
                    self.push_newline();
                }
                // Single-line raw strings without a newline are common;
                // multi-line ones ending with \n already handled by lines().
            }
        }
    }
}

impl fmt::Display for PrettyPrinter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.buf)
    }
}

// -- CFormat impls --

impl CFormat for CExpr {
    fn fmt_c(&self, pp: &mut PrettyPrinter) {
        pp.fmt_expr(self);
    }
}

impl CFormat for CStmt {
    fn fmt_c(&self, pp: &mut PrettyPrinter) {
        pp.fmt_stmt(self);
    }
}

impl CFormat for [CStmt] {
    fn fmt_c(&self, pp: &mut PrettyPrinter) {
        for stmt in self {
            pp.fmt_stmt(stmt);
        }
    }
}

impl CFormat for Vec<CStmt> {
    fn fmt_c(&self, pp: &mut PrettyPrinter) {
        self.as_slice().fmt_c(pp);
    }
}
