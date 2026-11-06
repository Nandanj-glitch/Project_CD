#!/usr/bin/env python3
"""
minicc - a small C compiler written in Python (compiler design project)

Pipeline
    1. Lexical analysis     source text   -> tokens
    2. Syntax analysis      tokens        -> AST        (recursive descent)
    3. Semantic analysis    AST           -> checked AST (symbol tables, scopes)
    4. Optimization         AST           -> AST         (constant folding, dead branches)
       (+ --tac prints three-address code, --symbols prints the symbol table)
    5. Code generation      AST           -> x86-64 assembly (AT&T, System V ABI)

Supported C subset
    - types        : int, void (function return only)
    - declarations : int a, b = 5;
    - statements   : block, if/else, while, do-while, for, return, break, continue
    - expressions  : = += -= *= /= %=   || &&   == !=   < > <= >=
                     + - * / %   & | ^ << >>   unary - ! ~ +   ++ -- (prefix/postfix)
    - functions    : definitions, prototypes, recursion, up to 6 parameters
    - literals     : integers, 'c' characters, "strings" (only as printf args)
    - printf       : built-in, linked from libc
    - comments     : // and /* */ ; preprocessor lines (#include) are ignored

Usage
    python minicc.py prog.c            # writes prog.s and builds ./prog (needs gcc)
    python minicc.py prog.c --run      # build and run
    python minicc.py prog.c -S         # only generate prog.s
    python minicc.py prog.c --tokens --ast --tac --symbols --no-opt
Targets: Linux x86-64 and 32-bit Windows MinGW (selected from the installed GCC).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------
class CompileError(Exception):
    def __init__(self, msg, line=0):
        super().__init__(msg)
        self.msg = msg
        self.line = line


# ----------------------------------------------------------------------------
# 1. LEXER
# ----------------------------------------------------------------------------
KEYWORDS = {"int", "void", "if", "else", "while", "for", "return", "break", "continue", "do"}

TOKEN_RE = re.compile(r'''
    (?P<WS>[ \t\r]+)
  | (?P<NL>\n)
  | (?P<LCOMMENT>//[^\n]*)
  | (?P<BCOMMENT>/\*.*?\*/)
  | (?P<PRE>\#[^\n]*)
  | (?P<NUM>\d+)
  | (?P<CHAR>'(?:\\.|[^\\'])')
  | (?P<STR>"(?:\\.|[^\\"\n])*")
  | (?P<ID>[A-Za-z_]\w*)
  | (?P<OP>\+\+|--|\+=|-=|\*=|/=|%=|==|!=|<=|>=|<<|>>|&&|\|\||[-+*/%=<>!~&|^(){};,])
''', re.VERBOSE | re.DOTALL)

ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}


def unescape(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append(ESCAPES.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


class Token:
    def __init__(self, kind, value, line):
        self.kind, self.value, self.line = kind, value, line

    def __repr__(self):
        return f"{self.kind:<4} {self.value!r:<12} (line {self.line})"


def tokenize(src):
    tokens, pos, line = [], 0, 1
    while pos < len(src):
        m = TOKEN_RE.match(src, pos)
        if not m:
            raise CompileError(f"unexpected character {src[pos]!r}", line)
        kind, text = m.lastgroup, m.group()
        if kind == "NL":
            line += 1
        elif kind == "BCOMMENT":
            line += text.count("\n")
        elif kind in ("WS", "LCOMMENT", "PRE"):
            pass
        elif kind == "NUM":
            tokens.append(Token("NUM", int(text), line))
        elif kind == "CHAR":
            tokens.append(Token("NUM", ord(unescape(text[1:-1])), line))
        elif kind == "STR":
            tokens.append(Token("STR", unescape(text[1:-1]), line))
        elif kind == "ID":
            tokens.append(Token("KW" if text in KEYWORDS else "ID", text, line))
        else:
            tokens.append(Token("OP", text, line))
        pos = m.end()
    tokens.append(Token("EOF", "", line))
    return tokens


# ----------------------------------------------------------------------------
# AST
# ----------------------------------------------------------------------------
class Node:
    """Generic AST node: a 'kind' string plus arbitrary named fields."""

    def __init__(self, kind, line=0, **fields):
        self.kind = kind
        self.line = line
        self.__dict__.update(fields)


def dump(n, indent=0):
    pad = "  " * indent
    if isinstance(n, list):
        for x in n:
            dump(x, indent)
    elif isinstance(n, Node):
        scalars, children = [], []
        for k, v in vars(n).items():
            if k in ("kind", "line"):
                continue
            if isinstance(v, Node) or (isinstance(v, list) and any(isinstance(x, Node) for x in v)):
                children.append((k, v))
            else:
                scalars.append(f"{k}={v!r}")
        print(f"{pad}{n.kind}" + (f"  [{', '.join(scalars)}]" if scalars else ""))
        for k, v in children:
            print(f"{pad}  {k}:")
            dump(v, indent + 2)


# ----------------------------------------------------------------------------
# 2. PARSER (recursive descent)
#
#   program    -> function*
#   function   -> ("int"|"void") ID "(" params ")" (block | ";")
#   statement  -> block | decl ";" | if | while | for | return | break | continue
#               | expr ";" | ";"
#   expr       -> assignment ; precedence climbing for binary operators
# ----------------------------------------------------------------------------
BINARY_LEVELS = [["||"], ["&&"], ["|"], ["^"], ["&"], ["==", "!="], ["<", ">", "<=", ">="],
                 ["<<", ">>"], ["+", "-"], ["*", "/", "%"]]
ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%="}


class Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    # -- helpers
    @property
    def cur(self):
        return self.t[self.i]

    def check(self, *vals):
        return self.cur.kind in ("OP", "KW") and self.cur.value in vals

    def accept(self, *vals):
        if self.check(*vals):
            tok = self.cur
            self.i += 1
            return tok
        return None

    def expect(self, val):
        tok = self.accept(val)
        if not tok:
            found = self.cur.value if self.cur.kind != "EOF" else "end of file"
            raise CompileError(f"expected '{val}' but found '{found}'", self.cur.line)
        return tok

    def expect_id(self):
        if self.cur.kind != "ID":
            found = self.cur.value if self.cur.kind != "EOF" else "end of file"
            raise CompileError(f"expected an identifier but found '{found}'", self.cur.line)
        tok = self.cur
        self.i += 1
        return tok.value

    # -- top level
    def program(self):
        funcs = []
        while self.cur.kind != "EOF":
            funcs.append(self.function())
        return Node("Program", funcs=funcs)

    def function(self):
        line = self.cur.line
        if not self.check("int", "void"):
            raise CompileError("expected 'int' or 'void' at start of a function definition "
                               "(global variables are not supported)", line)
        rtype = self.cur.value
        self.i += 1
        name = self.expect_id()
        self.expect("(")
        params = []
        if self.check("void") and self.t[self.i + 1].value == ")":
            self.i += 1
        elif not self.check(")"):
            while True:
                self.expect("int")
                params.append(self.expect_id())
                if not self.accept(","):
                    break
        self.expect(")")
        body = None if self.accept(";") else self.block()
        return Node("Func", line, name=name, rtype=rtype, params=params, body=body)

    # -- statements
    def block(self):
        line = self.expect("{").line
        stmts = []
        while not self.check("}"):
            if self.cur.kind == "EOF":
                raise CompileError("missing '}' at end of file", self.cur.line)
            stmts.append(self.statement())
        self.expect("}")
        return Node("Block", line, stmts=stmts)

    def declaration(self):
        line = self.cur.line
        self.expect("int")
        items = []
        while True:
            name = self.expect_id()
            init = self.expression() if self.accept("=") else None
            items.append(Node("Decl", line, name=name, init=init))
            if not self.accept(","):
                break
        return Node("Decls", line, items=items)

    def statement(self):
        line = self.cur.line
        if self.check("{"):
            return self.block()
        if self.check("int"):
            d = self.declaration()
            self.expect(";")
            return d
        if self.accept("if"):
            self.expect("(")
            cond = self.expression()
            self.expect(")")
            then = self.statement()
            els = self.statement() if self.accept("else") else None
            return Node("If", line, cond=cond, then=then, els=els)
        if self.accept("do"):
            body = self.statement()
            self.expect("while")
            self.expect("(")
            cond = self.expression()
            self.expect(")")
            self.expect(";")
            return Node("DoWhile", line, body=body, cond=cond)
        if self.accept("while"):
            self.expect("(")
            cond = self.expression()
            self.expect(")")
            return Node("While", line, cond=cond, body=self.statement())
        if self.accept("for"):
            self.expect("(")
            if self.check("int"):
                init = self.declaration()
            elif self.check(";"):
                init = None
            else:
                init = Node("ExprStmt", line, expr=self.expression())
            self.expect(";")
            cond = None if self.check(";") else self.expression()
            self.expect(";")
            step = None if self.check(")") else self.expression()
            self.expect(")")
            return Node("For", line, init=init, cond=cond, step=step, body=self.statement())
        if self.accept("return"):
            value = None if self.check(";") else self.expression()
            self.expect(";")
            return Node("Return", line, value=value)
        if self.accept("break"):
            self.expect(";")
            return Node("Break", line)
        if self.accept("continue"):
            self.expect(";")
            return Node("Continue", line)
        if self.accept(";"):
            return Node("Empty", line)
        e = self.expression()
        self.expect(";")
        return Node("ExprStmt", line, expr=e)

    # -- expressions
    def expression(self):
        left = self.binary(0)
        if self.cur.kind == "OP" and self.cur.value in ASSIGN_OPS:
            op = self.cur
            self.i += 1
            if left.kind != "Var":
                raise CompileError("left side of assignment must be a variable", op.line)
            right = self.expression()
            if op.value != "=":  # desugar  x += e  ->  x = x + e
                right = Node("Binary", op.line, op=op.value[0],
                             left=Node("Var", op.line, name=left.name), right=right)
            return Node("Assign", op.line, target=left, value=right)
        return left

    def binary(self, level):
        if level == len(BINARY_LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while self.cur.kind == "OP" and self.cur.value in BINARY_LEVELS[level]:
            op = self.cur
            self.i += 1
            right = self.binary(level + 1)
            left = Node("Binary", op.line, op=op.value, left=left, right=right)
        return left

    def unary(self):
        t = self.cur
        if t.kind == "OP" and t.value in ("-", "+", "!", "~"):
            self.i += 1
            operand = self.unary()
            return operand if t.value == "+" else Node("Unary", t.line, op=t.value, operand=operand)
        if t.kind == "OP" and t.value in ("++", "--"):
            self.i += 1
            target = self.unary()
            if target.kind != "Var":
                raise CompileError(f"operand of '{t.value}' must be a variable", t.line)
            one = Node("Num", t.line, value=1)
            val = Node("Binary", t.line, op=t.value[0],
                       left=Node("Var", t.line, name=target.name), right=one)
            return Node("Assign", t.line, target=target, value=val)
        return self.postfix()

    def postfix(self):
        node = self.primary()
        if self.cur.kind == "OP" and self.cur.value in ("++", "--"):
            t = self.cur
            self.i += 1
            if node.kind != "Var":
                raise CompileError(f"operand of '{t.value}' must be a variable", t.line)
            return Node("PostIncr", t.line, target=node, delta=1 if t.value == "++" else -1)
        return node

    def primary(self):
        t = self.cur
        if t.kind == "NUM":
            self.i += 1
            return Node("Num", t.line, value=t.value)
        if t.kind == "STR":
            self.i += 1
            return Node("Str", t.line, value=t.value)
        if t.kind == "ID":
            self.i += 1
            if self.accept("("):
                args = []
                if not self.check(")"):
                    while True:
                        args.append(self.expression())
                        if not self.accept(","):
                            break
                self.expect(")")
                return Node("Call", t.line, name=t.value, args=args)
            return Node("Var", t.line, name=t.value)
        if self.accept("("):
            e = self.expression()
            self.expect(")")
            return e
        found = t.value if t.kind != "EOF" else "end of file"
        raise CompileError(f"unexpected '{found}' in expression", t.line)


# ----------------------------------------------------------------------------
# 3. SEMANTIC ANALYSIS
#    - symbol tables with nested scopes
#    - undeclared / redeclared variables, function arity, break/continue misuse
#    - assigns each local variable a stack offset relative to %rbp
# ----------------------------------------------------------------------------
BUILTINS = {"printf"}


class Semantic:
    def __init__(self):
        self.funcs = {}
        self.defined = set()

    def analyze(self, prog):
        for f in prog.funcs:
            if f.name in BUILTINS:
                raise CompileError(f"'{f.name}' is a built-in function and cannot be redefined", f.line)
            prev = self.funcs.get(f.name)
            if prev and len(prev.params) != len(f.params):
                raise CompileError(f"conflicting declaration of '{f.name}'", f.line)
            if f.body:
                if f.name in self.defined:
                    raise CompileError(f"redefinition of function '{f.name}'", f.line)
                self.defined.add(f.name)
            self.funcs[f.name] = f
        if "main" not in self.defined:
            raise CompileError("program has no 'main' function", 0)
        for f in prog.funcs:
            if f.body:
                self.function(f)

    def function(self, f):
        if len(f.params) > 6:
            raise CompileError("functions with more than 6 parameters are not supported", f.line)
        self.func = f
        self.scopes = [{}]
        self.offset = 0
        self.loops = 0
        f.param_offsets = []
        f.symbols = []
        for p in f.params:
            if p in self.scopes[-1]:
                raise CompileError(f"duplicate parameter '{p}'", f.line)
            self.offset += 8
            self.scopes[-1][p] = -self.offset
            f.param_offsets.append(-self.offset)
            f.symbols.append((p, -self.offset))
        self.stmt(f.body)
        f.frame_size = (self.offset + 15) // 16 * 16

    def declare(self, name, line):
        if name in self.scopes[-1]:
            raise CompileError(f"redeclaration of '{name}' in the same scope", line)
        self.offset += 8
        self.scopes[-1][name] = -self.offset
        self.func.symbols.append((name, -self.offset))
        return -self.offset

    def lookup(self, name, line):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise CompileError(f"undeclared variable '{name}'", line)

    def stmt(self, n):
        k = n.kind
        if k == "Block":
            self.scopes.append({})
            for s in n.stmts:
                self.stmt(s)
            self.scopes.pop()
        elif k == "Decls":
            for d in n.items:
                if d.init:
                    self.expr(d.init)
                d.offset = self.declare(d.name, d.line)
        elif k == "ExprStmt":
            self.expr(n.expr)
        elif k == "If":
            self.expr(n.cond)
            self.stmt(n.then)
            if n.els:
                self.stmt(n.els)
        elif k == "While":
            self.expr(n.cond)
            self.loops += 1
            self.stmt(n.body)
            self.loops -= 1
        elif k == "DoWhile":
            self.loops += 1
            self.stmt(n.body)
            self.loops -= 1
            self.expr(n.cond)
        elif k == "For":
            self.scopes.append({})
            if n.init:
                self.stmt(n.init)
            if n.cond:
                self.expr(n.cond)
            if n.step:
                self.expr(n.step)
            self.loops += 1
            self.stmt(n.body)
            self.loops -= 1
            self.scopes.pop()
        elif k == "Return":
            if n.value:
                if self.func.rtype == "void":
                    raise CompileError("void function should not return a value", n.line)
                self.expr(n.value)
        elif k in ("Break", "Continue"):
            if self.loops == 0:
                raise CompileError(f"'{k.lower()}' outside of a loop", n.line)

    def expr(self, n):
        k = n.kind
        if k == "Num":
            pass
        elif k == "Str":
            raise CompileError("string literals can only be used as arguments to printf", n.line)
        elif k == "Var":
            n.offset = self.lookup(n.name, n.line)
        elif k == "Assign":
            self.expr(n.target)
            self.expr(n.value)
        elif k == "PostIncr":
            self.expr(n.target)
        elif k == "Unary":
            self.expr(n.operand)
        elif k == "Binary":
            self.expr(n.left)
            self.expr(n.right)
        elif k == "Call":
            if n.name == "printf":
                if not n.args or n.args[0].kind != "Str":
                    raise CompileError("printf needs a format string as first argument", n.line)
                if len(n.args) > 6:
                    raise CompileError("printf supports at most 5 value arguments", n.line)
                for a in n.args[1:]:
                    self.expr(a)
                return
            f = self.funcs.get(n.name)
            if f is None:
                raise CompileError(f"call to undeclared function '{n.name}'", n.line)
            if len(n.args) != len(f.params):
                raise CompileError(f"'{n.name}' expects {len(f.params)} argument(s) "
                                   f"but {len(n.args)} given", n.line)
            for a in n.args:
                self.expr(a)


# ----------------------------------------------------------------------------
# 4. OPTIMIZATION: constant folding
# ----------------------------------------------------------------------------
def wrap64(v):
    v &= (1 << 64) - 1
    return v - (1 << 64) if v >> 63 else v


def fold(n):
    if isinstance(n, list):
        return [fold(x) for x in n]
    if not isinstance(n, Node):
        return n
    for k, v in list(vars(n).items()):
        if isinstance(v, (Node, list)):
            setattr(n, k, fold(v))

    if n.kind == "Binary" and n.left.kind == "Num" and n.right.kind == "Num":
        a, b, op = n.left.value, n.right.value, n.op
        if op in ("/", "%") and b == 0:
            return n  # leave it, will crash at run time like C
        if op in ("<<", ">>") and not 0 <= b < 64:
            return n
        q = abs(a) // abs(b) if b else 0
        q = q if (a < 0) == (b < 0) else -q  # C truncates toward zero
        table = {
            "+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
            "/": lambda: q, "%": lambda: a - q * b,
            "<": lambda: int(a < b), ">": lambda: int(a > b),
            "<=": lambda: int(a <= b), ">=": lambda: int(a >= b),
            "==": lambda: int(a == b), "!=": lambda: int(a != b),
            "&&": lambda: int(bool(a) and bool(b)), "||": lambda: int(bool(a) or bool(b)),
            "&": lambda: a & b, "|": lambda: a | b, "^": lambda: a ^ b,
            "<<": lambda: a << b, ">>": lambda: a >> b,
        }
        return Node("Num", n.line, value=wrap64(table[op]()))
    if n.kind == "Unary" and n.operand.kind == "Num":
        v = n.operand.value
        res = {"-": lambda: wrap64(-v), "~": lambda: ~v, "!": lambda: int(v == 0)}[n.op]()
        return Node("Num", n.line, value=res)
    # dead-branch elimination
    if n.kind == "If" and n.cond.kind == "Num":
        return n.then if n.cond.value else (n.els or Node("Empty", n.line))
    if n.kind == "While" and n.cond.kind == "Num" and n.cond.value == 0:
        return Node("Empty", n.line)
    return n


# ----------------------------------------------------------------------------
# 5. CODE GENERATION  (x86-64, AT&T syntax, System V ABI)
#    Every expression leaves its result in %rax. Binary operators push the
#    left operand on the stack, evaluate the right one, then pop.
# ----------------------------------------------------------------------------
ARG_REGS = ["%rdi", "%rsi", "%rdx", "%rcx", "%r8", "%r9"]
SETCC = {"==": "sete", "!=": "setne", "<": "setl", ">": "setg", "<=": "setle", ">=": "setge"}


def asm_escape(s):
    out = []
    for ch in s:
        if ch == "\n": out.append("\\n")
        elif ch == "\t": out.append("\\t")
        elif ch == "\r": out.append("\\r")
        elif ch == '"': out.append('\\"')
        elif ch == "\\": out.append("\\\\")
        elif 32 <= ord(ch) < 127: out.append(ch)
        else: out.append("\\%03o" % ord(ch))
    return "".join(out)


class CodeGen:
    def __init__(self, target="x86_64"):
        self.target = target
        self.is_mingw32 = target == "mingw32"
        self.out = []
        self.strings = []
        self.nlabel = 0
        self.depth = 0      # number of temporary stack words in flight
        self.loops = []     # stack of (continue_label, break_label)

    def emit(self, s):
        self.out.append("\t" + s)

    def raw(self, s):
        self.out.append(s)

    def label(self, hint="L"):
        self.nlabel += 1
        return f".L{hint}{self.nlabel}"

    def symbol(self, name):
        return f"_{name}" if self.is_mingw32 else name

    def generate(self, prog):
        self.raw("\t.text")
        for f in prog.funcs:
            if f.body:
                self.func(f)
        if self.strings:
            self.raw('\t.section .rdata,"dr"' if self.is_mingw32 else "\t.section .rodata")
            for lbl, s in self.strings:
                self.raw(f"{lbl}:")
                self.emit(f'.string "{asm_escape(s)}"')
        if not self.is_mingw32:
            self.raw('\t.section .note.GNU-stack,"",@progbits')
        return "\n".join(self.out) + "\n"

    def func(self, f):
        self.ret_label = f".Lret_{f.name}"
        self.depth = 0
        name = self.symbol(f.name)
        self.raw(f"\t.globl {name}")
        if self.is_mingw32:
            self.raw(f"\t.def {name}; .scl 2; .type 32; .endef")
            self.raw(f"{name}:")
            self.emit("pushl %ebp")
            self.emit("movl %esp, %ebp")
            self.emit("andl $-16, %esp")
            if f.frame_size:
                self.emit(f"subl ${f.frame_size}, %esp")
            for i, off in enumerate(f.param_offsets):
                self.emit(f"movl {8 + 4 * i}(%ebp), %eax")
                self.emit(f"movl %eax, {off}(%ebp)")
        else:
            self.raw(f"\t.type {name}, @function")
            self.raw(f"{name}:")
            self.emit("pushq %rbp")
            self.emit("movq %rsp, %rbp")
            if f.frame_size:
                self.emit(f"subq ${f.frame_size}, %rsp")
            for reg, off in zip(ARG_REGS, f.param_offsets):
                self.emit(f"movq {reg}, {off}(%rbp)")
        self.stmt(f.body)
        if self.is_mingw32:
            self.emit("movl $0, %eax            # default return value")
        else:
            self.emit("movq $0, %rax            # default return value")
        self.raw(f"{self.ret_label}:")
        if self.is_mingw32:
            self.emit("movl %ebp, %esp")
            self.emit("popl %ebp")
        else:
            self.emit("movq %rbp, %rsp")
            self.emit("popq %rbp")
        self.emit("ret")

    # -- statements
    def stmt(self, n):
        k = n.kind
        if k == "Block":
            for s in n.stmts:
                self.stmt(s)
        elif k == "Decls":
            for d in n.items:
                if d.init:
                    self.expr(d.init)
                    ins = "movl" if self.is_mingw32 else "movq"
                    reg = "%eax" if self.is_mingw32 else "%rax"
                    self.emit(f"{ins} {reg}, {d.offset}(%{'ebp' if self.is_mingw32 else 'rbp'})    # {d.name} = ...")
        elif k == "ExprStmt":
            self.expr(n.expr)
        elif k == "If":
            l_else, l_end = self.label("else"), self.label("endif")
            self.expr(n.cond)
            self.emit("cmpl $0, %eax" if self.is_mingw32 else "cmpq $0, %rax")
            self.emit(f"je {l_else}")
            self.stmt(n.then)
            self.emit(f"jmp {l_end}")
            self.raw(f"{l_else}:")
            if n.els:
                self.stmt(n.els)
            self.raw(f"{l_end}:")
        elif k == "While":
            l_top, l_end = self.label("while"), self.label("endwhile")
            self.raw(f"{l_top}:")
            self.expr(n.cond)
            self.emit("cmpl $0, %eax" if self.is_mingw32 else "cmpq $0, %rax")
            self.emit(f"je {l_end}")
            self.loops.append((l_top, l_end))
            self.stmt(n.body)
            self.loops.pop()
            self.emit(f"jmp {l_top}")
            self.raw(f"{l_end}:")
        elif k == "DoWhile":
            l_top, l_cont, l_end = self.label("do"), self.label("docond"), self.label("enddo")
            self.raw(f"{l_top}:")
            self.loops.append((l_cont, l_end))
            self.stmt(n.body)
            self.loops.pop()
            self.raw(f"{l_cont}:")
            self.expr(n.cond)
            self.emit("cmpl $0, %eax" if self.is_mingw32 else "cmpq $0, %rax")
            self.emit(f"jne {l_top}")
            self.raw(f"{l_end}:")
        elif k == "For":
            l_cond, l_step, l_end = self.label("for"), self.label("forstep"), self.label("endfor")
            if n.init:
                self.stmt(n.init)
            self.raw(f"{l_cond}:")
            if n.cond:
                self.expr(n.cond)
                self.emit("cmpl $0, %eax" if self.is_mingw32 else "cmpq $0, %rax")
                self.emit(f"je {l_end}")
            self.loops.append((l_step, l_end))
            self.stmt(n.body)
            self.loops.pop()
            self.raw(f"{l_step}:")
            if n.step:
                self.expr(n.step)
            self.emit(f"jmp {l_cond}")
            self.raw(f"{l_end}:")
        elif k == "Return":
            if n.value:
                self.expr(n.value)
            else:
                self.emit("movl $0, %eax" if self.is_mingw32 else "movq $0, %rax")
            self.emit(f"jmp {self.ret_label}")
        elif k == "Break":
            self.emit(f"jmp {self.loops[-1][1]}")
        elif k == "Continue":
            self.emit(f"jmp {self.loops[-1][0]}")

    # -- expressions
    def expr(self, n):
        k = n.kind
        if k == "Num":
            if self.is_mingw32:
                self.emit(f"movl ${n.value}, %eax")
            elif -2**31 <= n.value < 2**31:
                self.emit(f"movq ${n.value}, %rax")
            else:
                self.emit(f"movabsq ${n.value}, %rax")
        elif k == "Str":
            lbl = f".LC{len(self.strings)}"
            self.strings.append((lbl, n.value))
            if self.is_mingw32:
                self.emit(f"movl ${lbl}, %eax")
            else:
                self.emit(f"leaq {lbl}(%rip), %rax")
        elif k == "Var":
            self.emit(f"mov{'l' if self.is_mingw32 else 'q'} {n.offset}(%{'ebp' if self.is_mingw32 else 'rbp'}), %{'eax' if self.is_mingw32 else 'rax'}    # {n.name}")
        elif k == "Assign":
            self.expr(n.value)
            self.emit(f"mov{'l' if self.is_mingw32 else 'q'} %{'eax' if self.is_mingw32 else 'rax'}, {n.target.offset}(%{'ebp' if self.is_mingw32 else 'rbp'})    # {n.target.name} =")
        elif k == "PostIncr":
            if self.is_mingw32:
                self.emit(f"movl {n.target.offset}(%ebp), %eax    # {n.target.name}{'++' if n.delta > 0 else '--'}")
                self.emit("movl %eax, %ecx")
                self.emit(f"addl ${n.delta}, %ecx")
                self.emit(f"movl %ecx, {n.target.offset}(%ebp)")
            else:
                self.emit(f"movq {n.target.offset}(%rbp), %rax    # {n.target.name}{'++' if n.delta > 0 else '--'}")
                self.emit("movq %rax, %rcx")
                self.emit(f"addq ${n.delta}, %rcx")
                self.emit(f"movq %rcx, {n.target.offset}(%rbp)")
        elif k == "Unary":
            self.expr(n.operand)
            suffix = "l" if self.is_mingw32 else "q"
            acc = "%eax" if self.is_mingw32 else "%rax"
            if n.op == "-":
                self.emit(f"neg{suffix} {acc}")
            elif n.op == "~":
                self.emit(f"not{suffix} {acc}")
            else:  # '!'
                self.emit(f"cmp{suffix} $0, {acc}")
                self.emit("sete %al")
                self.emit("movzbl %al, %eax" if self.is_mingw32 else "movzbq %al, %rax")
        elif k == "Binary":
            self.binary(n)
        elif k == "Call":
            self.call(n)

    def binary(self, n):
        op = n.op
        if self.is_mingw32:
            self._binary32(n)
            return
        if op == "&&":
            l_false, l_end = self.label("false"), self.label("endand")
            self.expr(n.left)
            self.emit("cmpq $0, %rax")
            self.emit(f"je {l_false}")
            self.expr(n.right)
            self.emit("cmpq $0, %rax")
            self.emit(f"je {l_false}")
            self.emit("movq $1, %rax")
            self.emit(f"jmp {l_end}")
            self.raw(f"{l_false}:")
            self.emit("movq $0, %rax")
            self.raw(f"{l_end}:")
            return
        if op == "||":
            l_true, l_end = self.label("true"), self.label("endor")
            self.expr(n.left)
            self.emit("cmpq $0, %rax")
            self.emit(f"jne {l_true}")
            self.expr(n.right)
            self.emit("cmpq $0, %rax")
            self.emit(f"jne {l_true}")
            self.emit("movq $0, %rax")
            self.emit(f"jmp {l_end}")
            self.raw(f"{l_true}:")
            self.emit("movq $1, %rax")
            self.raw(f"{l_end}:")
            return

        self.expr(n.left)
        self.emit("pushq %rax")
        self.depth += 1
        self.expr(n.right)
        self.emit("movq %rax, %rcx")
        self.emit("popq %rax")
        self.depth -= 1
        if op == "+":
            self.emit("addq %rcx, %rax")
        elif op == "-":
            self.emit("subq %rcx, %rax")
        elif op == "*":
            self.emit("imulq %rcx, %rax")
        elif op == "&":
            self.emit("andq %rcx, %rax")
        elif op == "|":
            self.emit("orq %rcx, %rax")
        elif op == "^":
            self.emit("xorq %rcx, %rax")
        elif op == "<<":
            self.emit("salq %cl, %rax")
        elif op == ">>":
            self.emit("sarq %cl, %rax")
        elif op in ("/", "%"):
            self.emit("cqto")
            self.emit("idivq %rcx")
            if op == "%":
                self.emit("movq %rdx, %rax")
        else:
            self.emit("cmpq %rcx, %rax")
            self.emit(f"{SETCC[op]} %al")
            self.emit("movzbq %al, %rax")

    def call(self, n):
        if self.is_mingw32:
            pad = (-4 * (self.depth + len(n.args))) % 16
            if pad:
                self.emit(f"subl ${pad}, %esp")
                self.depth += pad // 4
            for a in reversed(n.args):
                self.expr(a)
                self.emit("pushl %eax")
                self.depth += 1
            self.emit(f"call {self.symbol(n.name)}")
            if n.args:
                self.emit(f"addl ${4 * len(n.args)}, %esp")
                self.depth -= len(n.args)
            if pad:
                self.emit(f"addl ${pad}, %esp")
                self.depth -= pad // 4
            return
        for a in n.args:
            self.expr(a)
            self.emit("pushq %rax")
            self.depth += 1
        for i in reversed(range(len(n.args))):
            self.emit(f"popq {ARG_REGS[i]}")
            self.depth -= 1
        pad = self.depth % 2 == 1  # keep %rsp 16-byte aligned at the call
        if pad:
            self.emit("subq $8, %rsp")
        if n.name == "printf":
            self.emit("movl $0, %eax            # no vector registers used")
            self.emit("call printf@PLT")
        else:
            self.emit(f"call {n.name}")
        if pad:
            self.emit("addq $8, %rsp")

    def _binary32(self, n):
        op = n.op
        if op == "&&":
            l_false, l_end = self.label("false"), self.label("endand")
            self.expr(n.left)
            self.emit("cmpl $0, %eax")
            self.emit(f"je {l_false}")
            self.expr(n.right)
            self.emit("cmpl $0, %eax")
            self.emit(f"je {l_false}")
            self.emit("movl $1, %eax")
            self.emit(f"jmp {l_end}")
            self.raw(f"{l_false}:")
            self.emit("movl $0, %eax")
            self.raw(f"{l_end}:")
            return
        if op == "||":
            l_true, l_end = self.label("true"), self.label("endor")
            self.expr(n.left)
            self.emit("cmpl $0, %eax")
            self.emit(f"jne {l_true}")
            self.expr(n.right)
            self.emit("cmpl $0, %eax")
            self.emit(f"jne {l_true}")
            self.emit("movl $0, %eax")
            self.emit(f"jmp {l_end}")
            self.raw(f"{l_true}:")
            self.emit("movl $1, %eax")
            self.raw(f"{l_end}:")
            return

        self.expr(n.left)
        self.emit("pushl %eax")
        self.depth += 1
        self.expr(n.right)
        self.emit("movl %eax, %ecx")
        self.emit("popl %eax")
        self.depth -= 1
        if op == "+":
            self.emit("addl %ecx, %eax")
        elif op == "-":
            self.emit("subl %ecx, %eax")
        elif op == "*":
            self.emit("imull %ecx, %eax")
        elif op == "&":
            self.emit("andl %ecx, %eax")
        elif op == "|":
            self.emit("orl %ecx, %eax")
        elif op == "^":
            self.emit("xorl %ecx, %eax")
        elif op == "<<":
            self.emit("sall %cl, %eax")
        elif op == ">>":
            self.emit("sarl %cl, %eax")
        elif op in ("/", "%"):
            self.emit("cltd")
            self.emit("idivl %ecx")
            if op == "%":
                self.emit("movl %edx, %eax")
        else:
            self.emit("cmpl %ecx, %eax")
            self.emit(f"{SETCC[op]} %al")
            self.emit("movzbl %al, %eax")


# ----------------------------------------------------------------------------
# Intermediate code: three-address code (printed with --tac)
# ----------------------------------------------------------------------------
class TAC:
    def __init__(self):
        self.code, self.nt, self.nl, self.loops = [], 0, 0, []

    def tmp(self):
        self.nt += 1
        return f"t{self.nt}"

    def lab(self):
        self.nl += 1
        return f"L{self.nl}"

    def e(self, s):
        self.code.append("    " + s)

    def mark(self, label):
        self.code.append(f"{label}:")

    def generate(self, prog):
        for f in prog.funcs:
            if f.body:
                self.nt = 0
                self.code.append(f"func {f.name}({', '.join(f.params)}):")
                self.stmt(f.body)
                self.code.append("endfunc\n")
        return "\n".join(self.code)

    def stmt(self, n):
        k = n.kind
        if k == "Block":
            for s in n.stmts:
                self.stmt(s)
        elif k == "Decls":
            for d in n.items:
                if d.init:
                    self.e(f"{d.name} = {self.expr(d.init)}")
        elif k == "ExprStmt":
            self.expr(n.expr)
        elif k == "If":
            c = self.expr(n.cond)
            l_else, l_end = self.lab(), self.lab()
            self.e(f"ifFalse {c} goto {l_else}")
            self.stmt(n.then)
            self.e(f"goto {l_end}")
            self.mark(l_else)
            if n.els:
                self.stmt(n.els)
            self.mark(l_end)
        elif k == "While":
            top, end = self.lab(), self.lab()
            self.mark(top)
            c = self.expr(n.cond)
            self.e(f"ifFalse {c} goto {end}")
            self.loops.append((top, end))
            self.stmt(n.body)
            self.loops.pop()
            self.e(f"goto {top}")
            self.mark(end)
        elif k == "DoWhile":
            top, cont, end = self.lab(), self.lab(), self.lab()
            self.mark(top)
            self.loops.append((cont, end))
            self.stmt(n.body)
            self.loops.pop()
            self.mark(cont)
            c = self.expr(n.cond)
            self.e(f"if {c} goto {top}")
            self.mark(end)
        elif k == "For":
            cond, step, end = self.lab(), self.lab(), self.lab()
            if n.init:
                self.stmt(n.init)
            self.mark(cond)
            if n.cond:
                c = self.expr(n.cond)
                self.e(f"ifFalse {c} goto {end}")
            self.loops.append((step, end))
            self.stmt(n.body)
            self.loops.pop()
            self.mark(step)
            if n.step:
                self.expr(n.step)
            self.e(f"goto {cond}")
            self.mark(end)
        elif k == "Return":
            self.e(f"return {self.expr(n.value) if n.value else 0}")
        elif k == "Break":
            self.e(f"goto {self.loops[-1][1]}")
        elif k == "Continue":
            self.e(f"goto {self.loops[-1][0]}")

    def expr(self, n):
        k = n.kind
        if k == "Num":
            return str(n.value)
        if k == "Str":
            return '"' + asm_escape(n.value) + '"'
        if k == "Var":
            return n.name
        if k == "Assign":
            v = self.expr(n.value)
            self.e(f"{n.target.name} = {v}")
            return n.target.name
        if k == "PostIncr":
            t = self.tmp()
            self.e(f"{t} = {n.target.name}")
            self.e(f"{n.target.name} = {n.target.name} {'+' if n.delta > 0 else '-'} 1")
            return t
        if k == "Unary":
            v = self.expr(n.operand)
            t = self.tmp()
            self.e(f"{t} = {n.op}{v}")
            return t
        if k == "Binary":
            l, r = self.expr(n.left), self.expr(n.right)
            t = self.tmp()
            self.e(f"{t} = {l} {n.op} {r}")
            return t
        if k == "Call":
            args = [self.expr(a) for a in n.args]
            for a in args:
                self.e(f"param {a}")
            t = self.tmp()
            self.e(f"{t} = call {n.name}, {len(args)}")
            return t


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def compile_source(src, optimize=True, show_tokens=False, show_ast=False,
                   show_tac=False, show_symbols=False, target="x86_64"):
    tokens = tokenize(src)
    if show_tokens:
        print("=== TOKENS ===")
        for t in tokens:
            print(t)
    ast = Parser(tokens).program()
    Semantic().analyze(ast)
    if show_symbols:
        print("=== SYMBOL TABLE ===")
        for f in ast.funcs:
            if f.body:
                print(f"{f.rtype} {f.name}({', '.join(f.params)})   frame = {f.frame_size} bytes")
                for name, off in f.symbols:
                    print(f"    {name:<12} int   {off}(%rbp)")
    if optimize:
        ast = fold(ast)
    if show_ast:
        print("=== AST ===")
        dump(ast)
    if show_tac:
        print("=== THREE-ADDRESS CODE ===")
        print(TAC().generate(ast))
    return CodeGen(target).generate(ast)


def main():
    ap = argparse.ArgumentParser(description="minicc - a small C compiler in Python")
    ap.add_argument("source", help="C source file")
    ap.add_argument("-o", "--output", help="output executable name")
    ap.add_argument("-S", action="store_true", help="only generate assembly")
    ap.add_argument("--run", action="store_true", help="run the program after building")
    ap.add_argument("--tokens", action="store_true", help="print the token stream")
    ap.add_argument("--ast", action="store_true", help="print the AST")
    ap.add_argument("--tac", action="store_true", help="print three-address code")
    ap.add_argument("--symbols", action="store_true", help="print the symbol table")
    ap.add_argument("--no-opt", action="store_true", help="disable constant folding")
    args = ap.parse_args()

    with open(args.source) as fh:
        src = fh.read()
    gcc = shutil.which("gcc")
    target = "x86_64"
    if gcc:
        machine = subprocess.run(
            [gcc, "-dumpmachine"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if machine == "mingw32" or machine.startswith("i686-w64-mingw32"):
            target = "mingw32"
        elif not machine.startswith("x86_64") or "mingw" in machine:
            print(f"unsupported GCC target '{machine}'; supported targets are x86-64 and MinGW 32-bit",
                  file=sys.stderr)
            sys.exit(1)
    try:
        asm = compile_source(src, not args.no_opt, args.tokens, args.ast, args.tac,
                             args.symbols, target)
    except CompileError as e:
        loc = f"{args.source}:{e.line}: " if e.line else f"{args.source}: "
        print(f"{loc}error: {e.msg}", file=sys.stderr)
        sys.exit(1)

    base = os.path.splitext(args.source)[0]
    asm_path = base + ".s"
    with open(asm_path, "w") as fh:
        fh.write(asm)
    print(f"wrote {asm_path}")
    if args.S:
        return

    if not gcc:
        print("gcc not found; assemble manually with:  gcc prog.s -o prog")
        return
    exe = args.output or base
    if target == "mingw32" and not exe.lower().endswith(".exe"):
        exe += ".exe"
    r = subprocess.run(["gcc", asm_path, "-o", exe])
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"built {exe}")
    if args.run:
        sys.exit(subprocess.run([os.path.abspath(exe)]).returncode)


if __name__ == "__main__":
    main()
