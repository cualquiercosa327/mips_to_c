import attr
import traceback
import copy

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any

from parse_instruction import *
from flow_graph import *
from parse_file import *

ARGUMENT_REGS = [
    'a0', 'a1', 'a2', 'a3',
    'f12', 'f14',
]

# TODO: include temporary floating-point registers
CALLER_SAVE_REGS = ARGUMENT_REGS + [
    'at',
    't0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9',
    'hi', 'lo', 'condition_bit', 'return_reg'
]

SPECIAL_REGS = [
    's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
    'ra',
    '31'
]

DEBUG = False
IGNORE_ERRORS = False


@attr.s
class StackInfo:
    function: Function = attr.ib()
    allocated_stack_size: int = attr.ib(default=0)
    is_leaf: bool = attr.ib(default=True)
    local_vars_region_bottom: int = attr.ib(default=0)
    return_addr_location: int = attr.ib(default=0)
    callee_save_reg_locations: Dict[Register, int] = attr.ib(factory=dict)
    local_vars: List['LocalVar'] = attr.ib(factory=list)
    arguments: List['PassedInArg'] = attr.ib(factory=list)
    temp_vars: List['EvalOnceStmt'] = attr.ib(factory=list)
    temp_name_counter: Dict[str, int] = attr.ib(factory=dict)
    stack_types: Dict[int, Optional[str]] = attr.ib(factory=dict)

    def temp_var_generator(self, prefix: str) -> Callable[[], str]:
        def gen() -> str:
            counter = self.temp_name_counter.get(prefix, 0) + 1
            self.temp_name_counter[prefix] = counter
            return f'temp_{prefix}' + (f'_{counter}' if counter > 1 else '')
        return gen

    def in_subroutine_arg_region(self, location: int) -> bool:
        assert not self.is_leaf
        if self.callee_save_reg_locations:
            subroutine_arg_top = min(self.callee_save_reg_locations.values())
            assert self.return_addr_location > subroutine_arg_top
        else:
            subroutine_arg_top = self.return_addr_location

        return location < subroutine_arg_top

    def in_local_var_region(self, location: int) -> bool:
        return self.local_vars_region_bottom <= location < self.allocated_stack_size

    def location_above_stack(self, location: int) -> bool:
        return location >= self.allocated_stack_size

    def add_local_var(self, var: 'LocalVar') -> None:
        if var in self.local_vars:
            return
        self.local_vars.append(var)
        # Make sure the local vars stay sorted in order on the stack.
        self.local_vars.sort(key=lambda v: v.value)

    def add_argument(self, arg: 'PassedInArg') -> None:
        if any(a.value == arg.value for a in self.arguments):
            return
        self.set_stack_type(arg.value, arg.guessed_type)
        self.arguments.append(arg)
        self.arguments.sort(key=lambda a: a.value)

    def get_stack_type(self, expr) -> Optional[str]:
        if isinstance(expr, SubroutineArg):
            return expr.type
        return self.stack_types.get(expr.value, None)

    def set_stack_type(self, location: int, type: Optional[str]) -> None:
        # Refine in the opposite order to make the new type take precedence
        self.stack_types[location] = refine_type(type,
                self.stack_types.get(location, None))

    def get_stack_var(self, location: int, type: Optional[str]) -> 'Expression':
        # This is either a local variable or an argument.
        if self.in_local_var_region(location):
            self.set_stack_type(location, type)
            return LocalVar(location, stack_info=self)
        elif self.location_above_stack(location):
            self.set_stack_type(location, type)
            return PassedInArg(location, stack_info=self, guessed_type=None)
        elif self.in_subroutine_arg_region(location):
            return SubroutineArg(location, type=type)
        else:
            # Some annoying bookkeeping instruction. To avoid
            # further special-casing, just return whatever - it won't matter.
            return LocalVar(location, stack_info=self)

    def __str__(self) -> str:
        return '\n'.join([
            f'Stack info for function {self.function.name}:',
            f'Allocated stack size: {self.allocated_stack_size}',
            f'Leaf? {self.is_leaf}',
            f'Bottom of local vars region: {self.local_vars_region_bottom}',
            f'Location of return addr: {self.return_addr_location}',
            f'Locations of callee save registers: {self.callee_save_reg_locations}'
        ])

def get_stack_info(function: Function, start_node: Node) -> StackInfo:
    info = StackInfo(function)

    # The goal here is to pick out special instructions that provide information
    # about this function's stack setup.
    for inst in start_node.block.instructions:
        if not inst.args:
            continue

        destination = typing.cast(Register, inst.args[0])

        if inst.mnemonic == 'addiu' and destination.register_name == 'sp':
            # Moving the stack pointer.
            assert isinstance(inst.args[2], NumberLiteral)
            info.allocated_stack_size = -inst.args[2].value
        elif inst.mnemonic == 'sw' and destination.register_name == 'ra':
            # Saving the return address on the stack.
            assert isinstance(inst.args[1], AddressMode)
            assert inst.args[1].rhs.register_name == 'sp'
            info.is_leaf = False
            if inst.args[1].lhs:
                assert isinstance(inst.args[1].lhs, NumberLiteral)
                info.return_addr_location = inst.args[1].lhs.value
            else:
                # Note that this should only happen in the rare case that
                # this function only calls subroutines with no arguments.
                info.return_addr_location = 0
        elif (inst.mnemonic == 'sw' and
              destination.is_callee_save() and
              isinstance(inst.args[1], AddressMode) and
              inst.args[1].rhs.register_name == 'sp'):
            # Initial saving of callee-save register onto the stack.
            assert isinstance(inst.args[1].rhs, Register)
            if inst.args[1].lhs:
                assert isinstance(inst.args[1].lhs, NumberLiteral)
                info.callee_save_reg_locations[destination] = inst.args[1].lhs.value
            else:
                info.callee_save_reg_locations[destination] = 0

    # Find the region that contains local variables.
    if info.is_leaf and info.callee_save_reg_locations:
        # In a leaf with callee-save registers, the local variables
        # lie directly above those registers.
        info.local_vars_region_bottom = max(info.callee_save_reg_locations.values()) + 4
    elif info.is_leaf:
        # In a leaf without callee-save registers, the local variables
        # lie directly at the bottom of the stack.
        info.local_vars_region_bottom = 0
    else:
        # In a non-leaf, the local variables lie above the location of the
        # return address.
        info.local_vars_region_bottom = info.return_addr_location + 4

    # Done.
    return info

def format_hex(val: int) -> str:
    return format(val, 'x').upper()

@attr.s(frozen=True)
class BinaryOp:
    left: 'Expression' = attr.ib()
    op: str = attr.ib()
    right: 'Expression' = attr.ib()
    in_type: Optional[str] = attr.ib()
    right_in_type: Optional[str] = attr.ib(default=
            attr.Factory(lambda self: self.in_type, takes_self=True))

    def is_boolean(self) -> bool:
        return self.op in ['==', '!=', '>', '<', '>=', '<=']

    def negated(self) -> 'BinaryOp':
        assert self.is_boolean()
        return BinaryOp(
            left=self.left,
            op={
                '==': '!=',
                '!=': '==',
                '>' : '<=',
                '<' : '>=',
                '>=':  '<',
                '<=':  '>',
            }[self.op],
            right=self.right,
            in_type=self.in_type,
            right_in_type=self.right_in_type
        )

    def __str__(self) -> str:
        return f'({cast(self.left, self.in_type)} {self.op} {cast(self.right, self.right_in_type)})'

@attr.s(frozen=True)
class UnaryOp:
    op: str = attr.ib()
    expr: 'Expression' = attr.ib()
    in_type: Optional[str] = attr.ib()
    type: str = attr.ib()

    def __str__(self) -> str:
        return f'{self.op}{self.expr}'

@attr.s(frozen=True)
class Cast:
    to_type: str = attr.ib()
    expr: 'Expression' = attr.ib()
    in_type: Optional[str] = attr.ib()

    def __str__(self) -> str:
        return f'({type_to_str(self.to_type)}) {self.expr}'

@attr.s(cmp=False)
class FuncCall:
    func_name: str = attr.ib()
    args: List['Expression'] = attr.ib()
    type: Optional[str] = attr.ib(default=None)

    def __str__(self) -> str:
        return f'{self.func_name}({", ".join(str(arg) for arg in self.args)})'

@attr.s(frozen=True)
class LocalVar:
    value: int = attr.ib()
    stack_info: StackInfo = attr.ib(repr=False)

    def __str__(self) -> str:
        return f'sp{format_hex(self.value)}'

@attr.s(frozen=True)
class PassedInArg:
    value: int = attr.ib()
    stack_info: StackInfo = attr.ib(repr=False)
    guessed_type: Optional[str] = attr.ib()

    def declaration_str(self) -> str:
        type = self.stack_info.get_stack_type(self)
        return f'{type_to_str(type)} {self}'

    def __str__(self) -> str:
        return f'arg{format_hex(self.value)}'

@attr.s(cmp=False)
class SubroutineArg:
    value: int = attr.ib()
    type: Optional[str] = attr.ib()

    def __str__(self) -> str:
        return f'subroutine_arg{format_hex(self.value)}'

@attr.s(cmp=False)
class StructAccess:
    struct_var = attr.ib()
    offset: int = attr.ib()
    type: Optional[str] = attr.ib()

    def __str__(self) -> str:
        return f'{self.struct_var}->unk{format_hex(self.offset)}'

@attr.s(frozen=True)
class Literal:
    value: int = attr.ib()
    type: str = attr.ib(default='int')

    def __str__(self) -> str:
        if self.type == 'f32':
            return str(convert_to_float(self.value))
        if abs(self.value) < 10:
            return str(self.value)
        return hex(self.value)

@attr.s(cmp=False)
class EvalOnceExpr:
    wrapped_expr: 'Expression' = attr.ib()
    exact: bool = attr.ib()
    var: Union[str, Callable[[], str]] = attr.ib()
    num_usages: int = attr.ib(default=0)

    def __attrs_post_init__(self):
        # Enable deterministic names, to aid debugging:
        # self.get_var_name()
        pass

    def get_var_name(self) -> str:
        if not isinstance(self.var, str):
            self.var = self.var()
        return self.var

    def needs_var(self) -> bool:
        if self.exact:
            return self.num_usages != 1
        else:
            return self.num_usages > 1

    def __str__(self) -> str:
        if self.needs_var():
            return self.get_var_name()
        else:
            return str(self.wrapped_expr)

@attr.s(frozen=True)
class WrapperExpr:
    wrapped_expr: 'Expression' = attr.ib()

    def __str__(self) -> str:
        return str(self.wrapped_expr)

@attr.s
class EvalOnceStmt:
    expr: EvalOnceExpr = attr.ib()

    def need_decl(self) -> bool:
        return self.expr.num_usages > 1

    def should_write(self) -> bool:
        return self.expr.needs_var()

    def __str__(self) -> str:
        if self.expr.exact and self.expr.num_usages == 0:
            return f'{self.expr.wrapped_expr};'
        return f'{self.expr.get_var_name()} = {self.expr.wrapped_expr};'

@attr.s
class FuncCallStmt:
    expr: EvalOnceExpr = attr.ib()

    def should_write(self) -> bool:
        return True

    def __str__(self) -> str:
        return f'{self.expr};'

@attr.s
class StoreStmt:
    source: 'Expression' = attr.ib()
    dest: 'Expression' = attr.ib()
    type: str = attr.ib()

    def should_write(self) -> bool:
        return True

    def __str__(self) -> str:
        type_str = f'({type_to_str(self.type)}) '
        dest = self.dest
        if ((isinstance(dest, LocalVar) or isinstance(dest, PassedInArg)) and
                (self.type == get_type(dest) or self.type == 'int')):
            type_str = ''
        return f'{type_str}{dest} = {self.source};'

@attr.s
class CommentStmt:
    contents: str = attr.ib()

    def should_write(self) -> bool:
        return True

    def __str__(self) -> str:
        return f'// {self.contents}'

Expression = Union[
    BinaryOp,
    UnaryOp,
    Cast,
    FuncCall,
    GlobalSymbol,
    Literal,
    LocalVar,
    PassedInArg,
    StructAccess,
    SubroutineArg,
    EvalOnceExpr,
    WrapperExpr,
]

Statement = Union[
    StoreStmt,
    FuncCallStmt,
    EvalOnceStmt,
    CommentStmt,
]

def is_trivial_expr(expr: Expression) -> bool:
    if isinstance(expr, WrapperExpr):
        return is_trivial_expr(expr.wrapped_expr)
    trivial_types = [EvalOnceExpr, Literal, GlobalSymbol, LocalVar,
            PassedInArg, SubroutineArg]
    return expr is None or any(isinstance(expr, t) for t in trivial_types)

def simplify_expr(expr: Expression) -> Expression:
    if isinstance(expr, WrapperExpr):
        return simplify_expr(expr.wrapped_expr)
    if isinstance(expr, EvalOnceExpr) and not expr.needs_var():
        return simplify_expr(expr.wrapped_expr)
    if isinstance(expr, BinaryOp):
        left = simplify_expr(expr.left)
        right = simplify_expr(expr.right)
        if (isinstance(left, BinaryOp) and left.is_boolean() and
                right == Literal(0)):
            if expr.op == '==':
                return simplify_expr(negate(left))
            if expr.op == '!=':
                return left
        return BinaryOp(left=left, op=expr.op, right=right, in_type=expr.in_type)
    if isinstance(expr, UnaryOp) and expr.op == '!':
        sub = simplify_expr(expr.expr)
        if isinstance(sub, UnaryOp) and sub.op == '!':
            return sub.expr
        if isinstance(sub, BinaryOp) and sub.is_boolean():
            return sub.negated()
    return expr

def negate(expr: Expression) -> Expression:
    return UnaryOp(op='!', expr=expr, in_type=None, type='s32')

# Replace a type by a potentially more informative one (if our best guess so
# far is just "some integral type")
def refine_type(orig_type: Optional[str], new_type: Optional[str]) -> Optional[str]:
    if new_type is None:
        return orig_type
    if orig_type is None or orig_type == 'int':
        return new_type
    return orig_type

def get_type(expr: Expression) -> Optional[str]:
    if isinstance(expr, WrapperExpr) or isinstance(expr, EvalOnceExpr):
        return get_type(expr.wrapped_expr)
    if isinstance(expr, BinaryOp):
        ltype = get_type(expr.left)
        rtype = get_type(expr.right)
        if (ltype == 'void*') + (rtype == 'void*') == 1:
            return 'void*'
        if expr.is_boolean() or expr.in_type == 'int':
            return 's32'
        return expr.in_type
    if (isinstance(expr, FuncCall) or isinstance(expr, UnaryOp) or
            isinstance(expr, StructAccess) or isinstance(expr, SubroutineArg)):
        return expr.type
    if isinstance(expr, Cast):
        return expr.to_type
    if isinstance(expr, GlobalSymbol):
        return 'void*'
    if isinstance(expr, Literal):
        return expr.type
    if isinstance(expr, LocalVar) or isinstance(expr, PassedInArg):
        return expr.stack_info.get_stack_type(expr)
    return None

def set_type(expr: Expression, type: Optional[str]) -> None:
    if isinstance(expr, WrapperExpr) or isinstance(expr, EvalOnceExpr):
        set_type(expr.wrapped_expr, type)
    elif (isinstance(expr, FuncCall) or isinstance(expr, StructAccess) or
            isinstance(expr, SubroutineArg)):
        expr.type = refine_type(expr.type, type)
    elif isinstance(expr, LocalVar) or isinstance(expr, PassedInArg):
        expr.stack_info.set_stack_type(expr.value, type)

def cast(expr: Expression, to_type: Optional[str]) -> Expression:
    if to_type is None:
        return expr
    orig_type = get_type(expr)
    if to_type != orig_type and to_type != 'int':
        return Cast(to_type=to_type, expr=expr, in_type=None)
    return expr

def type_to_str(type: Optional[str]) -> str:
    if type is None:
        return '?'
    if type == 'int':
        return 's32'
    return type

def replace_occurrences(expr: Expression, pattern: Expression, replacement: Expression):
    if expr == pattern:
        return replacement
    new_expr = expr
    if any(isinstance(expr, t) for t in [BinaryOp, UnaryOp, Cast, FuncCall]):
        for k,v in expr.__dict__.items():
            v2 = replace_occurrences(v, pattern, replacement)
            if v != v2:
                new_expr = copy.copy(new_expr)
                new_expr.__dict__[k] = v2
    return new_expr

@attr.s
class RegInfo:
    contents: Dict[Register, Expression] = attr.ib()
    stack_info: StackInfo = attr.ib(repr=False)

    def get_nouse(self, key: Register) -> Expression:
        ret = self.contents[key]
        if isinstance(ret, PassedInArg):
            self.stack_info.add_argument(ret)
        return ret

    def __getitem__(self, key: Register) -> Expression:
        ret = self.get_nouse(key)
        if isinstance(ret, EvalOnceExpr):
            ret.num_usages += 1
        return ret

    def __contains__(self, key: Register) -> bool:
        return key in self.contents

    def __setitem__(self, key: Register, value: Optional[Expression]) -> None:
        assert key != Register('zero')
        if value is not None:
            self.contents[key] = value
        elif key in self.contents:
            del self.contents[key]
        if key.register_name in ['f0', 'v0']:
            self[Register('return_reg')] = value

    def __delitem__(self, key: Register) -> None:
        assert key != Register('zero')
        del self.contents[key]

    def get_raw(self, key: Register) -> Optional[Expression]:
        return self.contents.get(key, None)

    def clear_caller_save_regs(self) -> None:
        for reg in map(Register, CALLER_SAVE_REGS):
            assert reg != Register('zero')
            if reg in self.contents:
                del self.contents[reg]

    def replace_occurrences(self, pattern: Expression, replacement: Expression):
        for k,v in self.contents.items():
            self.contents[k] = replace_occurrences(v, pattern, replacement)

    def copy(self) -> 'RegInfo':
        return RegInfo(contents=self.contents.copy(), stack_info=self.stack_info)

    def __str__(self) -> str:
        return ', '.join(f"{k}: {v}" for k,v in sorted(self.contents.items()))


@attr.s
class BlockInfo:
    """
    Contains translated assembly code (to_write), the block's branch condition,
    block's final register states, and a return value (if this is a return node).
    """
    to_write: List[Statement] = attr.ib()
    branch_condition: Optional[Expression] = attr.ib()
    final_register_states: RegInfo = attr.ib()
    return_reg: Optional[Expression] = attr.ib(default=None)

    def __str__(self) -> str:
        newline = '\n\t'
        return '\n'.join([
            f'To write: {newline.join(str(write) for write in self.to_write if write.should_write())}',
            f'Branch condition: {self.branch_condition}',
            f'Final register states: {self.final_register_states}'])


@attr.s
class InstrArgs:
    raw_args: List[Argument] = attr.ib()
    regs: RegInfo = attr.ib(repr=False)

    def reg_ref(self, index: int) -> Register:
        ret = self.raw_args[index]
        assert isinstance(ret, Register)
        return ret

    def reg(self, index: int) -> Expression:
        return self.regs[self.reg_ref(index)]

    def reg_nouse(self, index: int) -> Expression:
        return self.regs.get_nouse(self.reg_ref(index))

    def imm(self, index: int) -> Expression:
        return literal_expr(self.raw_args[index])

    def memory_ref(self, index: int) -> Union[AddressMode, GlobalSymbol]:
        ret = self.raw_args[index]
        assert isinstance(ret, AddressMode) or isinstance(ret, GlobalSymbol)
        return ret


def deref(
    arg: Union[AddressMode, GlobalSymbol],
    regs: RegInfo,
    stack_info: StackInfo,
    type: Optional[str]
) -> Expression:
    if isinstance(arg, AddressMode):
        if arg.lhs is None:
            location = 0
        else:
            assert isinstance(arg.lhs, NumberLiteral)  # macros were removed
            location = arg.lhs.value
        if arg.rhs.register_name == 'sp':
            return stack_info.get_stack_var(location, type=type)
        else:
            # Struct member is being dereferenced.
            return StructAccess(struct_var=regs[arg.rhs], offset=location, type=type)
    else:
        # Keep GlobalSymbol's as-is.
        assert isinstance(arg, GlobalSymbol)
        return arg

def literal_expr(arg: Argument) -> Expression:
    if isinstance(arg, GlobalSymbol):
        return arg
    if isinstance(arg, NumberLiteral):
        return Literal(arg.value)
    if isinstance(arg, BinOp):
        return BinaryOp(left=literal_expr(arg.lhs), op=arg.op,
                right=literal_expr(arg.rhs), in_type='int')
    assert False, f'argument {arg} must be a literal'

def load_upper(args: InstrArgs, regs: RegInfo) -> Expression:
    expr = args.imm(1)
    if isinstance(expr, BinaryOp) and expr.op == '>>':
        # Something like "lui REG (lhs >> 16)". Just take "lhs".
        assert expr.right == Literal(16)
        return expr.left
    elif isinstance(expr, Literal):
        # Something like "lui 0x1", meaning 0x10000. Shift left and return.
        return BinaryOp(left=expr, op='<<', right=Literal(16), in_type='int')
    else:
        # Something like "lui REG %hi(arg)", but we got rid of the macro.
        return expr

def handle_ori(args: InstrArgs, regs: RegInfo) -> Expression:
    expr = args.imm(1)
    if isinstance(expr, BinaryOp):
        # Something like "ori REG (lhs & 0xFFFF)". We (hopefully) already
        # handled this above, but let's put lhs into this register too.
        assert expr.op == '&'
        assert expr.right == Literal(0xFFFF)
        return expr.left
    else:
        # Regular bitwise OR.
        return BinaryOp(left=args.reg(0), op='|', right=expr, in_type='int')

def handle_addi(args: InstrArgs, regs: RegInfo, stack_info: StackInfo) -> Expression:
    if len(args.raw_args) == 2:
        # Used to be "addi reg1 reg2 %lo(...)", but we got rid of the macro.
        # Return the former argument of the macro.
        return args.imm(1)
    else:
        assert len(args.raw_args) == 3
        if args.reg_ref(1).register_name == 'sp':
            # Adding to sp, i.e. passing an address.
            lit = args.imm(2)
            assert isinstance(lit, Literal)
            return UnaryOp(op='&', expr=LocalVar(lit.value, stack_info=stack_info),
                    in_type=None, type='void*')
        else:
            # Regular binary addition.
            return BinaryOp(left=args.reg(1), op='+', right=args.imm(2), in_type='int')

def make_store(args: InstrArgs, stack_info: StackInfo, type: str) -> Optional[StoreStmt]:
    source_reg = args.reg_ref(0)
    source_val = args.reg(0)
    target = args.memory_ref(1)
    if (source_reg.register_name in (SPECIAL_REGS + ARGUMENT_REGS) and
            isinstance(target, AddressMode) and
            target.rhs.register_name == 'sp'):
        # TODO: This isn't really right, but it helps get rid of some pointless stores.
        return None
    return StoreStmt(
        source=source_val, dest=deref(target, args.regs, stack_info, type=type), type=type
    )

def convert_to_float(num: int):
    if num == 0:
        return 0.0
    rep =  f'{num:032b}'  # zero-padded binary representation of num
    dec = lambda x: int(x, 2)  # integer value for num
    sign = dec(rep[0])
    expo = dec(rep[1:9])
    frac = dec(rep[9:])
    return ((-1) ** sign) * (2 ** (expo - 127)) * (frac / (2 ** 23) + 1)

def strip_macros(arg: Argument) -> Argument:
    if isinstance(arg, Macro):
        return arg.argument
    elif isinstance(arg, AddressMode) and isinstance(arg.lhs, Macro):
        assert arg.lhs.macro_name == 'lo'  # %hi(...)(REG) doesn't make sense.
        return arg.lhs.argument
    else:
        return arg


def translate_block_body(
    block: Block, regs: RegInfo, stack_info: StackInfo
) -> BlockInfo:
    """
    Given a block and current register contents, return a BlockInfo containing
    the translated AST for that block.
    """

    InstrMap = Dict[str, Callable[[InstrArgs], Expression]]
    StoreInstrMap = Dict[str, Callable[[InstrArgs], Optional[StoreStmt]]]
    MaybeInstrMap = Dict[str, Callable[[InstrArgs], Optional[Expression]]]
    PairInstrMap = Dict[str, Callable[[InstrArgs], Tuple[Optional[Expression], Optional[Expression]]]]

    cases_source_first_expression: StoreInstrMap = {
        # Storage instructions
        'sb': lambda a: make_store(a, stack_info, type='s8'),
        'sh': lambda a: make_store(a, stack_info, type='s16'),
        'sw': lambda a: make_store(a, stack_info, type='int'),
        # Floating point storage/conversion
        'swc1': lambda a: make_store(a, stack_info, type='f32'),
        'sdc1': lambda a: make_store(a, stack_info, type='f64'),
    }
    cases_source_first_register: InstrMap = {
        # Floating point moving instruction
        'mtc1': lambda a: a.reg_nouse(0),
    }
    cases_branches: MaybeInstrMap = {
        # Branch instructions/pseudoinstructions
        # TODO! These are wrong. (Are they??)
        'b': lambda a: None,
        'beq': lambda a:  BinaryOp(left=a.reg(0), op='==', right=a.reg(1), in_type=None),
        'bne': lambda a:  BinaryOp(left=a.reg(0), op='!=', right=a.reg(1), in_type=None),
        'beqz': lambda a: BinaryOp(left=a.reg(0), op='==', right=Literal(0), in_type='s32'),
        'bnez': lambda a: BinaryOp(left=a.reg(0), op='!=', right=Literal(0), in_type='s32'),
        'blez': lambda a: BinaryOp(left=a.reg(0), op='<=', right=Literal(0), in_type='s32'),
        'bgtz': lambda a: BinaryOp(left=a.reg(0), op='>',  right=Literal(0), in_type='s32'),
        'bltz': lambda a: BinaryOp(left=a.reg(0), op='<',  right=Literal(0), in_type='s32'),
        'bgez': lambda a: BinaryOp(left=a.reg(0), op='>=', right=Literal(0), in_type='s32'),
    }
    cases_float_branches: MaybeInstrMap = {
        # Floating-point branch instructions
        # We don't have to do any work here, since the condition bit was already set.
        'bc1t': lambda a: None,
        'bc1f': lambda a: None,
    }
    cases_jumps: MaybeInstrMap = {
        # Unconditional jumps
        'jal': lambda a: a.imm(0),  # not sure what arguments!
        'jr':  lambda a: None       # not sure what to return!
    }
    cases_float_comp: InstrMap = {
        # Floating point comparisons
        'c.eq.s': lambda a: BinaryOp(left=a.reg(0), op='==', right=a.reg(1), in_type='f32'),
        'c.le.s': lambda a: BinaryOp(left=a.reg(0), op='<=', right=a.reg(1), in_type='f32'),
        'c.lt.s': lambda a: BinaryOp(left=a.reg(0), op='<',  right=a.reg(1), in_type='f32'),
    }
    cases_special: InstrMap = {
        # Handle these specially to get better debug output.
        # These should be unspecial'd at some point by way of an initial
        # pass-through, similar to the stack-info acquisition step.
        'lui':  lambda a: load_upper(a, regs),
        'ori':  lambda a: handle_ori(a, regs),
        'addi': lambda a: handle_addi(a, regs, stack_info),
    }
    cases_hi_lo: PairInstrMap = {
        # Div and mul output results to LO/HI registers.
        # Be careful to only read registers once, using get_nouse for one of the reads.
        # TODO: this isn't right, we need a better system for uses
        'div': lambda a: (BinaryOp(left=a.reg(1), op='%', right=a.reg(2), in_type='s32'), # hi
                          BinaryOp(left=a.reg_nouse(1), op='/',
                              right=a.reg_nouse(2), in_type='s32')), # lo
        'multu': lambda a: (None, # hi
            BinaryOp(left=a.reg(0), op='*', right=a.reg(1), in_type='int')), # lo
    }
    cases_destination_first: InstrMap = {
        # Flag-setting instructions
        'slt': lambda a:  BinaryOp(left=a.reg(1), op='<', right=a.reg(2), in_type='int'),
        'slti': lambda a: BinaryOp(left=a.reg(1), op='<', right=a.imm(2), in_type='int'),
        # LRU (non-floating)
        'addu': lambda a: BinaryOp(left=a.reg(1), op='+', right=a.reg(2), in_type='int'),
        'subu': lambda a: BinaryOp(left=a.reg(1), op='-', right=a.reg(2), in_type='int'),
        'negu': lambda a: UnaryOp(op='-', expr=a.reg(1), in_type='s32', type='s32'),
        # Hi/lo register uses (used after division/multiplication)
        'mfhi': lambda a: regs.get_nouse(Register('hi')),
        'mflo': lambda a: regs.get_nouse(Register('lo')),
        # Floating point arithmetic
        'div.s': lambda a: BinaryOp(left=a.reg(1), op='/', right=a.reg(2), in_type='f32'),
        'mul.s': lambda a: BinaryOp(left=a.reg(1), op='*', right=a.reg(2), in_type='f32'),
        # Floating point conversions
        'cvt.d.s': lambda a: Cast(to_type='f64', expr=a.reg(1), in_type='f32'),
        'cvt.s.d': lambda a: Cast(to_type='f32', expr=a.reg(1), in_type='f64'),
        'cvt.w.d': lambda a: Cast(to_type='s32', expr=a.reg(1), in_type='f64'),
        'cvt.s.u': lambda a: Cast(to_type='f32', expr=a.reg(1), in_type='u32'),
        'trunc.w.s': lambda a: Cast(to_type='s32', expr=a.reg(1), in_type='f32'),
        'trunc.w.d': lambda a: Cast(to_type='s32', expr=a.reg(1), in_type='f64'),
        # Bit arithmetic
        'and': lambda a: BinaryOp(left=a.reg(1), op='&', right=a.reg(2), in_type='int'),
        'or': lambda a:  BinaryOp(left=a.reg(1), op='^', right=a.reg(2), in_type='int'),
        'xor': lambda a: BinaryOp(left=a.reg(1), op='^', right=a.reg(2), in_type='int'),

        'andi': lambda a: BinaryOp(left=a.reg(1), op='&',  right=a.imm(2), in_type='int'),
        'xori': lambda a: BinaryOp(left=a.reg(1), op='^',  right=a.imm(2), in_type='int'),
        'sll': lambda a:  BinaryOp(left=a.reg(1), op='<<', right=a.imm(2), in_type='int'),
        'sllv': lambda a: BinaryOp(left=a.reg(1), op='<<', right=a.reg(2), in_type='int'),
        'srl': lambda a:  BinaryOp(left=a.reg(1), op='>>', right=a.imm(2),
                in_type='u32', right_in_type='int'),
        'srlv': lambda a:  BinaryOp(left=a.reg(1), op='>>', right=a.reg(2),
                in_type='u32', right_in_type='int'),
        # Move pseudoinstruction
        'move': lambda a: a.reg_nouse(1),
        # Floating point moving instructions
        'mfc1': lambda a: a.reg_nouse(1),
        'mov.s': lambda a: a.reg_nouse(1),
        'mov.d': lambda a: a.reg_nouse(1),
        # Loading instructions
        'li': lambda a: a.imm(1),
        'lb': lambda a: deref(a.memory_ref(1), regs, stack_info, type='s8'),
        'lh': lambda a: deref(a.memory_ref(1), regs, stack_info, type='s16'),
        'lw': lambda a: deref(a.memory_ref(1), regs, stack_info, type=None),
        'lbu': lambda a: deref(a.memory_ref(1), regs, stack_info, type='u8'),
        'lhu': lambda a: deref(a.memory_ref(1), regs, stack_info, type='u16'),
        'lwu': lambda a: deref(a.memory_ref(1), regs, stack_info, type=None),
        # Floating point loading instructions
        'lwc1': lambda a: deref(a.memory_ref(1), regs, stack_info, type='f32'),
        'ldc1': lambda a: deref(a.memory_ref(1), regs, stack_info, type='f64'),
    }
    # TODO!
    cases_repeats = {
        # Addition and division, unsigned vs. signed, doesn't matter (?)
        'addiu': 'addi',
        'divu': 'div',
        # Single-precision float addition is the same as regular addition.
        'add.s': 'addu',
        'sub.s': 'subu',
        'neg.s': 'negu',
        # TODO: Deal with doubles differently.
        'add.d': 'addu',
        'sub.d': 'subu',
        'neg.d': 'negu',
        'div.d': 'div.s',
        'mul.d': 'mul.s',
        # Casting (the above applies here too)
        'cvt.d.w': 'cvt.d.s',
        'cvt.s.w': 'cvt.s.d',
        'cvt.w.s': 'cvt.w.d',
        # Floating point comparisons (the above also applies)
        'c.lt.d': 'c.lt.s',
        'c.eq.d': 'c.eq.s',
        'c.le.d': 'c.le.s',
        # Arithmetic right-shifting
        'sra': 'srl',
        'srav': 'srlv',
        # Flag setting.
        'sltiu': 'slti',
        'sltu': 'slt',
        # FCSR-using instructions
        'ctc1': 'mtc1',
        'cfc1': 'mfc1',
    }

    def propagate_types(expr: Optional[Expression]):
        if isinstance(expr, BinaryOp):
            set_type(expr.left, expr.in_type)
            set_type(expr.right, expr.right_in_type)
        elif isinstance(expr, UnaryOp) or isinstance(expr, Cast):
            set_type(expr.expr, expr.in_type)
        elif isinstance(expr, StructAccess):
            set_type(expr.struct_var, 'void*')

    to_write: List[Statement] = []
    def set_reg(reg: Register, value: Optional[Expression]) -> None:
        propagate_types(value)
        if value is not None and not is_trivial_expr(value):
            value = EvalOnceExpr(value, exact=False,
                    var=stack_info.temp_var_generator(reg.register_name))
            stmt = EvalOnceStmt(value)
            stack_info.temp_vars.append(stmt)
            to_write.append(stmt)
        if isinstance(value, PassedInArg):
            # Wrap the argument to better distinguish arguments we are called
            # with from arguments passed to subroutines.
            value = WrapperExpr(value)
        regs[reg] = value

    subroutine_args: List[Tuple[Expression, int]] = []
    branch_condition: Optional[Expression] = None
    for instr in block.instructions:
        # Save the current mnemonic.
        mnemonic = instr.mnemonic
        if mnemonic == 'nop':
            continue
        if mnemonic in cases_repeats:
            # Determine "true" mnemonic.
            mnemonic = cases_repeats[mnemonic]

        # HACK: Remove any %hi(...) or %lo(...) macros; we will just put the
        # full value into each intermediate register, because this really
        # doesn't affect program behavior almost ever.
        if (mnemonic in ['addi', 'addiu'] and len(instr.args) == 3 and
                isinstance(instr.args[2], Macro)):
            mnemonic = 'set_hilo'
        simpler_args = list(map(strip_macros, instr.args))

        args = InstrArgs(simpler_args, regs)

        # Figure out what code to generate!
        if mnemonic in cases_source_first_expression:
            # Store a value in a permanent place.
            to_store = cases_source_first_expression[mnemonic](args)
            if to_store is not None:
                set_type(to_store.source, to_store.type)
                propagate_types(to_store.dest)
            if to_store is not None and isinstance(to_store.dest, SubroutineArg):
                # About to call a subroutine with this argument.
                subroutine_args.append((to_store.source, to_store.dest.value))
            elif to_store is not None:
                if isinstance(to_store.dest, LocalVar):
                    stack_info.add_local_var(to_store.dest)
                # This needs to be written out.
                to_write.append(to_store)

                # TODO: figure something out here
                # If the expression is used again, read it from memory instead of
                # duplicating it -- duplicated C expressions are probably rare,
                # and the expression might be invalidated sooner (e.g. if it
                # refers to the store destination).
                if False and not isinstance(to_store.source, Literal):
                    regs.replace_occurrences(to_store.source, to_store.dest)
                    subroutine_args = [
                        (replace_occurrences(val, to_store.source, to_store.dest), pos)
                        for (val, pos) in subroutine_args]

        elif mnemonic in cases_source_first_register:
            # Just 'mtc1'. It's reversed, so we have to specially handle it.
            set_reg(args.reg_ref(1), cases_source_first_register[mnemonic](args))

        elif mnemonic in cases_branches:
            assert branch_condition is None
            branch_condition = cases_branches[mnemonic](args)
            propagate_types(branch_condition)

        elif mnemonic in cases_float_branches:
            assert branch_condition is None
            cond_bit = regs[Register('condition_bit')]
            if mnemonic == 'bc1t':
                branch_condition = cond_bit
            else:
                assert mnemonic == 'bc1f'
                branch_condition = negate(cond_bit)
            propagate_types(branch_condition)

        elif mnemonic in cases_jumps:
            result = cases_jumps[mnemonic](args)
            if result is None:
                # Return from the function.
                assert mnemonic == 'jr'
                # TODO: Maybe assert ReturnNode?
                # TODO: Figure out what to return. (Look through $v0 and $f0)
            else:
                # Function call. Well, let's double-check:
                assert mnemonic == 'jal'
                target = args.imm(0)
                assert isinstance(target, GlobalSymbol)
                func_args: List[Expression] = []
                # At most one of $f12 and $a0 may be passed, and at most one of
                # $f14 and $a1. We could try to figure out which ones, and cap
                # the function call at the point where a register is empty, but
                # for now we'll leave that for manual fixup.
                for register in map(Register, ['f12', 'f14', 'a0', 'a1', 'a2', 'a3']):
                    # The latter check verifies that the register is not a
                    # placeholder. This might give false positives for the
                    # first function call if an argument passed in the same
                    # position as we received it, but that's impossible to do
                    # anything about without access to function signatures.
                    # (In other cases PassedInArg's get wrapped by WrapperExpr.)
                    if register in regs and not isinstance(regs.get_raw(register), PassedInArg):
                        func_args.append(regs[register])
                # Add the arguments after a3.
                subroutine_args.sort(key=lambda a: a[1])
                for arg in subroutine_args:
                    func_args.append(arg[0])
                # Reset subroutine_args, for the next potential function call.
                subroutine_args = []

                call = FuncCall(target.symbol_name, func_args)
                call_temp = EvalOnceExpr(call, exact=True,
                        var=stack_info.temp_var_generator('ret'))

                # Clear out caller-save registers, for clarity and to ensure
                # that argument regs don't get passed into the next function.
                regs.clear_caller_save_regs()

                # We don't know what this function's return register is,
                # be it $v0, $f0, or something else, so this hack will have
                # to do. (TODO: handle it...)
                regs[Register('f0')] = call_temp
                regs[Register('v0')] = call_temp

                # Write out the function call
                to_write.append(EvalOnceStmt(call_temp))

        elif mnemonic in cases_float_comp:
            set_reg(Register('condition_bit'),
                    cases_float_comp[mnemonic](args))

        elif mnemonic in cases_special:
            output = args.reg_ref(0)
            res = cases_special[mnemonic](args)

            # Keep track of all local variables that we take addresses of.
            if (output.register_name != 'sp' and
                    isinstance(res, UnaryOp) and
                    res.op == '&' and
                    isinstance(res.expr, LocalVar) and
                    res.expr not in stack_info.local_vars):
                stack_info.add_local_var(res.expr)

            set_reg(output, res)

        elif mnemonic in cases_hi_lo:
            hi, lo = cases_hi_lo[mnemonic](args)
            set_reg(Register('hi'), hi)
            set_reg(Register('lo'), lo)

        elif mnemonic in cases_destination_first:
            set_reg(args.reg_ref(0), cases_destination_first[mnemonic](args))

        else:
            assert False, f"I don't know how to handle {mnemonic}!"

    return BlockInfo(to_write, branch_condition, regs)


def translate_graph_from_block(
    node: Node, regs: RegInfo, stack_info: StackInfo
) -> None:
    """
    Given a FlowGraph node and a dictionary of register contents, give that node
    its appropriate BlockInfo (which contains the AST of its code).
    """
    # Do not recalculate block info.
    if node.block.block_info is not None:
        return

    if DEBUG:
        print(f'\nNode in question: {node.block}')

    # Translate the given node and discover final register states.
    try:
        block_info = translate_block_body(node.block, regs, stack_info)
        if DEBUG:
            print(block_info)
    except Exception as e:  # TODO: handle issues better
        if IGNORE_ERRORS:
            traceback.print_exc()
            error_stmt = CommentStmt('Error: ' + str(e).replace('\n', ''))
            block_info = BlockInfo([error_stmt], None,
                    RegInfo(contents={}, stack_info=stack_info))
        else:
            raise e

    node.block.add_block_info(block_info)

    # Translate descendants recursively. Pass a copy of the dictionary since
    # it will be modified.
    if isinstance(node, BasicNode):
        translate_graph_from_block(node.successor, regs.copy(), stack_info)
    elif isinstance(node, ConditionalNode):
        translate_graph_from_block(node.conditional_edge, regs.copy(), stack_info)
        translate_graph_from_block(node.fallthrough_edge, regs.copy(), stack_info)
    else:
        assert isinstance(node, ReturnNode)
        block_info.return_reg = \
            block_info.final_register_states.get_raw(Register('return_reg'))

@attr.s
class FunctionInfo:
    stack_info: StackInfo = attr.ib()
    flow_graph: FlowGraph = attr.ib()

def translate_to_ast(function: Function) -> FunctionInfo:
    """
    Given a function, produce a FlowGraph that both contains control-flow
    information and has AST transformations for each block of code and
    branch condition.
    """
    # Initialize info about the function.
    flow_graph: FlowGraph = build_callgraph(function)
    stack_info = get_stack_info(function, flow_graph.nodes[0])

    initial_regs: Dict[Register, Expression] = {
        Register('zero'): Literal(0, type='zero'),
        Register('a0'): PassedInArg(0, stack_info=stack_info, guessed_type='int'),
        Register('a1'): PassedInArg(1, stack_info=stack_info, guessed_type='int'),
        Register('a2'): PassedInArg(2, stack_info=stack_info, guessed_type=None),
        Register('a3'): PassedInArg(3, stack_info=stack_info, guessed_type=None),
        Register('f12'): PassedInArg(0, stack_info=stack_info, guessed_type='f32'),
        Register('f14'): PassedInArg(1, stack_info=stack_info, guessed_type='f32'),
        **{Register(name): GlobalSymbol(name) for name in SPECIAL_REGS}
    }

    print(stack_info)
    print('\nNow, we attempt to translate:')
    start_node = flow_graph.nodes[0]
    start_reg = RegInfo(contents=initial_regs, stack_info=stack_info)
    translate_graph_from_block(start_node, start_reg, stack_info)
    return FunctionInfo(stack_info, flow_graph)
