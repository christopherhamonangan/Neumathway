import math
import copy
import re
from fractions import Fraction

class ASTNode:
    def __init__(self, token_type, val=None, left=None, right=None, node_id=0):
        self.token_type = token_type
        self.val = val
        self.left = left
        self.right = right
        self.node_id = node_id

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_reducible(self):
        if self.token_type in ['NUM', 'x', 'y', '=']:
            return False
        unary_set = {'sqrt', 'abs', 'pow2', 'sin', 'cos', 'tan', 'ln', 'exp', 
                     'circle_area', 'fact', 'quad', 'diff', 'int', 'det', 'trans', 'lim'}
        if self.token_type in unary_set:
            return self.left is not None and self.left.token_type == 'NUM'
        return (self.left is not None and self.left.token_type == 'NUM' and 
                self.right is not None and self.right.token_type == 'NUM')

    def to_string(self):
        if self.token_type == 'NUM':
            if isinstance(self.val, Fraction) and self.val.denominator == 1:
                return str(self.val.numerator)
            return str(self.val)
        if self.token_type in ['x', 'y']:
            return self.token_type
        
        unary_set = {'sqrt', 'abs', 'pow2', 'sin', 'cos', 'tan', 'ln', 'exp', 
                     'circle_area', 'fact', 'quad', 'diff', 'int', 'det', 'trans', 'lim'}
        if self.token_type in unary_set:
            return f"{self.token_type}({self.left.to_string()})"
        if self.token_type in ['pow', 'hypot', 'gcd', 'matmul', 'spldv', 'det']:
            if self.token_type == 'det':
                return f"det({self.left.to_string()})"
            return f"{self.token_type}({self.left.to_string()}, {self.right.to_string()})"
        if self.token_type == '=':
            return f"{self.left.to_string()} = {self.right.to_string()}"
        return f"({self.left.to_string()} {self.token_type} {self.right.to_string()})"

class MathTextParser:
    @staticmethod
    def tokenize(expr_str):
        expr_str = expr_str.replace(" ", "").lower()
        expr_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', expr_str)
        expr_str = expr_str.replace("[", "(").replace("]", ")")
        pattern = r'(circle_area|hypot|pow2|sqrt|abs|sin|sin|cos|tan|exp|pow|gcd|fact|ln|quad|diff|int|lim|spldv|det|trans|matmul|\d+\.?\d*|[a-zA-Z]|==|=|[\+\-\*\/\(\),])'
        tokens = re.findall(pattern, expr_str)
        return ['=' if t == '==' else t for t in tokens]

    @classmethod
    def parse(cls, expr_str):
        if '=' in expr_str and not expr_str.startswith('spldv'):
            parts = expr_str.split('=')
            lhs_node = cls._parse_sub_expr(cls.tokenize(parts[0]))
            rhs_node = cls._parse_sub_expr(cls.tokenize(parts[1]))
            return ASTNode('=', left=lhs_node, right=rhs_node)
        return cls._parse_sub_expr(cls.tokenize(expr_str))

    @classmethod
    def _parse_sub_expr(cls, tokens):
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
        output_queue = []
        op_stack = []
        i = 0
        unary_fns = {'sqrt', 'abs', 'pow2', 'sin', 'cos', 'tan', 'ln', 'exp', 'circle_area', 'fact', 'quad', 'diff', 'int', 'trans', 'lim'}
        binary_fns = {'pow', 'hypot', 'gcd', 'matmul', 'spldv', 'det'}

        while i < len(tokens):
            t = tokens[i]
            if re.match(r'^\d+\.?\d*$', t):
                output_queue.append(ASTNode('NUM', val=Fraction(int(float(t)), 1)))
            elif t in ['x', 'y']:
                output_queue.append(ASTNode(t))
            elif t in unary_fns or t in binary_fns:
                op_stack.append(t)
            elif t in precedence:
                if t == '-' and (i == 0 or tokens[i-1] in ['(', '+', '-', '*', '/', ',']):
                    output_queue.append(ASTNode('NUM', val=Fraction(0, 1)))
                while (op_stack and op_stack[-1] in precedence and
                       precedence[op_stack[-1]] >= precedence[t]):
                    cls._pop_operator(op_stack, output_queue)
                op_stack.append(t)
            elif t == '(':
                op_stack.append('(')
            elif t == ',':
                while op_stack and op_stack[-1] != '(':
                    cls._pop_operator(op_stack, output_queue)
            elif t == ')':
                while op_stack and op_stack[-1] != '(':
                    cls._pop_operator(op_stack, output_queue)
                if op_stack and op_stack[-1] == '(':
                    op_stack.pop()
                if op_stack and (op_stack[-1] in unary_fns or op_stack[-1] in binary_fns):
                    cls._pop_operator(op_stack, output_queue)
            i += 1
        while op_stack:
            cls._pop_operator(op_stack, output_queue)
        return output_queue[0]

    @staticmethod
    def _pop_operator(op_stack, output_queue):
        op = op_stack.pop()
        unary_fns = {'sqrt', 'abs', 'pow2', 'sin', 'cos', 'tan', 'ln', 'exp', 'circle_area', 'fact', 'quad', 'diff', 'int', 'trans', 'lim'}
        if op in unary_fns:
            arg = output_queue.pop() if output_queue else ASTNode('NUM', val=Fraction(0,1))
            output_queue.append(ASTNode(op, left=arg))
        elif op in {'pow', 'hypot', 'gcd', 'matmul', 'spldv', 'det'}:
            r = output_queue.pop() if output_queue else ASTNode('NUM', val=Fraction(0,1))
            l = output_queue.pop() if output_queue else ASTNode('NUM', val=Fraction(0,1))
            output_queue.append(ASTNode(op, left=l, right=r))
        else:
            r = output_queue.pop() if output_queue else ASTNode('NUM', val=Fraction(0,1))
            l = output_queue.pop() if output_queue else ASTNode('NUM', val=Fraction(0,1))
            output_queue.append(ASTNode(op, left=l, right=r))

class SymbolicUniversalEngine:
    @staticmethod
    def execute_step(node):
        op = node.token_type
        if op == 'sqrt':
            val_in = node.left.val
            if val_in < 0: return False, None, "DomainError: Negative square root"
            num_sqrt = math.isqrt(int(val_in.numerator))
            den_sqrt = math.isqrt(int(val_in.denominator))
            if num_sqrt * num_sqrt == val_in.numerator and den_sqrt * den_sqrt == val_in.denominator:
                res = Fraction(num_sqrt, den_sqrt)
            else:
                res = Fraction(int(math.isqrt(int(float(val_in)))), 1)
            return True, res, "OK"
            
        if op == 'abs': return True, abs(node.left.val), "OK"
        if op == 'pow2': return True, node.left.val ** 2, "OK"
        
        if op == 'sin':
            deg = int(float(node.left.val))
            rad_map = {0: Fraction(0, 1), 30: Fraction(1, 2), 90: Fraction(1, 1)}
            return True, rad_map.get(deg, Fraction(int(round(math.sin(math.radians(deg))*100)), 100)), "OK"
            
        if op == 'cos':
            deg = int(float(node.left.val))
            rad_map = {0: Fraction(1, 1), 60: Fraction(1, 2), 90: Fraction(0, 1)}
            return True, rad_map.get(deg, Fraction(int(round(math.cos(math.radians(deg))*100)), 100)), "OK"
            
        if op == 'tan':
            deg = int(float(node.left.val))
            if deg == 90: return False, None, "DomainError: Tan(90°) undefined"
            rad_map = {0: Fraction(0, 1), 45: Fraction(1, 1)}
            return True, rad_map.get(deg, Fraction(int(round(math.tan(math.radians(deg))*100)), 100)), "OK"
            
        if op == 'ln':
            v = float(node.left.val)
            if v <= 0: return False, None, "DomainError: Logarithm of non-positive number"
            return True, Fraction(int(round(math.log(v)*100)), 100), "OK"
            
        if op == 'exp':
            return True, Fraction(int(round(math.exp(float(node.left.val)))), 1), "OK"
            
        if op == 'circle_area':
            r = node.left.val
            return True, Fraction(int(round(3.14159 * float(r * r))), 1), "OK"
            
        if op == 'fact':
            n = int(float(node.left.val))
            if n < 0 or n > 12: return False, None, "DomainError: Factorial out of safe range (0-12)"
            return True, Fraction(math.factorial(n), 1), "OK"

        if node.right is None:
            return False, None, f"Unary node without handler: {op}"

        l_val, r_val = node.left.val, node.right.val
        if op == '+': return True, l_val + r_val, "OK"
        if op == '-': return True, l_val - r_val, "OK"
        if op == '*': return True, l_val * r_val, "OK"
        if op == '/':
            if r_val == 0: return False, None, "ZeroDivisionError"
            return True, l_val / r_val, "OK"
        if op == 'pow': return True, Fraction(int(float(l_val) ** int(float(r_val))), 1), "OK"
        if op == 'hypot': return True, Fraction(int(round(math.hypot(float(l_val), float(r_val)))), 1), "OK"
        if op == 'gcd': return True, Fraction(math.gcd(int(float(l_val)), int(float(r_val))), 1), "OK"

        return False, None, f"Unknown Op: {op}"

    @staticmethod
    def solve_quadratic(a, b, c):
        D = b**2 - 4*a*c
        if D < 0: return False, None, [f"Discriminant D = {D} < 0 (Complex Roots)"]
        sqrt_D = math.isqrt(D)
        x1 = Fraction(-b + sqrt_D, 2*a)
        x2 = Fraction(-b - sqrt_D, 2*a)
        steps = [
            f"Canonical Form: ({a})x^2 + ({b})x + ({c}) = 0",
            f"Compute Discriminant D = b^2 - 4ac: ({b})^2 - 4({a})({c}) = {D}",
            f"Quadratic Formula: x = [-({b}) ± √{D}] / [2({a})]",
            f"Exact Real Roots: x1 = {x1}, x2 = {x2}"
        ]
        return True, (x1, x2), steps

    @staticmethod
    def solve_matrix_det(a, b, c, d):
        det_val = a * d - b * c
        steps = [
            f"Matrix A: [[{a}, {b}], [{c}, {d}]]",
            f"Main Diagonal (a * d): {a} * {d} = {a * d}",
            f"Secondary Diagonal (b * c): {b} * {c} = {b * c}",
            f"Determinant: det(A) = ({a * d}) - ({b * c}) = {det_val}"
        ]
        return True, det_val, steps

    @staticmethod
    def solve_spldv(a1, b1, c1, a2, b2, c2):
        D = a1 * b2 - a2 * b1
        if D == 0: return False, None, ["Singular System (D = 0)"]
        Dx = c1 * b2 - c2 * b1
        Dy = a1 * c2 - a2 * c1
        x = Fraction(Dx, D)
        y = Fraction(Dy, D)
        steps = [
            f"System: 1) {a1}x + {b1}y = {c1}  |  2) {a2}x + {b2}y = {c2}",
            f"Main Determinant D = a1*b2 - a2*b1 : ({a1})*({b2}) - ({a2})*({b1}) = {D}",
            f"Determinant X (Dx): ({c1})*({b2}) - ({c2})*({b1}) = {Dx}",
            f"Determinant Y (Dy): ({a1})*({c2}) - ({a2})*({c1}) = {Dy}",
            f"Exact Solution: x = {x}, y = {y}"
        ]
        return True, (x, y), steps

def find_reducible_nodes(curr):
    redexes = []
    if curr is None: return redexes
    if curr.is_reducible(): redexes.append(curr)
    redexes.extend(find_reducible_nodes(curr.left))
    redexes.extend(find_reducible_nodes(curr.right))
    return redexes

def reduce_tree_at(root, target_node, new_val):
    if root is target_node: return ASTNode('NUM', val=new_val)
    def replace(curr):
        if curr is None: return
        if curr.left is target_node: curr.left = ASTNode('NUM', val=new_val); return
        if curr.right is target_node: curr.right = ASTNode('NUM', val=new_val); return
        replace(curr.left); replace(curr.right)
    replace(root)
    return root

def evaluate_expression_tree(ast_root):
    curr_tree = copy.deepcopy(ast_root)
    history = [curr_tree.to_string()]
    
    for _ in range(25):
        if curr_tree.token_type == 'NUM': break
        redexes = find_reducible_nodes(curr_tree)
        if not redexes: break
        
        target = redexes[0]
        is_valid, res_val, msg = SymbolicUniversalEngine.execute_step(target)
        if not is_valid:
            return history, None, msg
            
        curr_tree = reduce_tree_at(curr_tree, target, res_val)
        history.append(f"{curr_tree.to_string()}  [Reduction: {target.token_type} -> {res_val}]")
        if curr_tree.token_type == 'NUM': break
        
    final_val = curr_tree.val if curr_tree.token_type == 'NUM' else None
    return history, final_val, "OK"