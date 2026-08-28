import re
from dataclasses import dataclass
from dataclasses import field

from .statistics import ALL_STATS


def _tokenize(expr: str) -> list[tuple]:
    """Tokenize a single TABLE dimension expression into a flat list of tokens."""
    pattern = re.compile(
        r"(?P<fmt>format)\s*=\s*(?P<fmt_spec>[0-9]+[.,][0-9]+[_s]*)"
        r"|(?P<denom><[^>]*>)"
        r"|(?P<labeled>[A-Za-z_][A-Za-z0-9_%]*)\s*=\s*"
        r'(?:"(?P<dq_label>[^"]*)"|\'(?P<sq_label>[^\']*)\')'
        r"|(?P<name>[A-Za-z_][A-Za-z0-9_%]*)"
        r"|(?P<op>[*()])"
        r"|(?P<space>\s+)"
    )

    tokens = []
    pos = 0

    for m in pattern.finditer(expr):
        if m.start() != pos:
            invalid = expr[pos : m.start()]
            raise SyntaxError(
                f"Unexpected character(s) {invalid!r} at position {pos}"
            )

        pos = m.end()

        if m.group("fmt"):
            tokens.append(("FMT", m.group("fmt_spec")))
        elif m.group("denom"):
            inner = m.group("denom")[1:-1].strip()
            tokens.append(("DENOM", inner))
        elif m.group("labeled"):
            label = (
                m.group("dq_label")
                if m.group("dq_label") is not None
                else m.group("sq_label")
            )
            tokens.append(("NAME", m.group("labeled"), label))
        elif m.group("name"):
            tokens.append(("NAME", m.group("name"), None))
        elif m.group("op"):
            tokens.append(("OP", m.group("op")))
        elif m.group("space"):
            tokens.append(("SP", " "))

    if pos != len(expr):
        invalid = expr[pos:]
        raise SyntaxError(
            f"Unexpected character(s) {invalid!r} at position {pos}"
        )

    cleaned = []
    for token in tokens:
        if token[0] == "SP" and cleaned and cleaned[-1][0] == "SP":
            continue
        cleaned.append(token)

    while cleaned and cleaned[0][0] == "SP":
        cleaned.pop(0)

    while cleaned and cleaned[-1][0] == "SP":
        cleaned.pop()

    return cleaned


@dataclass
class DimNode:
    """A node in the parsed TABLE expression tree.

    Attributes
    ----------
    kind : str
        One of:
          'var'    — a measure or class variable name, or a statistic keyword
          'all'    — the ALL/TOTAL marginal-total keyword
          'cross'  — a * b  (children = [a, b])
          'concat' — a b    (children = [a, b, ...])
          'group'  — (...)  (children = [inner_node])

    name : str or None
        The original token text (uppercased for stat/all keywords).

    label : str or None
        The display label from name='Label' syntax.
        None  → use the default (name or 'TOTAL')
        ''    → suppress the label level in the header entirely

    fmt : str or None
        Raw format specification string from *format=W.D[_s] syntax,
        e.g. '7,1' or '12.0s'.  Set by _apply_fmt() and read by
        _classify_path() to populate spec['fmt'].

    denom : str or None
        Denominator definition from PCTN<...> / PCTSUM<...> syntax,
        e.g. 'income' or 'gender all'.  Read by _classify_path() to
        populate spec['denom_def'], which routes the cell to
        _compute_custom_pct().

    children : list of DimNode
        Sub-nodes for cross/concat/group kinds.
    """

    kind: str
    name: str | None = None
    label: str | None = None
    fmt: str | None = None
    denom: str | None = None
    children: list["DimNode"] = field(default_factory=list)

    def display_label(self) -> str:
        """Display label."""
        if self.label is None:
            return self.name if self.name else ""
        return self.label

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the node."""
        suffix = f"='{self.label}'" if self.label is not None else ""
        if self.kind == "var":
            return f"{self.name}{suffix}"
        if self.kind == "all":
            return f"ALL{suffix}"
        sep = " * " if self.kind == "cross" else " "
        inner = sep.join(repr(c) for c in self.children)
        if self.kind == "group":
            return f"({inner})"
        return inner


class _Parser:
    def __init__(self, tokens: list[tuple]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _skip_sp(self):
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] == "SP":
            self.pos += 1

    def peek(self) -> tuple | None:
        p = self.pos
        while p < len(self.tokens) and self.tokens[p][0] == "SP":
            p += 1
        return self.tokens[p] if p < len(self.tokens) else None

    def consume_op(self, val: str):
        self._skip_sp()
        t = self.tokens[self.pos]
        if t != ("OP", val):
            raise SyntaxError(f"Expected operator '{val}', got {t}")
        self.pos += 1

    def parse(self) -> DimNode:
        node = self._parse_concat()
        self._skip_sp()

        if self.pos != len(self.tokens):
            raise SyntaxError(
                f"Unexpected token: {self.tokens[self.pos]}"
            )

        return node

    def _parse_concat(self) -> DimNode:
        nodes = [self._parse_cross()]
        while True:
            saved = self.pos
            while self.pos < len(self.tokens) and self.tokens[self.pos][0] == "SP":
                self.pos += 1
            if self.pos >= len(self.tokens):
                break
            nxt = self.tokens[self.pos]
            if nxt[0] == "OP" and nxt[1] in (")", "*"):
                self.pos = saved
                break
            nodes.append(self._parse_cross())
        return nodes[0] if len(nodes) == 1 else DimNode(kind="concat", children=nodes)

    def _parse_cross(self) -> DimNode:
        nodes = [self._parse_atom()]
        while True:
            p = self.peek()
            if p and p[0] == "OP" and p[1] == "*":
                # Look ahead past the '*' (and any space) to see if a FMT
                # token follows. If so, this '*format=...' applies to the
                # PRECEDING node (e.g. mean*format=7.1), not a new atom.
                saved = self.pos
                self._skip_sp()
                self.pos += 1  # consume '*'
                self._skip_sp()
                if self.pos < len(self.tokens) and self.tokens[self.pos][0] == "FMT":
                    fmt_spec = self.tokens[self.pos][1]
                    self.pos += 1
                    self._apply_fmt(nodes[-1], fmt_spec)
                    continue
                # Not a format suffix — restore and parse as a normal cross atom
                self.pos = saved
                self._skip_sp()
                self.pos += 1
                nodes.append(self._parse_atom())
            else:
                break
        return nodes[0] if len(nodes) == 1 else DimNode(kind="cross", children=nodes)

    def _apply_fmt(self, node: DimNode, fmt_spec: str):
        """Apply a format= spec to a node.

        For a leaf node (var/all), the format is set directly on it - this
        is the simple "mean*format=7,1" case.

        For a group node (i.e. node came from "(...)" - e.g. the result of
        "(mean gmean)*format=7,1"), the format must be propagated to every
        LEAF inside the group's subtree instead of being set on the group
        node itself. This is necessary because _expand_node flattens away
        group/cross/concat wrapper nodes when building leaf paths, so any
        .fmt set only on the group node would never be seen by
        _classify_path - it has to live on the var/all leaves directly.

        A leaf's own format (set via "var*format=..." written explicitly
        inside the group) takes precedence and is NOT overwritten - this
        lets "(mean*format=7,1 gmean)*format=8,2" give gmean the outer 8,2
        while mean keeps its own explicit 7,1.
        """
        if node.kind in ("var", "all"):
            if node.fmt is None:
                node.fmt = fmt_spec
            return
        # group / cross / concat: recurse into every child
        for child in node.children:
            self._apply_fmt(child, fmt_spec)

    def _parse_atom(self) -> DimNode:
        self._skip_sp()
        if self.pos >= len(self.tokens):
            raise SyntaxError("Unexpected end of expression")
        t = self.tokens[self.pos]
        if t[0] == "OP" and t[1] == "(":
            self.pos += 1
            inner = self._parse_concat()
            self.consume_op(")")
            return DimNode(kind="group", children=[inner])
        if t[0] == "NAME":
            self.pos += 1
            name_upper = t[1].upper()
            label = t[2]
            # Consume an immediately following FMT token if present
            fmt = self._consume_fmt()
            denom = self._consume_denom()
            if name_upper in ("ALL", "TOTAL"):
                return DimNode(
                    kind="all", name="ALL", label=label, fmt=fmt, denom=denom
                )
            return DimNode(kind="var", name=t[1], label=label, fmt=fmt, denom=denom)
        raise SyntaxError(f"Unexpected token: {t}")

    def _consume_fmt(self) -> str | None:
        """Consume and return a FMT token."""
        p = self.pos
        # Allow one optional space between token and format=
        if p < len(self.tokens) and self.tokens[p][0] == "SP":
            p += 1
        if p < len(self.tokens) and self.tokens[p][0] == "FMT":
            self.pos = p + 1
            return self.tokens[p][1]
        return None

    def _consume_denom(self) -> str | None:
        """Consume and return a DENOM token.

        (e.g. the 'income' from
        pctsum<income>) immediately after the current position, or None.
        """
        p = self.pos
        if p < len(self.tokens) and self.tokens[p][0] == "SP":
            p += 1
        if p < len(self.tokens) and self.tokens[p][0] == "DENOM":
            self.pos = p + 1
            return self.tokens[p][1]
        return None


def _split_dimensions(expr: str) -> list[str]:
    """Split on top-level commas (not inside parentheses or quoted strings).

    A comma that is the DECIMAL SEPARATOR in a format=W,D[_|s] spec
    (immediately following the width digits of "format=", e.g.
    "format=7,2" or "format=7,2_") is NOT treated as a dimension separator.
    """
    parts, current, depth = [], [], 0
    in_str = False
    str_char = None
    i = 0
    while i < len(expr):
        ch = expr[i]
        if not in_str and ch in ("'", '"'):
            in_str = True
            str_char = ch
            current.append(ch)
        elif in_str and ch == str_char:
            in_str = False
            str_char = None
            current.append(ch)
        elif in_str:
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            # Check if this comma is the decimal separator in "format=<digits>,"
            so_far = "".join(current)
            if re.search(r"format\s*=\s*[0-9]+$", so_far):
                current.append(ch)  # decimal comma in format=W,D — not a separator
            else:
                parts.append("".join(current))
                current = []
        else:
            current.append(ch)
        i += 1
    if current:
        parts.append("".join(current))
    return parts


def parse_table(table_str: str) -> tuple:
    """Parse table."""
    dims = _split_dimensions(table_str)
    if len(dims) > 2:
        raise ValueError(
            "TABLE supports at most 2 dimensions (row, col). The page dimension is not supported."
        )
    result = []
    for dim in dims:
        tokens = _tokenize(dim.strip())
        result.append(_Parser(tokens).parse())
    return tuple(result)


def _expand_node(node: DimNode) -> list[list[DimNode]]:
    from itertools import product as iproduct

    if node.kind in ("var", "all"):
        return [[node]]
    if node.kind == "group":
        return _expand_node(node.children[0])
    if node.kind == "concat":
        result = []
        for child in node.children:
            result.extend(_expand_node(child))
        return result
    if node.kind == "cross":
        child_paths = [_expand_node(c) for c in node.children]
        return [
            [leaf for path in combo for leaf in path]
            for combo in iproduct(*child_paths)
        ]
    raise ValueError(f"Unknown node kind: {node.kind}")


def _expand_node_with_branch(node: DimNode):
    """Like _expand_node.

    But additionally returns a top-level branch index for
    each path, used to order specs that come from a TOP-LEVEL concatenation
    (space-separated dimension root) in written left-to-right order.

    Only the OUTERMOST node matters: if the dimension root is itself a
    'concat' (e.g. "origin ALL='Total'" or "origin * type ALL='Subtotal'"
    which parses as concat(cross(origin,type), all)), each top-level child
    gets its own branch index (0, 1, 2, ...) in written order. All paths
    expanded from that child share that branch index.

    If the root is NOT a concat (e.g. pure cross/group like
    "(n rowpctn)*(all age)*(all inc)"), every path gets branch 0 — meaning
    the positional lexicographic sort (via path_order) is solely
    responsible for ordering.

    Returns: list of (branch_index, path) tuples.
    """
    if node.kind == "concat":
        result = []
        for branch_idx, child in enumerate(node.children):
            for path in _expand_node(child):
                result.append((branch_idx, path))
        return result
    return [(0, path) for path in _expand_node(node)]


def _classify_path(
    path: list[DimNode], measure_list: list[str], groupby_list: list[str]
):
    measure_map = {m.upper(): m for m in measure_list}
    groupby_map = {g.upper(): g for g in groupby_list}

    group_keys = []
    var = None
    var_label = None
    stat = None
    stat_label = None
    has_all = False
    all_label = None

    for node in path:
        upper = node.name.upper() if node.name else ""
        if node.kind == "all":
            has_all = True
            all_label = node.label
        elif upper in groupby_map:
            orig = groupby_map[upper]
            lbl = node.label if node.label is not None else orig
            group_keys.append((orig, lbl))
        elif upper in measure_map:
            orig = measure_map[upper]
            var = orig
            var_label = node.label if node.label is not None else orig
        elif upper in ALL_STATS:
            stat = upper
            stat_label = node.label if node.label is not None else upper
        else:
            raise ValueError(
                f"Token {node.name!r} not found in measure=, groupby=, or known statistics.\n"
                f"Tip: if your label uses the same quote character as the surrounding "
                f"Python string, Python will terminate the string early.\n"
                f"Known statistics: {sorted(ALL_STATS)}"
            )

    path_order = []
    for node in path:
        upper = node.name.upper() if node.name else ""
        if node.kind == "all":
            path_order.append(
                ("all", node.label if node.label is not None else "TOTAL", None)
            )
        elif upper in groupby_map:
            orig_name = groupby_map[upper]
            lbl = node.label if node.label is not None else orig_name
            # Store original column name as third element so _key_to_label
            # can detect whether the label was explicitly renamed by the user.
            path_order.append(("group", lbl, orig_name))
        elif upper in measure_map:
            lbl = node.label if node.label is not None else measure_map[upper]
            path_order.append(("var", lbl, None))
        elif upper in ALL_STATS:
            lbl = node.label if node.label is not None else upper
            path_order.append(("stat", lbl, None))

    # Collect any format specs from the path nodes.
    # The innermost (last) non-None fmt wins so e.g. stat*format=7.1
    # overrides a measure-level format.
    fmt = None
    for node in path:
        if node.fmt is not None:
            fmt = node.fmt

    # Collect denom_def from any node that carries one (stat node with <...>)
    denom_def = None
    for node in path:
        if node.denom is not None:
            denom_def = node.denom

    return {
        "group_keys": group_keys,
        "var": var,
        "var_label": var_label,
        "stat": stat,
        "stat_label": stat_label,
        "has_all": has_all,
        "all_label": all_label,
        "path_order": path_order,
        "fmt": fmt,
        "denom_def": denom_def,
    }
