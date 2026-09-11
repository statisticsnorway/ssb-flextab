import re
from dataclasses import dataclass
from dataclasses import field
from itertools import product as iproduct
from typing import Any

from .statistics import ALL_STATS

# A token is a 3-tuple (kind, value, label). The label slot is only ever
# populated for "NAME" tokens (via name='Label' syntax); every other kind
# carries None in that position, so all tokens share one uniform shape.
Token = tuple[str, str, str | None]

_TOKEN_PATTERN = re.compile(
    r"(?P<fmt>format)\s*=\s*(?P<fmt_spec>[0-9]+[.,][0-9]+[_s]*)"
    r"|(?P<denom><[^>]*>)"
    r"|(?P<labeled>[A-Za-z_][A-Za-z0-9_%]*)\s*=\s*"
    r'(?:"(?P<dq_label>[^"]*)"|\'(?P<sq_label>[^\']*)\')'
    r"|(?P<n>[A-Za-z_][A-Za-z0-9_%]*)"
    r"|(?P<op>[*()])"
    r"|(?P<space>\s+)"
)


def _match_to_token(m: re.Match[str]) -> Token | None:
    """Convert a single regex match into a Token, or None for a discarded match."""
    if m.group("fmt"):
        return ("FMT", m.group("fmt_spec"), None)
    if m.group("denom"):
        inner = m.group("denom")[1:-1].strip()
        return ("DENOM", inner, None)
    if m.group("labeled"):
        label = (
            m.group("dq_label")
            if m.group("dq_label") is not None
            else m.group("sq_label")
        )
        return ("NAME", m.group("labeled"), label)
    if m.group("n"):
        return ("NAME", m.group("n"), None)
    if m.group("op"):
        return ("OP", m.group("op"), None)
    if m.group("space"):
        return ("SP", " ", None)
    return None


def _scan_tokens(expr: str) -> list[Token]:
    """Scan `expr` with the token pattern, raising on any unmatched gap."""
    tokens: list[Token] = []
    pos = 0

    for m in _TOKEN_PATTERN.finditer(expr):
        if m.start() != pos:
            invalid = expr[pos : m.start()]
            raise SyntaxError(f"Unexpected character(s) {invalid!r} at position {pos}")
        pos = m.end()
        token = _match_to_token(m)
        if token is not None:
            tokens.append(token)

    if pos != len(expr):
        invalid = expr[pos:]
        raise SyntaxError(f"Unexpected character(s) {invalid!r} at position {pos}")

    return tokens


def _clean_tokens(tokens: list[Token]) -> list[Token]:
    """Collapse consecutive SP tokens and strip leading/trailing SP."""
    cleaned: list[Token] = []
    for token in tokens:
        if token[0] == "SP" and cleaned and cleaned[-1][0] == "SP":
            continue
        cleaned.append(token)

    while cleaned and cleaned[0][0] == "SP":
        cleaned.pop(0)

    while cleaned and cleaned[-1][0] == "SP":
        cleaned.pop()

    return cleaned


def _tokenize(expr: str) -> list[Token]:
    """Tokenize a single TABLE dimension expression into a flat list of tokens."""
    return _clean_tokens(_scan_tokens(expr))


@dataclass
class DimNode:
    """A node in the parsed TABLE expression tree.

    Attributes
    ----------
    kind : str
        Node type. Supported values are:

        - ``"var"``: a measure or class variable name, or a statistic keyword.
        - ``"all"``: the ``ALL`` or ``TOTAL`` marginal-total keyword.
        - ``"cross"``: a crossed expression such as ``a * b``.
        - ``"concat"``: a concatenated expression such as ``a b``.
        - ``"group"``: a grouped expression such as ``(...)``.

    name : str | None
        Original token text. Statistic and ``ALL``/``TOTAL`` keywords are
        stored in uppercase.

    label : str | None
        Display label supplied using ``name='Label'`` syntax.

        If None, the default label is used. An empty string suppresses the
        corresponding label level entirely.

    fmt : str | None
        Raw format specification from ``*format=W.D[_s]`` syntax, for example
        ``"7,1"`` or ``"12.0s"``.

        The value is used by ``_classify_path()`` when constructing the
        corresponding table specification.

    denom : str | None
        Denominator definition from ``PCTN<...>`` or ``PCTSUM<...>`` syntax,
        for example ``"income"`` or ``"gender all"``.

        The value is used by ``_classify_path()`` when constructing the
        custom-percentage specification.

    children : list[DimNode]
        Child nodes for ``cross``, ``concat``, and ``group`` nodes.
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
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _skip_sp(self) -> None:
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] == "SP":
            self.pos += 1

    def peek(self) -> Token | None:
        p = self.pos
        while p < len(self.tokens) and self.tokens[p][0] == "SP":
            p += 1
        return self.tokens[p] if p < len(self.tokens) else None

    def consume_op(self, val: str) -> None:
        self._skip_sp()
        t = self.tokens[self.pos]
        if t[0] != "OP" or t[1] != val:
            raise SyntaxError(f"Expected operator '{val}', got {t}")
        self.pos += 1

    def parse(self) -> DimNode:
        node = self._parse_concat()
        self._skip_sp()

        if self.pos != len(self.tokens):
            raise SyntaxError(f"Unexpected token: {self.tokens[self.pos]}")

        return node

    def _should_continue_concat(self) -> bool:
        """Advance past whitespace; return whether another concat member follows."""
        saved = self.pos
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] == "SP":
            self.pos += 1
        if self.pos >= len(self.tokens):
            return False
        nxt = self.tokens[self.pos]
        if nxt[0] == "OP" and nxt[1] in (")", "*"):
            self.pos = saved
            return False
        return True

    def _parse_concat(self) -> DimNode:
        nodes = [self._parse_cross()]
        while self._should_continue_concat():
            nodes.append(self._parse_cross())
        return nodes[0] if len(nodes) == 1 else DimNode(kind="concat", children=nodes)

    def _try_consume_trailing_fmt(self, node: DimNode) -> bool:
        """If '*format=...' follows immediately, consume and apply it.

        Returns True if a trailing format was consumed, False if the
        position was restored because no FMT token follows the '*'.
        """
        saved = self.pos
        self._skip_sp()
        self.pos += 1  # consume '*'
        self._skip_sp()
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] == "FMT":
            fmt_spec = self.tokens[self.pos][1]
            self.pos += 1
            self._apply_fmt(node, fmt_spec)
            return True
        self.pos = saved
        return False

    def _consume_cross_star(self, nodes: list[DimNode]) -> bool:
        """Try to consume one '*'-joined cross step. Returns whether to keep looping."""
        p = self.peek()
        if not (p and p[0] == "OP" and p[1] == "*"):
            return False
        if self._try_consume_trailing_fmt(nodes[-1]):
            return True
        # Not a format suffix — restore and parse as a normal cross atom
        self._skip_sp()
        self.pos += 1
        nodes.append(self._parse_atom())
        return True

    def _parse_cross(self) -> DimNode:
        nodes = [self._parse_atom()]
        while self._consume_cross_star(nodes):
            pass
        return nodes[0] if len(nodes) == 1 else DimNode(kind="cross", children=nodes)

    def _apply_fmt(self, node: DimNode, fmt_spec: str) -> None:
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


def _is_format_decimal_comma(chars_so_far: str) -> bool:
    """Whether a comma at this position is the decimal separator in format=W,D."""
    return bool(re.search(r"format\s*=\s*[0-9]+$", chars_so_far))


@dataclass
class _DimSplitScanner:
    """Stateful character-by-character scanner used by ``_split_dimensions``.

    Tracks whether we're inside a quoted string or parentheses so that a
    comma there is not mistaken for a dimension separator.
    """

    parts: list[str] = field(default_factory=list)
    current: list[str] = field(default_factory=list)
    depth: int = 0
    in_str: bool = False
    str_char: str | None = None

    def flush(self) -> None:
        self.parts.append("".join(self.current))
        self.current = []

    def _consume_comma(self, ch: str) -> None:
        so_far = "".join(self.current)
        if _is_format_decimal_comma(so_far):
            self.current.append(ch)  # decimal comma in format=W,D — not a separator
        else:
            self.flush()

    def consume(self, ch: str) -> None:
        if not self.in_str and ch in ("'", '"'):
            self.in_str = True
            self.str_char = ch
            self.current.append(ch)
        elif self.in_str and ch == self.str_char:
            self.in_str = False
            self.str_char = None
            self.current.append(ch)
        elif self.in_str:
            self.current.append(ch)
        elif ch == "(":
            self.depth += 1
            self.current.append(ch)
        elif ch == ")":
            self.depth -= 1
            self.current.append(ch)
        elif ch == "," and self.depth == 0:
            self._consume_comma(ch)
        else:
            self.current.append(ch)


def _split_dimensions(expr: str) -> list[str]:
    """Split on top-level commas (not inside parentheses or quoted strings).

    A comma that is the DECIMAL SEPARATOR in a format=W,D[_|s] spec
    (immediately following the width digits of "format=", e.g.
    "format=7,2" or "format=7,2_") is NOT treated as a dimension separator.
    """
    scanner = _DimSplitScanner()
    for ch in expr:
        scanner.consume(ch)
    if scanner.current:
        scanner.flush()
    return scanner.parts


def parse_table(table_str: str) -> tuple[DimNode, ...]:
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


def _expand_concat(node: DimNode) -> list[list[DimNode]]:
    result = []
    for child in node.children:
        result.extend(_expand_node(child))
    return result


def _expand_cross(node: DimNode) -> list[list[DimNode]]:
    child_paths = [_expand_node(c) for c in node.children]
    return [
        [leaf for path in combo for leaf in path] for combo in iproduct(*child_paths)
    ]


def _expand_node(node: DimNode) -> list[list[DimNode]]:
    if node.kind in ("var", "all"):
        return [[node]]
    if node.kind == "group":
        return _expand_node(node.children[0])
    if node.kind == "concat":
        return _expand_concat(node)
    if node.kind == "cross":
        return _expand_cross(node)
    raise ValueError(f"Unknown node kind: {node.kind}")


def _expand_node_with_branch(node: DimNode) -> list[tuple[int, list[DimNode]]]:
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


def _classify_node(
    node: DimNode,
    groupby_map: dict[str, str],
    measure_map: dict[str, str],
) -> tuple[str, str | None]:
    """Classify one path node, returning (category, orig_name_or_None).

    category is one of: 'all', 'group', 'var', 'stat'.
    """
    upper = node.name.upper() if node.name else ""
    if node.kind == "all":
        return "all", None
    if upper in groupby_map:
        return "group", groupby_map[upper]
    if upper in measure_map:
        return "var", measure_map[upper]
    if upper in ALL_STATS:
        return "stat", upper
    raise ValueError(
        f"Token {node.name!r} not found in measure=, groupby=, or known statistics.\n"
        f"Tip: if your label uses the same quote character as the surrounding "
        f"Python string, Python will terminate the string early.\n"
        f"Known statistics: {sorted(ALL_STATS)}"
    )


def _default_label(label: str | None, default: str) -> str:
    return label if label is not None else default


def _classify_all_entry(node: DimNode) -> tuple[str | None, tuple[str, str, None]]:
    """Build the (all_label, path_order entry) pair for an 'all' node."""
    return node.label, ("all", _default_label(node.label, "TOTAL"), None)


def _classify_group_entry(
    node: DimNode, orig: str
) -> tuple[tuple[str, str], tuple[str, str, str]]:
    """Build the (group_keys entry, path_order entry) pair for a 'group' node."""
    lbl = _default_label(node.label, orig)
    return (orig, lbl), ("group", lbl, orig)


def _classify_var_entry(node: DimNode, orig: str) -> tuple[str, tuple[str, str, None]]:
    """Build the (var_label, path_order entry) pair for a 'var' node."""
    lbl = _default_label(node.label, orig)
    return lbl, ("var", lbl, None)


def _classify_stat_entry(node: DimNode, orig: str) -> tuple[str, tuple[str, str, None]]:
    """Build the (stat_label, path_order entry) pair for a 'stat' node."""
    lbl = _default_label(node.label, orig)
    return lbl, ("stat", lbl, None)


def _classify_path_nodes(
    path: list[DimNode],
    measure_list: list[str],
    groupby_list: list[str],
) -> tuple[
    list[tuple[str, str]],
    str | None,
    str | None,
    str | None,
    str | None,
    bool,
    str | None,
    list[tuple[str, str, str | None]],
]:
    """Classify each node in `path` in a single pass.

    Builds both the summary fields (group_keys/var/stat/has_all/...) and
    path_order.
    """
    measure_map = {m.upper(): m for m in measure_list}
    groupby_map = {g.upper(): g for g in groupby_list}

    group_keys: list[tuple[str, str]] = []
    path_order: list[tuple[str, str, str | None]] = []
    var = None
    var_label = None
    stat = None
    stat_label = None
    has_all = False
    all_label = None

    for node in path:
        category, orig = _classify_node(node, groupby_map, measure_map)
        if category == "all":
            has_all = True
            all_label, all_path_entry = _classify_all_entry(node)
            path_order.append(all_path_entry)
        elif category == "group":
            assert orig is not None  # guaranteed by _classify_node for "group"
            group_entry, group_path_entry = _classify_group_entry(node, orig)
            group_keys.append(group_entry)
            path_order.append(group_path_entry)
        elif category == "var":
            assert orig is not None  # guaranteed by _classify_node for "var"
            var = orig
            var_label, var_path_entry = _classify_var_entry(node, orig)
            path_order.append(var_path_entry)
        else:  # 'stat'
            assert orig is not None  # guaranteed by _classify_node for "stat"
            stat = orig
            stat_label, stat_path_entry = _classify_stat_entry(node, orig)
            path_order.append(stat_path_entry)

    return group_keys, var, var_label, stat, stat_label, has_all, all_label, path_order


def _collect_fmt(path: list[DimNode]) -> str | None:
    """Return the innermost (last) non-None fmt.

    E.g. stat*format=7.1 overrides a measure-level format.
    """
    fmt = None
    for node in path:
        if node.fmt is not None:
            fmt = node.fmt
    return fmt


def _collect_denom(path: list[DimNode]) -> str | None:
    """Collect denom_def from any node that carries one (stat node with <...>)."""
    denom_def = None
    for node in path:
        if node.denom is not None:
            denom_def = node.denom
    return denom_def


def _classify_path(
    path: list[DimNode], measure_list: list[str], groupby_list: list[str]
) -> dict[str, Any]:
    (
        group_keys,
        var,
        var_label,
        stat,
        stat_label,
        has_all,
        all_label,
        path_order,
    ) = _classify_path_nodes(path, measure_list, groupby_list)

    return {
        "group_keys": group_keys,
        "var": var,
        "var_label": var_label,
        "stat": stat,
        "stat_label": stat_label,
        "has_all": has_all,
        "all_label": all_label,
        "path_order": path_order,
        "fmt": _collect_fmt(path),
        "denom_def": _collect_denom(path),
    }
