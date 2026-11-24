"""Generate the documented exponent profiles without changing the engine."""

from importlib.resources import files
from pathlib import Path

from ..artifacts import sha256

EXPONENTS = tuple((a, b) for a in (0.5, 1.0) for b in (0.5, 1.0, 2.0))


def profile_name(a: float = 1, b: float = 1) -> str:
    if (a, b) not in EXPONENTS:
        raise ValueError(f"Vespa bound/max profiles support only {EXPONENTS}")
    return "slotspace" if a == b == 1 else f"slotspace_a{a:g}_b{b:g}".replace(".", "_")


def write_application(out: Path) -> dict[str, str]:
    """Write into a new application directory; hosts.xml targets a dedicated container."""
    out.mkdir()
    (out / "schemas").mkdir()
    source = files("slotspace.vespa_app").joinpath("app")
    for name in ("services.xml", "hosts.xml"):
        (out / name).write_bytes(source.joinpath(name).read_bytes())
    base = source.joinpath("schemas", "doc.sd").read_text(encoding="utf-8")
    extra = []
    for a, b in EXPONENTS:
        name = profile_name(a, b)
        if name == "slotspace":
            continue
        extra.append(f"""
    rank-profile {name} inherits slotspace {{
        function gate() {{
            expression: reduce(query(rk) * map(attribute(tk), f(p)(pow(p, {a:g}))), max, key)
        }}
        function textpart() {{
            expression: reduce(gate * map(simf, f(x)(pow(max(0, x), {b:g}))), max, fact)
        }}
    }}
""")
    (out / "schemas" / "doc.sd").write_text(base.rsplit("}", 1)[0] + "".join(extra) + "}\n", encoding="utf-8")
    return {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
