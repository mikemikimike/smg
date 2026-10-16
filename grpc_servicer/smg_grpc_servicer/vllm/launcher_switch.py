"""The servicer switch, installed into vLLM's gRPC launcher by the launcher's
own import of this package.

vLLM's launcher (``vllm/entrypoints/launchers/grpc_server.py``, behind
``vllm serve --grpc`` and ``python -m vllm.entrypoints.grpc_server``) imports
this package's servicer classes at module level, then defines ``serve_grpc``,
which builds an AsyncLLM before it instantiates them. The Rust path has to
take the process before that engine exists, so the switch must run ahead of
``serve_grpc``; the launcher's import of this package is the first moment this
package runs inside it, and :func:`install_launcher_switch` uses that moment.

The launcher module is mid-import then (``serve_grpc`` not yet defined), so
its class is swapped for a ``ModuleType`` subclass whose first attribute
access after the import (the CLI's ``from ... import serve_grpc``, the
deprecated shim's ``import main, serve_grpc``) rebinds ``serve_grpc`` in the
module dict to :func:`with_servicer_switch`'s wrapper: the three lines an
in-source hook would carry, with the launcher's own ``args``. The launcher's
``main()`` reads the rebound name from its globals, so every documented entry
runs the switch. A launcher imported after this package (something imported
the servicer first, or a launcher that imports it lazily) is caught by a
meta-path finder that binds the switch as the launcher loads.

What the switch binds is specific to the launcher's shape: the module under
one of :data:`LAUNCHER_MODULES`, defining :data:`HOOK_ATTRIBUTE`. A launcher
file executed directly as ``__main__`` is the one form neither hook reaches,
and a launcher of another shape (a release that moved the module or renamed
the coroutine) is never bound. Rather than let such a launcher build the
engine and fail in the Python servicer's guard once the model has loaded,
:func:`require_launcher_hook` checks the hook as soon as the Rust
implementation is selected, from the launcher's own argument parse (the
plugin's parse step, :mod:`smg_grpc_servicer.vllm.plugin`), and refuses with
one line naming the installed vLLM and the launchers the switch binds. A
launcher that reaches the Python servicer regardless is refused there
(:func:`smg_grpc_servicer.vllm.rust.require_python_impl`).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.abc
import importlib.metadata
import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

from smg_grpc_servicer.vllm import SERVICER_IMPL_ENV

LAUNCHER_MODULES = (
    "vllm.entrypoints.launchers.grpc_server",
    "vllm.entrypoints.grpc_server",
)
# The launcher's coroutine the switch binds over.
HOOK_ATTRIBUTE = "serve_grpc"
# The launchers the switch binds, for the refusal line: the coroutine above
# in a module imported under one of LAUNCHER_MODULES, which is how every
# release with the gRPC launcher starts it under `vllm serve <model> --grpc`
# (the module is vllm.entrypoints.grpc_server on 0.28 and 0.29, and
# vllm.entrypoints.launchers.grpc_server from 0.30 on, the old name a shim
# that imports it; checked through 0.31).
SUPPORTED_LAUNCHERS = (
    "the switch binds the serve_grpc of vllm.entrypoints.launchers.grpc_server (vLLM 0.30 and "
    "later) or of vllm.entrypoints.grpc_server (0.28 and 0.29) imported under that name, as "
    "`vllm serve <model> --grpc` does on vLLM 0.28 and later (checked through 0.31)"
)
_SWITCH_MARK = "__smg_servicer_switch__"


def with_servicer_switch(
    serve_grpc: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """``serve_grpc`` behind the flag: Rust takes the process (and its exit
    code), Python runs the launcher's own coroutine."""
    if getattr(serve_grpc, _SWITCH_MARK, False):
        return serve_grpc

    @functools.wraps(serve_grpc)
    async def serve_grpc_with_switch(args: Any, *rest: Any, **kwargs: Any) -> Any:
        from smg_grpc_servicer.rust_lifecycle import ServicerConfigError, configuration_exit
        from smg_grpc_servicer.vllm.rust import (
            configure_logging,
            resolve_servicer_impl,
            serve_rust,
        )

        try:
            if resolve_servicer_impl(args) == "rust":
                raise SystemExit(await serve_rust(args))
        except ServicerConfigError as error:
            # A knob the servicer cannot use: one named ERROR line and exit
            # code 2, not the launcher's traceback.
            configure_logging()
            raise configuration_exit(error) from None
        return await serve_grpc(args, *rest, **kwargs)

    setattr(serve_grpc_with_switch, _SWITCH_MARK, True)
    return serve_grpc_with_switch


def _rebind(namespace: dict[str, Any]) -> None:
    serve_grpc = namespace.get(HOOK_ATTRIBUTE)
    if callable(serve_grpc) and not getattr(serve_grpc, _SWITCH_MARK, False):
        namespace[HOOK_ATTRIBUTE] = with_servicer_switch(serve_grpc)


class _SwitchedLauncher(types.ModuleType):
    """The launcher module once hooked: any attribute access binds the switch
    over a ``serve_grpc`` the module body defined since."""

    def __getattribute__(self, name: str) -> Any:
        if name != "__dict__":
            _rebind(object.__getattribute__(self, "__dict__"))
        return super().__getattribute__(name)


class _SwitchingLoader(importlib.abc.Loader):
    """The launcher's own loader, plus the switch bound right after the module
    body ran (so the name is rebound before anything can call it)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def create_module(self, spec: Any) -> Any:
        create = getattr(self.inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: types.ModuleType) -> None:
        self.inner.exec_module(module)
        _rebind(vars(module))

    def __getattr__(self, name: str) -> Any:  # get_code, get_source, is_package, ...
        return getattr(self.inner, name)


class _LauncherFinder(importlib.abc.MetaPathFinder):
    """For a launcher imported after this package: the real spec from the
    other finders, with the loader wrapped in :class:`_SwitchingLoader`."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in LAUNCHER_MODULES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and not isinstance(spec.loader, _SwitchingLoader):
                    spec.loader = _SwitchingLoader(spec.loader)
                return spec
        return None


def install_launcher_switch() -> list[str]:
    """Hook every launcher module that is importing or imported, and arrange
    for one imported later (this package imported first, or a launcher that
    imports it lazily) to be hooked as it loads. Idempotent; returns the names
    hooked now. A module another extension already replaced with its own
    class is left alone."""
    if not any(isinstance(finder, _LauncherFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LauncherFinder())
    hooked: list[str] = []
    for name in LAUNCHER_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if type(module) is _SwitchedLauncher:
            hooked.append(name)
            continue
        if type(module) is not types.ModuleType:
            continue
        _rebind(vars(module))
        module.__class__ = _SwitchedLauncher
        hooked.append(name)
    return hooked


# ---------------------------------------------------------------------------
# The hook, verified before the engine is built
# ---------------------------------------------------------------------------

# What the switch finds of the launcher (`LauncherHook.state`).
BOUND = "bound"  # the switch is bound over the launcher's serve_grpc
MISSING = "missing"  # a launcher module is loaded and defines no serve_grpc
MAIN = "main"  # the launcher runs as __main__, where no import hook reaches it
ABSENT = "absent"  # no launcher module is loaded, or importable under the known names
PENDING = "pending"  # a launcher module is still importing (it is importing this package)
UNKNOWN = "unknown"  # the launcher's import failed on something its own import reports


@dataclasses.dataclass(frozen=True)
class LauncherHook:
    """What the switch found of the launcher: the ``state`` (one of the
    constants above) and the module it describes, when there is one."""

    state: str
    module: str | None = None

    @property
    def bindable(self) -> bool:
        """Whether the switch binds, or may still bind, on this launcher: the
        hook is bound, the launcher is mid-import (its ``serve_grpc`` comes
        after this package's import, and the hooked module binds it), or its
        import failed on something its own import reports."""
        return self.state in (BOUND, PENDING, UNKNOWN)


def launcher_hook() -> LauncherHook:
    """What the switch finds of the launcher in this process: the known
    modules in ``sys.modules`` (bound, loaded without the hook, or still
    importing), else a launcher running as ``__main__``, else nothing loaded."""
    missing: str | None = None
    pending = False
    for name in LAUNCHER_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        if getattr(getattr(module, "__spec__", None), "_initializing", False):
            pending = True
            continue
        namespace = vars(module)
        _rebind(namespace)
        hook = namespace.get(HOOK_ATTRIBUTE)
        if callable(hook) and getattr(hook, _SWITCH_MARK, False):
            return LauncherHook(BOUND, name)
        missing = missing or name
    if missing is not None:
        return LauncherHook(MISSING, missing)
    if pending:
        return LauncherHook(PENDING)
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    main_name = getattr(main_spec, "name", None)
    if main_name in LAUNCHER_MODULES:
        return LauncherHook(MAIN, main_name)
    return LauncherHook(ABSENT)


def _import_launcher() -> LauncherHook:
    """Import the launcher under the first known name the installed vLLM has
    (what ``vllm serve --grpc`` does next; the finder binds the switch as it
    loads) and report what the switch finds of it. A name the installation
    lacks is skipped; an import that fails on anything else is left to the
    launcher's own import to report."""
    for name in LAUNCHER_MODULES:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            absent = error.name or ""
            if absent and (name == absent or name.startswith(absent + ".")):
                continue
            return LauncherHook(UNKNOWN, name)
        except Exception:  # noqa: BLE001 - the launcher's own import reports its failure
            return LauncherHook(UNKNOWN, name)
        return launcher_hook()
    return LauncherHook(ABSENT)


def require_launcher_hook(import_launcher: bool = True) -> LauncherHook:
    """The hook as the switch finds it, once the Rust implementation is
    selected: bound (or still binding), or a :class:`ServicerConfigError`
    naming what the installed vLLM lacks and the launchers the switch binds.
    A launcher not loaded yet is imported first under its known names when
    ``import_launcher`` is set (``vllm serve --grpc`` imports it right after
    its parse); a launcher running as ``__main__`` is never re-imported."""
    from smg_grpc_servicer.rust_lifecycle import ServicerConfigError

    hook = launcher_hook()
    if hook.state == ABSENT and import_launcher:
        hook = _import_launcher()
    if hook.bindable:
        return hook
    raise ServicerConfigError(refusal_line(hook))


def refusal_line(hook: LauncherHook) -> str:
    """The one line a launcher the switch cannot bind is refused with: the
    installed vLLM and what was found of its launcher, the launchers the
    switch binds, and the way out."""
    version = installed_engine_version()
    installed = f"vLLM {version}" if version else "vLLM (version unknown)"
    way_out = "install a vLLM release the switch binds"
    if hook.state == MAIN:
        found = f"its gRPC launcher {hook.module} runs as __main__, where no import hook reaches it"
        way_out = "start it as `vllm serve <model> --grpc`"
    elif hook.state == MISSING:
        found = f"its gRPC launcher {hook.module} defines no {HOOK_ATTRIBUTE}"
    else:
        found = f"it has no gRPC launcher module under {' or '.join(LAUNCHER_MODULES)}"
    return (
        f"--servicer-impl rust ({SERVICER_IMPL_ENV}=rust) refused: the servicer switch cannot "
        f"bind on the installed {installed}: {found}; {SUPPORTED_LAUNCHERS}; {way_out}, or "
        f"select --servicer-impl python (or unset {SERVICER_IMPL_ENV}) for the Python servicer"
    )


def installed_engine_version() -> str | None:
    """The installed vLLM's version: the imported package's, else the
    distribution's; ``None`` when neither is known."""
    version = getattr(sys.modules.get("vllm"), "__version__", None)
    if not version:
        try:
            version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            version = None
    return str(version) if version else None
