# Python Code Standards

- Generate code for Python 3.12+ using modern, idiomatic syntax, favoring clarity and expressive constructs over legacy patterns
- Prefer explicit named parameters; avoid `**kwargs` except for true pass-through scenarios (e.g., decorators/adapters). If used, document all consumed keys
- Never use mutable defaults (`list`, `dict`, `set`); use `None` and initialize inside the function
- Avoid mutating input arguments unless explicitly documented or clearly indicated by the function name; otherwise return a new object
- Signal errors with specific exceptions; do not use sentinel return values (`None`, `False`, `-1`) unless explicitly required and properly typed
- Require full type annotations on all functions; only use `Optional[T]` when `None` has explicit semantic meaning, not as a generic default