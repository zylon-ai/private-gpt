from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from workflows import Context

    AnyContext = Context[Any]
else:
    from workflows import Context

    AnyContext = Context
