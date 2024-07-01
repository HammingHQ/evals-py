import asyncio
from typing import TypeVar, ParamSpec, Callable, Any, Dict, Optional, Mapping
import inspect
import logging

from hamming import get_client


logger = logging.getLogger(__name__)

R = TypeVar("R", covariant=True)
P = ParamSpec("P")

def is_async(func: Callable[P, R]) -> bool:
    return asyncio.iscoroutinefunction(func)

def traced(
    *, 
    name: Optional[str] = None, 
    metadata: Optional[Mapping[str, Any]] = None
):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        def inner(*args: Any, **kwargs: Any):
            client = get_client()
            signature = inspect.signature(func)
            inputs = _get_inputs(signature, *args, **kwargs)
            name_ = name or func.__name__
            with client.monitoring.start_item(
                inputs, name=name_, metadata=metadata
            ) as monitoring_item:
                result = func(*args, **kwargs)
                outputs = result if isinstance(result, dict) else {"result": result}
                monitoring_item.set_output(outputs)
            return result
        
        async def async_inner(*args: Any, **kwargs: Any):
            client = get_client()
            signature = inspect.signature(func)
            inputs = _get_inputs(signature, *args, **kwargs)
            name_ = name or func.__name__
            with client.monitoring.start_item(
                inputs, name=name_, metadata=metadata
            ) as monitoring_item:
                result = await func(*args, **kwargs)
                outputs = result if isinstance(result, dict) else {"result": result}
                monitoring_item.set_output(outputs)
            return result
        
        if is_async(func):
            return async_inner
        else:
            return inner
    return decorator

def _get_inputs(
    sig: inspect.Signature, 
    *args: Any, 
    **kwargs: Any
) -> Dict[str, Any]:
    try:
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()
        arguments = dict(bound_args.arguments)
        arguments.pop("self", None)
        arguments.pop("cls", None)
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                if name in arguments:
                    arguments.update(arguments[name])
                    arguments.pop(name)
        return arguments
    except BaseException as e:
        logger.debug(f"Error getting inputs for signature {sig}: {e}")
        return {}
