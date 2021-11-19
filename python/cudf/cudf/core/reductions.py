# Copyright (c) 2021, NVIDIA CORPORATION.


# TODO: Implement a factory that generates classes encapsulating passthrough
# operations like this. Then we can programmatically create suitable classes
# for reductions, scans, and binops (and perhaps others).


# TODO: The docstring for a class's reductions should be taken from formatting
# the reduce method since it needs to support all the same things.


# TODO: Consider using a pyi file to trick mypy into seeing the monkey-patched
# methods.


class Reducible:
    """Mixin encapsulating for reduction operations.

    Various classes in cuDF support
    `reductions <https://en.wikipedia.org/wiki/Reduction_Operator>`__.  In
    practice the reductions are implemented via dispatch to a lower-level API.
    For example, Frame-like objects dispatch to Columns, which in turn dispatch
    to libcudf implementations. As a result, rather than encoding the logic for
    different types of reductions, most classes can implement all reductions
    in terms of a single function that performs necessary pre- and
    post-processing of a result generated by calling to a lower-level API. This
    class encapsulates that paradigm.

    Notes
    -----
    The documentation for the reductions is generated by formatting the
    docstring for _reduce via `cls._reduce.__doc__.format(op=reduction)`.
    Therefore, subclasses are responsible for writing an appropriate docstring
    for the _reduce method (one that omits documentation of the op parameter).
    """

    _SUPPORTED_REDUCTIONS = {
        "sum",
        "product",
        "min",
        "max",
        "count",
        "size",
        "any",
        "all",
        "sum_of_squares",
        "mean",
        "var",
        "std",
        "median",
        "quantile",
        "argmax",
        "argmin",
        "nunique",
        "nth",
        "collect",
        "unique",
        "prod",
        "idxmin",
        "idxmax",
        "nunique",
        "first",
        "last",
    }

    # TODO: Add a return type.
    def _reduce(self, op: str, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _add_reduction(cls, reduction):
        def op(self, *args, **kwargs):
            return self._reduce(reduction, *args, **kwargs)

        # The default docstring is that
        op.__doc__ = cls._reduce.__doc__.format(op=reduction)
        setattr(cls, reduction, op)

    @classmethod
    def __init_subclass__(cls):
        # Only add the set of reductions that are valid for a particular class.
        # Subclasses may override the methods directly if they need special
        # behavior.
        valid_reductions = set()
        for base_cls in cls.__mro__:
            valid_reductions |= getattr(base_cls, "_VALID_REDUCTIONS", set())

        assert len(valid_reductions - cls._SUPPORTED_REDUCTIONS) == 0, (
            "Invalid requested reductions "
            f"{valid_reductions - cls._SUPPORTED_REDUCTIONS}"
        )
        for reduction in valid_reductions:
            # Allow overriding the operation in the classes.
            if not hasattr(cls, reduction):
                cls._add_reduction(reduction)
