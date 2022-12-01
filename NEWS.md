# User visible changes in `StructuredArrays` package

## Version 0.2.5

- New fast uniform arrays built by `FastUniformArray(val,dims)` which are
  immutable uniform arrays but with the elements value being part of the
  signature so that this value is known at compile time.


## Version 0.2.3

- Use package `ArrayTools`.


## Version 0.2.0

- For a mutable uniform array `A`, restrict `A[i] = val` to cases where `i`
  represents all the index range of `A`.
