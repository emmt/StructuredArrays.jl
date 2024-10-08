# User visible changes in `StructuredArrays` package

## Version 0.2.18

- Replace private methods `to_dim`, `to_axis`, `to_size`, and `to_axes` by `as_array_dim`,
  `as_array_axis`, `as_array_size`, and `as_array_axes` provided by `TypeUtils` (version ≥
  1.4).

## Version 0.2.17

- For a uniform or structured array `A`, `copy(A)` and `deepcopy(A)` simply yield `A` if
  it is immutable.

- Axes specified as instances of `Base.OneTo` are stored by length.

## Version 0.2.16

- Fix constructors for vector/matrix aliases.

## Version 0.2.15

- Update compatibility for `TypeUtils`.

## Version 0.2.14

- Optimize methods `reverse` and `unique` for uniform arrays.

- Ordinal ranges may be used to define axes provided they have unit-step.

- Axes of structured arrays may be specified as a mixture of integers (assumed
  to be dimension lengths) and integer-valued unit-ranges.

## Version 0.2.13

- Optimize reductions operations `minimum`, `maximum`, `extrema`, `all`, `any`,
  `sum`, `prod`, `count`, `findmin`, and `findmax` for uniform arrays with
  syntax `op([f=identity], A; dims=:)` where `op` is the reduction operation,
  `f` is an optional function, `A` is a uniform array, and keyword `dims` is
  the list of dimensions over which to reduce (all by default). As a result,
  these operations are much faster for uniform arrays with a number of
  operations scaling as `O(1)` instead of `O(length(A))`.

## Version 0.2.12

- Structured arrays may have offset axes.

## Version 0.2.11

- Structured arrays no longer directly store their length.

## Version 0.2.10

- More uniform and consistent API for constructors, using `@eval` to encode
  most constructors and aliases.

- Index style parameter of structured arrays must be a concrete one:
  `IndexLinear` or `IndexCartesian`.

## Version 0.2.9

This version introduces some bug fixes and some incompatibilities:

- Slicing of a uniform array no longer yields a uniform array although a view
  does. Having syntax `A[i...]`, with `A` a uniform array and `i...` sub-array
  indices, yields a uniform array was a bad idea that broke things like
  broadcasting for uniform arrays.

- Non-exported method `StructuredArrays.value` can be applied to a uniform
  array to retrieve the value of all its elements without indexing.

Other changes:

- Use `TypeUtils` instead of `ArrayTools`.

- Fix compatibility.


## Version 0.2.8

- Syntax `A[i...]`, with `A` a **uniform array** and `i...` sub-array indices,
  yields a uniform array (or a scalar). A similar optimization is done for
  `view(A, i...)` with `A` an **immutable uniform array**. Note that `A[i...]`
  may return a scalar, not `view(A, i...)`.

- New non-exported method `StructuredArrays.parameterless(T)` which yields type
  `T` without parameter specifications. For example:
  `StructuredArrays.parameterless(Vector{Float32})` yields `Array`.


## Version 0.2.7

- Element type `T` of an instance `A` of `StructuredArray{T,N}` is used to
  convert the value returned by the embedded function, say `func`, so `A[i...]`
  evaluates to `convert(T,func(i...))::T`.

- Fix `A[i]` result for a uniform array `A` when index `i` is not a single integer.

- `A[:]` and `A[r]` are optimized for a uniform array `A` and a range `r`.

- Zero-dimensional structured arrays are allowed.


## Version 0.2.5

- New fast uniform arrays built by `FastUniformArray(val,dims)` which are
  immutable uniform arrays but with the elements value being part of the
  signature so that this value is known at compile time.

- A few optimizations for uniform arrays of Booleans.


## Version 0.2.3

- Use package `ArrayTools`.


## Version 0.2.0

- For a mutable uniform array `A`, restrict `A[i] = val` to cases where `i`
  represents all the index range of `A`.
