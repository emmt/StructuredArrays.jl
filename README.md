# Structured arrays for Julia

| **License**                     | **Build Status**                                                | **Code Coverage**                                                   |
|:--------------------------------|:----------------------------------------------------------------|:--------------------------------------------------------------------|
| [![][license-img]][license-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] | [![][coveralls-img]][coveralls-url] [![][codecov-img]][codecov-url] |

StructuredArrays is a small [Julia](https://julialang.org/) package which
provides multi-dimensional arrays behaving like regular arrays but whose
elements have the same given value or are computed by applying a given function
to their indices.  The main advantage of such arrays is that they are very
light in terms of memory: their storage requirement is `O(1)` whatever their
size instead of `O(n)` for a usual array of `n` elements.


## Usage

### Uniform arrays

Call:

```julia
A = UniformArray(val, siz)
```

to create an array `A` which behaves as an immutable array of size `siz` whose
values are all `val`.  The array dimensions may be specified as
multiple arguments.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val`
for all linear indices `i` in the range `1:length(A)`.

Statements like `A[i] = val` are however not implemented because uniform
arrays are considered as immutable.  You may call
`MutableUniformArray(val,siz)` to create a uniform array, say `B`, whose
element value can be changed:

```julia
B = MutableUniformArray(val, siz)
```

A statement like `B[i] = val` is allowed to change the value of all the
elements of `B` provided index `i` represents all possible indices
in `B`.  Typically `B[:] = val` or `B[1:end] = val` are accepted but not
`B[1] = val` unless `B` has a single element.

Apart from all values being the same, uniform arrays should behaves like
ordinary Julia arrays.


### Structured arrays

Call:

```julia
A = StructuredArray(fnc, siz)
```

to create a structured array `A` which behaves as an array of size `siz` whose
values are a given function, here `fnc`, of its indices: `A[i]` is computed as
`fnc(i)`.  The array dimensions may be specified as multiple arguments.

An optional leading argument `S` may be used to specifiy another index style
than the default `IndexCartesian`:

```julia
A = StructuredArray(S, fnc, siz)
```

where `S` may be a sub-type of `IndexStyle` or an instance of such a sub-type.
If `S` is `IndexCartesian` (the default), the function `fnc` will be called
with `N` integer arguments, `N` being the number of dimensions.  If `S` is
`IndexCartesian`, the function `fnc` will be called with a single integer
argument.

For instance, the structure of a lower triangular matrix of size `m×n` would be
given by:

```julia
StructuredArray((i,j) -> (i ≥ j), m, n)
```

but with a constant small storage requirement whatever the size of the matrix.

Although the callable object `fnc` may not be a *pure function*, its return
type shall be stable and structured arrays are considered as immutable in the
sense that a statement like `A[i] = val` is not implemented.  The type of the
elements of structured array is guessed by applying `fnc` to the unit index.
The element type, say `T`, may also be explicitely specified:

```julia
StructuredArray{T}([S = IndexCartesian,] fnc, siz)
```


## Installation

StructuredArrays is not yet an [offical Julia package](https://pkg.julialang.org/)
but it is easy to install it from Julia as explained below.


### Using the package manager


At the [REPL of
Julia](https://docs.julialang.org/en/stable/manual/interacting-with-julia/),
hit the `]` key to switch to the package manager REPL (you should get a
`... pkg>` prompt) and type:

```julia
pkg> add https://github.com/emmt/StructuredArrays.jl
```

where `pkg>` represents the package manager prompt and `https` protocol has
been assumed; if `ssh` is more suitable for you, then type:

```julia
pkg> add git@github.com:emmt/StructuredArrays.jl
```

instead.  To check whether the StructuredArrays package works correctly, type:

```julia
pkg> test StructuredArrays
```

Later, to update to the last version (and run tests), you can type:

```julia
pkg> update StructuredArrays
pkg> build StructuredArrays
pkg> test StructuredArrays
```

If something goes wrong, it may be because you already have an old version of
StructuredArrays.  Uninstall StructuredArrays as follows:

```julia
pkg> rm StructuredArrays
pkg> gc
pkg> add https://github.com/emmt/StructuredArrays.jl
```

before re-installing.

To revert to Julia's REPL, hit the `Backspace` key at the `... pkg>` prompt.


### Installation in scripts

To install StructuredArrays in a Julia script, write:

```julia
if VERSION >= v"0.7.0-"
    using Pkg
end
Pkg.add(PackageSpec(url="https://github.com/emmt/StructuredArrays.jl", rev="master"));
```

or with `url="git@github.com:emmt/StructuredArrays.jl"` if you want to use `ssh`.

This also works from the Julia REPL.

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/StructuredArrays.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/StructuredArrays.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.org/emmt/StructuredArrays.jl.svg?branch=master
[travis-url]: https://travis-ci.org/emmt/StructuredArrays.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/StructuredArrays.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/StructuredArrays-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/StructuredArrays.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/StructuredArrays.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/StructuredArrays.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/StructuredArrays.jl?branch=master
