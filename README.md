# Uniform arrays for Julia

| **License**                     | **Build Status**                                                | **Code Coverage**                                                   |
|:--------------------------------|:----------------------------------------------------------------|:--------------------------------------------------------------------|
| [![][license-img]][license-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] | [![][coveralls-img]][coveralls-url] [![][codecov-img]][codecov-url] |

The UniformArrays package provides multi-dimensional arrays which behave
like regular arrays but whose elements all have the same value.  The main
advantage of such arrays is that they are very light in terms of memory:
their storage requirement is `O(1)` whatever their size instead of `O(n)`
for a usual array of `n` elements.

## Usage

Call:

```julia
A = UniformArray(val, siz)
```

to create an array `A` which behaves as an immutable array of size `siz` whose
values are all `val`.  The array dimensions may be specified as
multiple arguments.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val`
for all linear indices `i` in the range `1:length(A)`.

A statement like `A[i] = val` is not implemented as uniform arrays are
considered as immutable.  You may call `MutableUniformArray(val,siz)` to
create a uniform array, say `B`, whose element value can be changed:

```julia
B = MutableUniformArray(val, siz)
```

A statement like `B[i] = val` is allowed but changes the value of all the
elements of `B`.

Apart from all values being the same, uniform arrays should behaves like
ordinary Julia arrays.


## Installation

UniformArrays is not yet an [offical Julia package](https://pkg.julialang.org/)
but it is easy to install it from Julia as explained below.


### Using the package manager


At the [REPL of
Julia](https://docs.julialang.org/en/stable/manual/interacting-with-julia/),
hit the `]` key to switch to the package manager REPL (you should get a
`... pkg>` prompt) and type:

```julia
pkg> add https://github.com/emmt/UniformArrays.jl
```

where `pkg>` represents the package manager prompt and `https` protocol has
been assumed; if `ssh` is more suitable for you, then type:

```julia
pkg> add git@github.com:emmt/UniformArrays.jl
```

instead.  To check whether the UniformArrays package works correctly, type:

```julia
pkg> test UniformArrays
```

Later, to update to the last version (and run tests), you can type:

```julia
pkg> update UniformArrays
pkg> build UniformArrays
pkg> test UniformArrays
```

If something goes wrong, it may be because you already have an old version of
UniformArrays.  Uninstall UniformArrays as follows:

```julia
pkg> rm UniformArrays
pkg> gc
pkg> add https://github.com/emmt/UniformArrays.jl
```

before re-installing.

To revert to Julia's REPL, hit the `Backspace` key at the `... pkg>` prompt.


### Installation in scripts

To install UniformArrays in a Julia script, write:

```julia
if VERSION >= v"0.7.0-"
    using Pkg
end
Pkg.add(PackageSpec(url="https://github.com/emmt/UniformArrays.jl", rev="master"));
```

or with `url="git@github.com:emmt/UniformArrays.jl"` if you want to use `ssh`.

This also works from the Julia REPL.

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/UniformArrays.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/UniformArrays.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.org/emmt/UniformArrays.jl.svg?branch=master
[travis-url]: https://travis-ci.org/emmt/UniformArrays.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/UniformArrays.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/UniformArrays-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/UniformArrays.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/UniformArrays.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/UniformArrays.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/UniformArrays.jl?branch=master
