"""

Package `StructuredArrays` implements arrays whose elements have values that
only depend on the indices. Example of such arrays are uniform arrays with
constant values, structured arrays with boolean values indicating whether an
entry is significant.

"""
module StructuredArrays

export
    AbstractStructuredArray,
    AbstractStructuredMatrix,
    AbstractStructuredVector,
    AbstractUniformArray,
    AbstractUniformMatrix,
    AbstractUniformVector,
    FastUniformArray,
    FastUniformMatrix,
    FastUniformVector,
    MutableUniformArray,
    MutableUniformMatrix,
    MutableUniformVector,
    StructuredArray,
    StructuredMatrix,
    StructuredVector,
    UniformArray,
    UniformMatrix,
    UniformVector

using TypeUtils

import Base: @propagate_inbounds, front, tail, to_indices

const SubArrayRange = Union{Integer,OrdinalRange{<:Integer,<:Integer},Colon}
const ConcreteIndexStyle = Union{IndexLinear,IndexCartesian}

# Type of array axes and union of array size or axes.
const Axes{N} = NTuple{N,AbstractUnitRange{Int}}
const DimsOrAxes{N} = Union{Dims{N},Axes{N}}

abstract type AbstractStructuredArray{T,N,S<:ConcreteIndexStyle,I<:DimsOrAxes{N}} <: AbstractArray{T,N} end

abstract type AbstractUniformArray{T,N,I<:DimsOrAxes{N}} <: AbstractStructuredArray{T,N,IndexLinear,I} end

"""
    UniformArray(val, args...) -> A
    UniformArray{T}(val, args...) -> A
    UniformArray{T,N}(val, args...) -> A

build an immutable array `A` whose elements are all equal to `val`. The storage
requirement is `O(1)` instead of `O(length(A))` for a usual array `A`.
Subsequent arguments `args...` define the size or the axes of the array.
Optional parameters `T` and `N` are to specify the element type and the number
of dimensions of the array.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val` for
all linear indices `i` in the range `1:length(A)`.

A statement like `A[i] = val` is not implemented as uniform arrays are
considered as immutable. Call `MutableUniformArray(val,dims)` to create a
uniform array whose element value can be changed.

"""
struct UniformArray{T,N,I<:DimsOrAxes{N}} <: AbstractUniformArray{T,N,I}
    inds::I
    val::T
    UniformArray{T}(val, inds::I) where {T,N,I<:DimsOrAxes{N}} =
        new{T,N,I}(checked_size(inds), val)
end

"""
    FastUniformArray(val, args...) -> A
    FastUniformArray{T}(val, args...) -> A
    FastUniformArray{T,N}(val, args...) -> A

build an immutable uniform array `A` whose elements are all equal to `val`. The
difference with an instance of [`UniformArray`](@ref) is that `val` is part of
the type signature so that `val` can be known at compile time. A typical use is
to create all true/false masks.

"""
struct FastUniformArray{T,N,V,I<:DimsOrAxes{N}} <: AbstractUniformArray{T,N,I}
    inds::I
    FastUniformArray{T}(val, inds::I) where {T,N,I<:DimsOrAxes{N}} =
        new{T,N,as(T,val),I}(checked_size(inds))
end

"""
    MutableUniformArray(val, args...) -> A
    MutableUniformArray{T}(val, args...) -> A
    MutableUniformArray{T,N}(val, args...) -> A

build a mutable array `A` whose elements are initially all equal to `val`. The
difference with an instance of [`UniformArray`](@ref) is that the uniform value
can be changed.

A statement like `A[i] = val` is allowed but changes the value of all the
elements of `A`. Call `UniformArray(val,dims)` to create an immutable uniform
array whose element value cannot be changed.

"""
mutable struct MutableUniformArray{T,N,I<:DimsOrAxes{N}} <: AbstractUniformArray{T,N,I}
    inds::I
    val::T
    MutableUniformArray{T}(val, inds::I) where {T,N,I<:DimsOrAxes{N}} =
        new{T,N,I}(checked_size(inds), val)
end

"""
    StructuredArray([S = IndexCartesian,] func, args...) -> A
    StructuredArray{T}([S = IndexCartesian,] func, args...) -> A
    StructuredArray{T,N}([S = IndexCartesian,] func, args...) -> A
    StructuredArray{T,N,S}(func, args...) -> A

build an array `A` whose values are a given function, here `func`, of its
indices: `A[i]` is computed as `func(i)`. The storage requirement is `O(1)`
instead of `O(lenght(A))` for a usual array. Subsequent arguments `args...`
define the size or the axes of the array. Optional parameters `T`, `N`, and `S`
are to specify the element type, the number of dimensions, and the type of the
indexing style of the array. If specified as an argument not a type parameter,
`S` may also be an instance of `IndexStyle`.

If indexing style is `IndexCartesian` (the default), the function `func` will
be called with `N` integer arguments, `N` being the number of dimensions. If
`S` is `IndexLinear`, the function `func` will be called with a single integer
argument.

For instance, the structure of a lower triangular matrix of size `m×n` would be
given by:

    StructuredArray((i,j) -> (i ≥ j), m, n)

but with a constant small storage requirement whatever the size of the matrix.

Although the callable object `func` may not be a *pure function*, its return
type shall be stable and structured arrays are considered as immutable in the
sense that a statement like `A[i] = val` is not implemented. The type of the
elements of structured array is guessed by applying `func` to the unit index.
The element type, say `T`, may also be explicitely specified:

    StructuredArray{T}([S = IndexCartesian,] func, dims)

"""
struct StructuredArray{T,N,S,F,I<:DimsOrAxes{N}} <: AbstractStructuredArray{T,N,S,I}
    inds::I
    func::F
    StructuredArray{T,N,S}(func::F, inds::I) where {T,N,S<:ConcreteIndexStyle,F,I<:DimsOrAxes{N}} =
        new{T,N,S,F,I}(checked_size(inds), func)
end

# Aliases.
for (A, N) in ((:Vector, 1), (:Matrix, 2))
    @eval begin
        const $(Symbol("AbstractStructured",A)){T,S,I} = AbstractStructuredArray{T,$N,S,I}
        const $(Symbol("AbstractUniform",A)){T,I} = AbstractUniformArray{T,$N,I}
        const $(Symbol("MutableUniform",A)){T,I} = MutableUniformArray{T,$N,I}
        const $(Symbol("Uniform",A)){T,I} = UniformArray{T,$N,I}
        const $(Symbol("FastUniform",A)){T,V,I} = FastUniformArray{T,$N,V,I}
        const $(Symbol("Structured",A)){T,S,F,I} = StructuredArray{T,$N,S,F,I}
    end
end

# Specialize base abstract array methods for structured arrays.
for cls in (:StructuredArray, :FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        Base.length(A::$cls) = prod(size(A))
        Base.size(A::$cls) = to_size(getfield(A, :inds))
        Base.size(A::$cls, i::Integer) =
            i > ndims(A) ? 1 :
            i > zero(i) ? to_dim(getfield(A, :inds)[i]) : throw(BoundsError(size(A), i))
        Base.axes(A::$cls) = to_axes(getfield(A, :inds))
        Base.axes(A::$cls, i::Integer) =
            i > ndims(A) ? Base.OneTo(1) :
            i > zero(i) ? to_axis(getfield(A, :inds)[i]) : throw(BoundsError(axes(A), i))
     end
end
Base.has_offset_axes(A::AbstractUniformArray{T,0,Tuple{}}) where {T} = false
Base.has_offset_axes(A::AbstractUniformArray{T,N,Dims{N}}) where {T,N} = false
Base.has_offset_axes(A::StructuredArray{T,0,S,F,Tuple{}}) where {T,S,F} = false
Base.has_offset_axes(A::StructuredArray{T,N,S,F,Dims{N}}) where {T,N,S,F} = false

# Constructors that convert trailing argument(s) to array dimensions or axes.
for cls in (:StructuredArray, :FastUniformArray, :UniformArray, :MutableUniformArray)
    # 0-dimesional case.
    @eval begin
        $cls(     arg1)             = $cls(     arg1, ())
        $cls{T}(  arg1) where {T}   = $cls{T}(  arg1, ())
        $cls{T,N}(arg1) where {T,N} = $cls{T,N}(arg1, ())
    end
    if cls === :StructuredArray
        @eval begin
            $cls(::Union{S,Type{S}}, func) where {S<:ConcreteIndexStyle} =
                $cls(S, func, ())
            $cls{T}(::Union{S,Type{S}}, func) where {T,S<:ConcreteIndexStyle} =
                $cls{T}(S, func, ())
            $cls{T,0}(::Union{S,Type{S}}, func) where {T,S<:ConcreteIndexStyle} =
                $cls{T,0,S}(func, ())
            $cls{T,0,S}(::Union{S,Type{S}}, func) where {T,S<:ConcreteIndexStyle} =
                $cls{T,0,S}(func, ())
            $cls{T,0,S}(func) where {T,S<:ConcreteIndexStyle} =
                $cls{T,0,S}(func, ())
        end
    end
    # N-dimensional cases with N ≥ 1.
    for (type, to) in ((Integer,                      :to_size),
                       (AbstractUnitRange{<:Integer}, :to_axes))
        @eval begin
            $cls(     arg1, args::$type...)             = $cls(     arg1, args)
            $cls{T}(  arg1, args::$type...) where {T}   = $cls{T}(  arg1, args)
            $cls{T,N}(arg1, args::$type...) where {T,N} = $cls{T,N}(arg1, args)

            $cls(     arg1, args::NTuple{N,$type}) where {  N} = $cls(     arg1, $to(args))
            $cls{T}(  arg1, args::NTuple{N,$type}) where {T,N} = $cls{T}(  arg1, $to(args))
            $cls{T,N}(arg1, args::NTuple{N,$type}) where {T,N} = $cls{T,N}(arg1, $to(args))
        end
        if cls === :StructuredArray
            @eval begin
                $cls(::Union{S,Type{S}}, func, args::$type...) where {S<:ConcreteIndexStyle} =
                    $cls(S, func, args)
                $cls{T}(::Union{S,Type{S}}, func, args::$type...) where {T,S<:ConcreteIndexStyle} =
                    $cls{T}(S, func, args)
                $cls{T,N}(::Union{S,Type{S}}, func, args::$type...) where {T,N,S<:ConcreteIndexStyle} =
                    $cls{T,N,S}(func, args)
                $cls{T,N,S}(::Union{S,Type{S}}, func, args::$type...) where {T,N,S<:ConcreteIndexStyle} =
                    $cls{T,N,S}(func, args)
                $cls{T,N,S}(func, args::$type...) where {T,N,S<:ConcreteIndexStyle} =
                    $cls{T,N,S}(func, args)

                $cls(::Union{S,Type{S}}, func, args::NTuple{N,$type}) where {N,S<:ConcreteIndexStyle} =
                    $cls(S, func, $to(args))
                $cls{T}(::Union{S,Type{S}}, func, args::NTuple{N,$type}) where {T,N,S<:ConcreteIndexStyle} =
                    $cls{T}(S, func, $to(args))
                $cls{T,N}(::Union{S,Type{S}}, func, args::NTuple{N,$type}) where {T,N,S<:ConcreteIndexStyle} =
                    $cls{T,N,S}(func, $to(args))
                $cls{T,N,S}(::Union{S,Type{S}}, func, args::NTuple{N,$type}) where {T,N,S<:ConcreteIndexStyle} =
                    $cls{T,N,S}(func, $to(args))
                $cls{T,N,S}(func, args::NTuple{N,$type}) where {T,N,S<:ConcreteIndexStyle} =
                    $cls{T,N,S}(func, $to(args))
            end
        end
    end
end

# Default index style is Cartesian for StructuredArray.
let cls = :StructuredArray
    @eval begin
        $cls(func, args::DimsOrAxes{N}) where {N} = $cls(IndexCartesian, func, args)
        $cls{T}(func, args::DimsOrAxes{N}) where {T,N} = $cls{T,N}(IndexCartesian, func, args)
        $cls{T,N}(func, args::DimsOrAxes{N}) where {T,N} = $cls{T,N}(IndexCartesian, func, args)
    end
end

# Constructors that manage to call the inner constructors.
for cls in (:FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        $cls(val::T, args::DimsOrAxes{N}) where {T,N} = $cls{T}(val, args)
        $cls{T,N}(val, args::DimsOrAxes{N}) where {T,N} = $cls{T}(val, args)
    end
end
let cls = :StructuredArray
    @eval begin
        function $cls(::Union{S,Type{S}}, func, args::DimsOrAxes{N}) where {N,S<:ConcreteIndexStyle}
            T = guess_eltype(func, S, Val(N))
            return $cls{T,N,S}(func, args)
        end
        $cls{T}(::Union{S,Type{S}}, func, args::DimsOrAxes{N}) where {T,N,S<:ConcreteIndexStyle} =
            $cls{T,N,S}(func, args)
        $cls{T,N}(::Union{S,Type{S}}, func, args::DimsOrAxes{N}) where {T,N,S<:ConcreteIndexStyle} =
            $cls{T,N,S}(func, args)
    end
end

guess_eltype(func, ::Type{IndexLinear}, ::Val{N}) where {N} =
    Base.promote_op(func, Int)

@generated guess_eltype(func, ::Type{IndexCartesian}, ::Val{N}) where {N} =
    :(Base.promote_op(func, $(ntuple(x -> Int, Val(N))...)))

function Base.reverse(A::AbstractUniformArray; dims::Union{Colon,Integer,Tuple{Vararg{Integer}},
                                                           AbstractVector{<:Integer}} = :)
    if !(dims isa Colon)
        for d in dims
            d ≥ one(d) || throw(ArgumentError("dimension must be ≥ 1, got $d"))
        end
    end
    return A
end

Base.unique(A::AbstractUniformArray; dims::Union{Colon,Integer} = :) = _unique(A, dims)

_unique(A::AbstractUniformArray, ::Colon) = [value(A)]
function _unique(A::AbstractUniformArray{T,N,I}, d::Integer) where {T,N,I}
    if I <: Dims{N}
        inp_size = size(A)
        out_size = ntuple(i -> i == d ? 1 : @inbounds(inp_size[i]), Val(N))
        return UniformArray{T}(value(A), out_size)
    else
        inp_axes = axes(A)
        out_axes = ntuple(i -> i == d ? 1 : @inbounds(inp_axes[i]), Val(N))
        return UniformArray{T}(value(A), out_axes)
    end
end

# Optimize base reduction methods for uniform arrays.
for func in (:all, :any,
             :minimum, :maximum, :extrema,
             :count, :sum, :prod,
             :findmin, :findmax)
    _func = Symbol("_",func)
    @eval begin
        Base.$(func)(A::AbstractUniformArray; dims = :) = $(_func)(identity, A, dims)
        Base.$(func)(f::Function, A::AbstractUniformArray; dims = :) = $(_func)(f, A, dims)
    end
    if func ∈ (:all, :any)
        @eval begin
            function $(_func)(f, A::AbstractUniformArray, ::Colon)
                if isempty(A)
                    return _empty_result($(func))
                else
                    val = f(value(A))
                    return val isa Bool ? val : _unexpected_non_boolean(val)
                end
            end
            function $(_func)(f, A::AbstractUniformArray, dims)
                inds = _reduced_inds(getfield(A, :inds), dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    val = f(value(A))
                    val isa Bool || _unexpected_non_boolean(val)
                    return UniformArray(val, inds)
                end
            end
        end
    elseif func ∈ (:minimum, :maximum)
        @eval begin
            $(_func)(f, A::AbstractUniformArray, ::Colon) =
                isempty(A) ? _empty_result($(func)) : f(value(A))
            function $(_func)(f, A::AbstractUniformArray, dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    inds = _reduced_inds(getfield(A, :inds), dims)
                    val = f(value(A))
                    return UniformArray(val, inds)
                end
            end
        end
    elseif func === :extrema
        @eval begin
            function $(_func)(f, A::AbstractUniformArray, ::Colon)
                if isempty(A)
                    return _empty_result($(func))
                else
                    val = f(value(A))
                    return (val, val)
                end
            end
            function $(_func)(f, A::AbstractUniformArray, dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    inds = _reduced_inds(getfield(A, :inds), dims)
                    val = f(value(A))
                    return UniformArray((val,val), inds)
                end
            end
        end
    elseif func ∈ (:count, :sum, :prod)
        op = Symbol(_func,"_num_val")
        @eval begin
            function $(_func)(f, A::AbstractUniformArray, ::Colon)
                if isempty(A)
                    return _empty_result($(func))
                else
                    num = length(A)
                    val = f(value(A))
                    return $(op)(num, val)
                end
            end
            function $(_func)(f, A::AbstractUniformArray, dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    inds = _reduced_inds(getfield(A, :inds), dims)
                    num = div(length(A), prod(to_size(inds)))
                    val = f(value(A))
                    return UniformArray($(op)(num, val), inds)
                end
            end
        end
    elseif func ∈ (:findmin, :findmax)
        @eval begin
            function $(_func)(f, A::AbstractUniformArray, ::Colon)
                if isempty(A)
                    return _empty_result($(func))
                else
                    idx = CartesianIndex(map(first, axes(A)))
                    val = f(value(A))
                    return val, idx
                end
            end
            function $(_func)(f, A::AbstractUniformArray, dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    inds = _reduced_inds(getfield(A, :inds), dims)
                    val = f(value(A))
                    return UniformArray(val, inds), CartesianIndices(inds)
                end
            end
        end
    end
end

# Yield reduced axes/size of ().
_reduced_inds(I::DimsOrAxes, ::Colon) = ()
_reduced_inds(I::DimsOrAxes, dim::Integer) = _reduced_inds(I, (dim,))
function _reduced_inds(I::DimsOrAxes{N},
                       dims::Union{AbstractVector{<:Integer},
                                   Tuple{Vararg{Integer}}}) where {N}
    dmin = minimum(dims)
    dmin ≥ one(dmin) || throw(ArgumentError("region dimension(s) must be ≥ 1, got $dmin"))
    #all(d -> d ≥ one(d), dims) || throw(ArgumentError(
    #    "region dimension(s) must be ≥ 1, got $(minimum(dims))"))
    return ntuple(d -> d ∈ dims ? _reduced_axis(I[d]) : I[d], Val(N))
end
_reduced_inds(I::DimsOrAxes, dims) = throw(ArgumentError(
    "region dimension(s) must be a colon, an integer, or a vector/tuple of integers"))

_reduced_axis(::Int) = 1
_reduced_axis(::AbstractUnitRange{Int}) = Base.OneTo(1)

# Yield a uniform array with given value and axes/size or a scalar.
_uniform(val, ::Tuple{}) = val
_uniform(val, inds::Tuple) = UniformArray(val, inds)

_sum_num_val(num::Integer, val) = num*val
_prod_num_val(num::Integer, val) = val^num
_count_num_val(num::Integer, val::Bool) = val ? num : zero(num)
_count_num_val(num::Integer, val) = _unexpected_non_boolean(val)
@noinline _unexpected_non_boolean(::Union{T,Type{T}}) where {T} =
    throw(ArgumentError("unexpected non-boolean ($T) result"))

# Result for an empty array for some base reduction functions.
_empty_result(::typeof(all)) = true
_empty_result(::typeof(any)) = false
@noinline _empty_result(f::Function) =
    throw(ArgumentError("reducing by `$(typename(f))` over an empty region is not allowed"))

# See comments near `getindex` in `abstractarray.jl` for explanations about how
# `getindex` and `setindex!` methods are expected to be specialized depending
# on the indexing style: `i:Int` for linear indexing style or
# `i::Vararg{Int,N}` for `N` dimensional Cartesian indexing style.

Base.IndexStyle(::Type{<:AbstractStructuredArray{T,N,S}}) where {T,N,S} = S()

@inline function Base.getindex(A::StructuredArray{T,N,IndexLinear},
                               i::Int) where {T,N}
    @boundscheck checkbounds(A, i)
    return @inbounds as(T, A.func(i))
end

@inline function Base.getindex(A::StructuredArray{T,N,IndexCartesian},
                               I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, I...)
    return @inbounds as(T, A.func(I...))
end

"""
    StructuredArrays.value(A::AbstractUniformArray)

yields the value of the elements of the uniform array `A`.

"""
value(A::FastUniformArray{T,N,V}) where {T,N,V} = V
value(A::UniformArray) = getfield(A, :val)
value(A::MutableUniformArray) = getfield(A, :val)

setvalue!(A::MutableUniformArray{T}, x) where {T} = setfield!(A, :val, as(T, x))

@inline function Base.getindex(A::AbstractUniformArray, i::Int)
    @boundscheck checkbounds(A, i)
    return value(A)
end

# Fast view for immutable uniform arrays, may return a 0-dimensional array.
@inline @propagate_inbounds function Base.view(A::Union{UniformArray,FastUniformArray},
                                               I::Vararg{Any})
    return subarray(A, I...)
end

@inline function Base.setindex!(A::MutableUniformArray, x, i::Int)
    i:i == eachindex(IndexLinear(), A) || not_all_elements()
    setvalue!(A, x)
    return A
end

@inline function Base.setindex!(A::MutableUniformArray, x, rng::AbstractUnitRange{<:Integer})
    rng == eachindex(IndexLinear(), A) || not_all_elements()
    setvalue!(A, x)
    return A
end

@inline function Base.setindex!(A::MutableUniformArray, x, ::Colon)
    setvalue!(A, x)
    return A
end

@noinline not_all_elements() =
    error("all elements must be set at the same time")

@inline function subarray(A::AbstractUniformArray, I::Vararg{Any})
    # NOTE: Since we are calling the `subarray` method to produce an object
    # that is independent from `A`, it is not needed to unalias the result of
    # `to_indices` from `A` as is done in Julia base code (in `subarray.jl`) to
    # build views.
    J = to_indices(A, I)
    @boundscheck checkbounds(A, J...)
    # NOTE: `to_indices` converts colons (`:`) into slices (`Base.Slice`) which
    # are abstract unit ranges and thus abstract vectors of indices. So the
    # only thing to do is to concatenate the sizes of the indices returned by
    # `to_indices`.
    dims = concat(size, J)
    return parameterless(typeof(A))(value(A), concat(size, J))
end

@inline concat(f::Function, x::Tuple) = concat((), f, x) # start recursion
@inline concat(r::Tuple, f::Function, x::Tuple{}) = r    # finish recursion
@inline concat(r::Tuple, f::Function, x::Tuple) =        # pursue recursion
    concat((r..., f(first(x))...), f, tail(x))

"""
    StructuredArrays.checked_size(inds) -> inds

throws an `ArgumentError` exception if `inds` is not a valid array size
or array axes amd returns `inds` otherwise.

"""
function checked_size(dims::Dims{N}) where {N}
    flag = true
    @inbounds for i in 1:N
        flag &= dims[i] ≥ 0
    end
    flag || throw(ArgumentError("invalid array dimension(s)"))
    return dims
end
checked_size(inds::Axes) = inds
checked_size(::Tuple{}) = ()

"""
    StructuredArrays.to_dim(x) -> dim::Int

yields the length of array dimension specified by `x` which may be an integer
or a unit range.

"""
to_dim(dim::Int) = dim
to_dim(dim::Integer) = Int(dim)
to_dim(rng::AbstractUnitRange{<:Integer}) = as(Int, length(rng))
to_dim(rng::OrdinalRange{<:Integer}) = isone(step(rng)) ? length(rng) : non_unit_step(rng)

"""
    StructuredArrays.to_size(x) -> dims::Dims

yields the array dimension(s) specified by `x`.

"""
to_size(::Tuple{}) = ()
to_size(dims::Dims) = dims
to_size(dims::Tuple{Vararg{Integer}}) = map(to_dim, dims)
to_size(dim::Integer) = (to_dim(dim),)
to_size(rng::OrdinalRange{<:Integer}) = (to_dim(rng),)
to_size(inds::Tuple{Vararg{OrdinalRange{<:Integer}}}) = map(to_dim, inds)

"""
    StructuredArrays.to_axis(x) -> rng::AbstractUnitRange{Int}

yields the array axis specified by `x`.

"""
to_axis(dim::Integer) = Base.OneTo{Int}(dim)
to_axis(rng::AbstractUnitRange{Int}) = rng
to_axis(rng::AbstractUnitRange{<:Integer}) = convert_eltype(Int, rng)
to_axis(rng::OrdinalRange{<:Integer}) =
    isone(step(rng)) ? UnitRange{Int}(first(rng), last(rng)) : non_unit_step(rng)

"""
    StructuredArrays.to_axes(x) -> rngs::Tuple{Vararg{AbstractUnitRange{Int}}}

yields the array axes specified by `x`.

"""
to_axes(::Tuple{}) = ()
to_axes(inds::Tuple{Vararg{AbstractUnitRange{Int}}}) = inds
to_axes(inds::Tuple{Vararg{OrdinalRange{<:Integer}}}) = map(to_axis, inds)
to_axes(dims::Tuple{Vararg{Integer}}) = map(to_axis, dims)

@noinline non_unit_step(rng::AbstractRange) = throw(ArgumentError(
    "invalid non-unit step ($(step(rng))) range"))

end # module
