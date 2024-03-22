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

abstract type AbstractStructuredArray{T,N,S<:ConcreteIndexStyle} <:
    AbstractArray{T,N} end

abstract type AbstractUniformArray{T,N} <:
    AbstractStructuredArray{T,N,IndexLinear} end

"""
    UniformArray(val, dims) -> A

yields an array `A` which behaves as an immutable array of size `dims` whose
elements are all equal to `val`. The storage requirement is `O(1)` instead of
`O(length(A))` for a usual array `A`. The array dimensions may be specified as
multiple arguments.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val` for
all linear indices `i` in the range `1:length(A)`.

A statement like `A[i] = val` is not implemented as uniform arrays are
considered as immutable. Call `MutableUniformArray(val,dims)` to create a
uniform array whose element value can be changed.

"""
struct UniformArray{T,N} <: AbstractUniformArray{T,N}
    dims::Dims{N}
    val::T
    UniformArray{T}(val, dims::Dims{N}) where {T,N} =
        new{T,N}(checked_size(dims), val)
end

"""
    FastUniformArray(val, dims) -> A

yields an immutable uniform array of size `dims` and whose elements are all
equal to `val`. The difference with an instance of [`UniformArray`](@ref) is
that `val` is part of the type signature so that `val` can be known at compile
time. A typical use is to create all true/false masks.

"""
struct FastUniformArray{T,N,V} <: AbstractUniformArray{T,N}
    dims::Dims{N}
    FastUniformArray{T}(val, dims::Dims{N}) where {T,N} =
        new{T,N,as(T,val)}(checked_size(dims))
end

"""
    MutableUniformArray(val, dims) -> A

yields an array `A` which behaves as a mutable array of size `dims` whose
values are all `val`. The storage requirement is `O(1)` instead of
`O(prod(dims))` for a usual array. The array dimensions may be specified as
multiple arguments.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val` for
all linear indices `i` in the range `1:length(A)`.

A statement like `A[i] = val` is allowed but changes the value of all the
elements of `A`. Call `UniformArray(val,dims)` to create an immutable uniform
array whose element value cannot be changed.

"""
mutable struct MutableUniformArray{T,N} <: AbstractUniformArray{T,N}
    dims::Dims{N}
    val::T
    MutableUniformArray{T}(val, dims::Dims{N}) where {T,N} =
        new{T,N}(checked_size(dims), val)
end

"""
    StructuredArray([S = IndexCartesian,] func, dims) -> A

yields an array `A` which behaves as an array of size `dims` whose values are a
given function, here `func`, of its indices: `A[i]` is computed as `func(i)`.
The storage requirement is `O(1)` instead of `O(prod(dims))` for a usual array.
The array dimensions may be specified as multiple arguments.

The optional argument `S` may be used to specifiy another index style than the
default `IndexCartesian`, for instance `IndexLinear`. If specified, `S` may be
a sub-type of `IndexStyle` or an instance of such a sub-type. If `S` is
`IndexCartesian` (the default), the function `func` will be called with `N`
integer arguments, `N` being the number of dimensions. If `S` is
`IndexCartesian`, the function `func` will be called with a single integer
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
struct StructuredArray{T,N,S,F} <: AbstractStructuredArray{T,N,S}
    dims::Dims{N}
    func::F
    StructuredArray{T,N,S}(func::F, dims::Dims{N}) where {T,N,S<:ConcreteIndexStyle,F} =
        new{T,N,S,F}(checked_size(dims), func)
end

# Aliases.
for (A, N) in ((:Vector, 1), (:Matrix, 2))
    @eval begin
        const $(Symbol("AbstractStructured",A)){T,S} = AbstractStructuredArray{T,$N,S}
        const $(Symbol("AbstractUniform",A)){T} = AbstractUniformArray{T,$N}
        const $(Symbol("MutableUniform",A)){T} = MutableUniformArray{T,$N}
        const $(Symbol("Uniform",A)){T} = UniformArray{T,$N}
        const $(Symbol("FastUniform",A)){T,V} = FastUniformArray{T,$N,V}
        const $(Symbol("Structured",A)){T,S,F} = StructuredArray{T,$N,S,F}
    end
end

# Specialize base abstract array methods for structured arrays.
for cls in (:StructuredArray, :FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        Base.length(A::$cls) = prod(size(A))
        Base.size(A::$cls) = getfield(A, :dims)
    end
end
Base.axes(A::AbstractStructuredArray) = to_axes(size(A))

# Constructors that convert trailing argument(s) to array dimensions.
for cls in (:StructuredArray, :FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        $cls(     arg1, args::Integer...)             = $cls(     arg1, args)
        $cls{T}(  arg1, args::Integer...) where {T}   = $cls{T}(  arg1, args)
        $cls{T,N}(arg1, args::Integer...) where {T,N} = $cls{T,N}(arg1, args)

        $cls(     arg1, args::NTuple{N,Integer}) where {  N} = $cls(   arg1, to_size(args))
        $cls{T}(  arg1, args::NTuple{N,Integer}) where {T,N} = $cls{T}(arg1, to_size(args))
        $cls{T,N}(arg1, args::NTuple{N,Integer}) where {T,N} = $cls{T}(arg1, to_size(args))
    end
    if cls === :StructuredArray
        @eval begin
            $cls(::Union{S,Type{S}}, func, args::Integer...) where {S<:ConcreteIndexStyle} =
                $cls(S, func, args)
            $cls{T}(::Union{S,Type{S}}, func, args::Integer...) where {T,S<:ConcreteIndexStyle} =
                $cls{T}(S, func, args)
            $cls{T,N}(::Union{S,Type{S}}, func, args::Integer...) where {T,N,S<:ConcreteIndexStyle} =
                $cls{T,N,S}(func, args)
            $cls{T,N,S}(::Union{S,Type{S}}, func, args::Integer...) where {T,N,S<:ConcreteIndexStyle} =
                $cls{T,N,S}(func, args)
            $cls{T,N,S}(func, args::Integer...) where {T,N,S<:ConcreteIndexStyle} =
                $cls{T,N,S}(func, args)

            $cls(::Union{S,Type{S}}, func, args::NTuple{N,Integer}) where {N,S<:ConcreteIndexStyle} =
                $cls(S, func, to_size(args))
            $cls{T}(::Union{S,Type{S}}, func, args::NTuple{N,Integer}) where {T,N,S<:ConcreteIndexStyle} =
                $cls{T}(S, func, to_size(args))
            $cls{T,N}(::Union{S,Type{S}}, func, args::NTuple{N,Integer}) where {T,N,S<:ConcreteIndexStyle} =
                $cls{T,N,S}(func, to_size(args))
            $cls{T,N,S}(::Union{S,Type{S}}, func, args::NTuple{N,Integer}) where {T,N,S<:ConcreteIndexStyle} =
                $cls{T,N,S}(func, to_size(args))
            $cls{T,N,S}(func, args::NTuple{N,Integer}) where {T,N,S<:ConcreteIndexStyle} =
                $cls{T,N,S}(func, to_size(args))
        end
    end
end

# Default index style is Cartesian for StructuredArray.
let cls = :StructuredArray
    @eval begin
        $cls(func, args::Dims{N}) where {N} = $cls(IndexCartesian, func, args)
        $cls{T}(func, args::Dims{N}) where {T,N} = $cls{T,N}(IndexCartesian, func, args)
        $cls{T,N}(func, args::Dims{N}) where {T,N} = $cls{T,N,S}(IndexCartesian, func, args)
    end
end

# Constructors that manage to call the inner constructors.
for cls in (:FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        $cls(val::T, args::Dims{N}) where {T,N} = $cls{T}(val, args)
        $cls{T,N}(val, args::Dims{N}) where {T,N} = $cls{T}(val, args)
    end
end
let cls = :StructuredArray
    @eval begin
       function $cls(::Union{S,Type{S}}, func, args::Dims{N}) where {N,S<:ConcreteIndexStyle}
            T = guess_eltype(func, S, Val(N))
            return $cls{T,N,S}(func, args)
        end
        $cls{T}(::Union{S,Type{S}}, func, args::Dims{N}) where {T,N,S<:ConcreteIndexStyle} =
            $cls{T,N,S}(func, args)
        $cls{T,N}(::Union{S,Type{S}}, func, args::Dims{N}) where {T,N,S<:ConcreteIndexStyle} =
            $cls{T,N,S}(func, args)
    end
end

guess_eltype(func, ::Type{IndexLinear}, ::Val{N}) where {N} =
    Base.promote_op(func, Int)

@generated guess_eltype(func, ::Type{IndexCartesian}, ::Val{N}) where {N} =
    :(Base.promote_op(func, $(ntuple(x -> Int, Val(N))...)))

# Specialize some methods for (fast) uniform arrays of booleans.
Base.all(A::AbstractUniformArray{Bool}) = value(A)
Base.count(A::AbstractUniformArray{Bool}) = ifelse(value(A), length(A), 0)
Base.count(A::FastUniformArray{Bool,N,true}) where {N} = length(A)
Base.count(A::FastUniformArray{Bool,N,false}) where {N} = 0

# See comments near `getindex` in `abstractarray.jl` for explanations about how
# `getindex` and `setindex!` methods are expected to be specialized depending
# on the indexing style: `i:Int` for linear indexing style or
# `i::Vararg{Int,N}` for `N` dimensional Cartesian indexing style.

Base.IndexStyle(::Type{<:AbstractStructuredArray{T,N,S}}) where {T,N,S} = S()

@inline function Base.getindex(A::StructuredArray{T,N,IndexLinear},
                               i::Int) where {T,N}
    @boundscheck checkbounds(A, i)
    return convert(T, A.func(i))::T
end

@inline function Base.getindex(A::StructuredArray{T,N,IndexCartesian},
                               inds::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A, inds...)
    return convert(T, A.func(inds...))::T
end

"""
    StructuredArrays.value(A::AbstractUniformArray)

yields the value of the elements of the uniform array `A`.

"""
value(A::FastUniformArray{T,N,V}) where {T,N,V} = V
value(A::UniformArray) = getfield(A, :val)
value(A::MutableUniformArray) = getfield(A, :val)

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
    @boundscheck checkbounds(A, i)
    (i == length(A) == 1) || not_all_elements()
    A.val = x
    return A
end

@inline function Base.setindex!(A::MutableUniformArray, x,
                                i::AbstractUnitRange{<:Integer})
    ((first(i) == 1) & (last(i) == length(A))) || not_all_elements()
    A.val = x
    return A
end

@inline function Base.setindex!(A::MutableUniformArray, x, ::Colon)
    A.val = x
    A
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
    StructuredArrays.checked_size(dims) -> dims

throws an `ArgumentError` exception if `dims` is not a valid array list of
dimensions amd returns `dims` otherwise.

"""
function checked_size(dims::Dims{N}) where {N}
    flag = true
    @inbounds for i in 1:N
        flag &= dims[i] ≥ 0
    end
    flag || throw(ArgumentError("invalid array dimension(s)"))
    return dims
end

to_dim(dim::Int) = dim
to_dim(dim::Integer) = Int(dim)

to_size(::Tuple{}) = ()
to_size(dims::Dims) = dims
to_size(dims::Tuple{Vararg{Integer}}) = map(to_dim, dims)
to_size(dim::Integer) = (to_dim(dim),)
to_size(inds::Tuple{Vararg{AbstractUnitRange{<:Integer}}}) = map(length, inds)

to_axis(rng::AbstractUnitRange{Int}) = rng
to_axis(rng::AbstractUnitRange{<:Integer}) = convert_eltype(Int, rng)
to_axis(dim::Integer) = Base.OneTo{Int}(dim)

to_axes(::Tuple{}) = ()
to_axes(inds::Tuple{Vararg{AbstractUnitRange{Int}}}) = inds
to_axes(inds::Tuple{Vararg{AbstractUnitRange{<:Integer}}}) = map(to_axis, inds)
to_axes(dims::Tuple{Vararg{Integer}}) = map(to_axis, dims)

end # module
