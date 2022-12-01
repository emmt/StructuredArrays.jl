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

using ArrayTools
import Base: @propagate_inbounds

abstract type AbstractStructuredArray{T,N,S<:IndexStyle} <:
    AbstractArray{T,N} end

abstract type AbstractUniformArray{T,N} <:
    AbstractStructuredArray{T,N,IndexLinear} end

"""
    UniformArray(val, dims) -> A

yields an array `A` which behaves as an immutable array of size `dims` whose
elements are all equal to `val`. The storage requirement is `O(1)` instead of
`O(prod(dims))` for a usual array. The array dimensions may be specified as
multiple arguments.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val` for
all linear indices `i` in the range `1:length(A)`.

A statement like `A[i] = val` is not implemented as uniform arrays are
considered as immutable. Call `MutableUniformArray(val,dims)` to create a
uniform array whose element value can be changed.

"""
struct UniformArray{T,N} <: AbstractUniformArray{T,N}
    len::Int
    dims::Dims{N}
    val::T
    UniformArray{T,N}(val, dims::Dims{N}) where {T,N} =
        new{T,N}(checksize(dims), dims, val)
end

"""
    FastUniformArray(val, dims) -> A

yields an immutable uniform array of size `dims` and whose elements are all
equal to `val`. The difference with an instance of [`UniformArray`](@ref) is
that `val` is part of the type signature so that `val` can be known at compile
time. A typical use is to create all true/false masks.

"""
struct FastUniformArray{T,N,V} <: AbstractUniformArray{T,N}
    len::Int
    dims::Dims{N}
    FastUniformArray{T,N}(val::T, dims::Dims{N}) where {T,N} =
        new{T,N,val}(checksize(dims), dims)
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
    len::Int
    dims::Dims{N}
    val::T
    MutableUniformArray{T,N}(val, dims::Dims{N}) where {T,N} =
        new{T,N}(checksize(dims), dims, val)
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
    len::Int
    dims::Dims{N}
    func::F
    StructuredArray{T,N,S,F}(func, dims::Dims{N}) where {T,N,S,F} =
        new{T,N,S,F}(checksize(dims), dims, func)
end

const AbstractStructuredVector{T,S} = AbstractStructuredArray{T,1,S}
const StructuredVector{T,S,F} = StructuredArray{T,1,S,F}
const AbstractUniformVector{T} = AbstractUniformArray{T,1}
const MutableUniformVector{T} = MutableUniformArray{T,1}
const UniformVector{T} = UniformArray{T,1}
const FastUniformVector{T,V} = FastUniformArray{T,1,V}

const AbstractStructuredMatrix{T,S} = AbstractStructuredArray{T,2,S}
const StructuredMatrix{T,S,F} = StructuredArray{T,2,S,F}
const AbstractUniformMatrix{T} = AbstractUniformArray{T,2}
const MutableUniformMatrix{T} = MutableUniformArray{T,2}
const UniformMatrix{T} = UniformArray{T,2}
const FastUniformMatrix{T,V} = FastUniformArray{T,2,V}

# Specialize base abstract array methods for StructuredArray, UniformArray, and
# MutableUniformArray and provide basic constructors to convert trailing
# arguments to dimensions.
for cls in (:StructuredArray, :FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        Base.length(A::$cls) = getfield(A, :len)
        Base.size(A::$cls) = getfield(A, :dims)

        $cls(arg1, dims::Integer...) = $cls(arg1, dims)
        $cls{T}(arg1, dims::Integer...) where {T} = $cls{T}(arg1, dims)
        $cls{T,N}(arg1, dims::Integer...) where {T,N} = $cls{T,N}(arg1, dims)
    end
end
for cls in (:FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        $cls(val::T, dims::NTuple{N,Integer}) where {T,N} =
            $cls{T,N}(val, to_size(dims))
        $cls{T}(val, dims::NTuple{N,Integer}) where {T,N} =
            $cls{T,N}(val, to_size(dims))
        $cls{T,N}(val, dims::NTuple{N,Integer}) where {T,N} =
            $cls{T,N}(val, to_size(dims))
    end
end

# Make sure value has correct type.
FastUniformArray{T,N}(val, dims::Dims{N}) where {T,N} =
    FastUniformArray{T,N}(convert(T, val)::T, dims)

# Specialize some methods for (fast) uniform arrays of booleans.
Base.all(A::AbstractUniformArray{Bool}) = first(A)
Base.all(A::FastUniformArray{Bool,N,V}) where {N,V} = V
Base.count(A::AbstractUniformArray{Bool}) = ifelse(first(A), length(A), 0)
Base.count(A::FastUniformArray{Bool,N,true}) where {N} = length(A)
Base.count(A::FastUniformArray{Bool,N,false}) where {N} = 0

StructuredArray(func, dims::NTuple{N,Integer}) where {N} =
    StructuredArray(IndexCartesian, func, to_size(dims))
StructuredArray{T}(func, dims::NTuple{N,Integer}) where {T,N} =
    StructuredArray{T}(IndexCartesian, func, to_size(dims))
StructuredArray{T,N}(func, dims::NTuple{N,Integer}) where {T,N} =
    StructuredArray{T}(func, dims)

# All constructors for StructuredArray are based on the 2 first ones (only
# depending on whether parameter T is provided or not) so the other constructors
# just make sure and arguments have correct type.

function StructuredArray(::Type{S},
                         func::F,
                         dims::Dims{N}) where {N,S<:IndexStyle,F}
    T = guess_eltype(func, S, Val(N))
    StructuredArray{T,N,S,F}(func, dims)
end

function StructuredArray{T}(::Type{S},
                            func::F,
                            dims::Dims{N}) where {T,N,S<:IndexStyle,F}
    StructuredArray{T,N,S,F}(func, dims)
end

guess_eltype(func, ::Type{IndexLinear}, ::Val{N}) where {N} =
    Base.promote_op(func, Int)

@generated guess_eltype(func, ::Type{IndexCartesian}, ::Val{N}) where {N} =
    :(Base.promote_op(func, $(ntuple(x -> Int, Val(N))...)))

# Index style specified and size specified as a tuple or by trailing arguments.

function StructuredArray(::Union{S,Type{S}}, func,
                         dims::Integer...) where {S<:IndexStyle}
    StructuredArray(S, func, dims)
end
function StructuredArray(::Union{S,Type{S}}, func,
                         dims::NTuple{N,Integer}) where {N,S<:IndexStyle}
    StructuredArray(S, func, to_size(dims))
end

function StructuredArray{T}(::Union{S,Type{S}}, func,
                            dims::Integer...) where {T,S<:IndexStyle}
    StructuredArray{T}(S, func, dims)
end
function StructuredArray{T}(::Union{S,Type{S}}, func,
                            dims::NTuple{N,Integer}) where {T,N,S<:IndexStyle}
    StructuredArray{T}(S, func, to_size(dims))
end

function StructuredArray{T,N}(::Union{S,Type{S}}, func,
                              dims::Integer...) where {T,N,S<:IndexStyle}
    StructuredArray{T,N}(S, func, dims) # keep the N to check
end
function StructuredArray{T,N}(::Union{S,Type{S}}, func,
                              dims::NTuple{N,Integer}) where {T,N,S<:IndexStyle}
    StructuredArray{T}(S, func, to_size(dims))
end

function StructuredArray{T,N,S}(::Union{S,Type{S}}, func,
                                dims::Integer...) where {T,N,S<:IndexStyle}
    StructuredArray{T,N}(S, func, dims) # keep the N to check
end
function StructuredArray{T,N,S}(::Union{S,Type{S}}, func,
                                dims::NTuple{N,Integer}) where {T,N,S<:IndexStyle}
    StructuredArray{T}(S, func, to_size(dims))
end

function StructuredArray{T,N,S}(func,
                                dims::Integer...) where {T,N,S<:IndexStyle}
    StructuredArray{T,N}(S, func, dims) # keep the N to check
end
function StructuredArray{T,N,S}(func,
                                dims::NTuple{N,Integer}) where {T,N,S<:IndexStyle}
    StructuredArray{T}(S, func, to_size(dims))
end

# See comments near `getindex` in `abstractarray.jl` for explanations about how
# `getindex` and `setindex!` methods are expected to be specialized depending
# on the indexing style.

Base.IndexStyle(::Type{<:AbstractStructuredArray{T,N,S}}) where {T,N,S} = S()

@inline function Base.getindex(A::StructuredArray{T,N,IndexLinear},
                               i::Int) :: T where {T,N}
    @boundscheck checkbounds(A, i)
    return A.func(i)
end

@inline function Base.getindex(A::StructuredArray{T,N,IndexCartesian},
                               inds::Vararg{Int, N}) :: T where {T,N}
    @boundscheck checkbounds(A, inds...)
    return A.func(inds...)
end

getval(A::FastUniformArray{T,N,V}) where {T,N,V} = V
getval(A::AbstractUniformArray) = getfield(A, :val)

@inline function Base.getindex(A::AbstractUniformArray, I...)
    @boundscheck checkbounds(A, I...)
    return getval(A)
end

@inline function Base.setindex!(A::MutableUniformArray, x, i::Int)
    @boundscheck checkbounds(A, i)
    (i == length(A) == 1) || not_all_elements()
    A.val = x
    return A
end

@inline function Base.setindex!(A::MutableUniformArray, x,
                                i::AbstractUnitRange{<:Integer})
    (first(i) == 1 && last(i) == length(A)) || not_all_elements()
    A.val = x
    return A
end

@inline function Base.setindex!(A::MutableUniformArray, x, ::Colon)
    A.val = x
    A
end

"""
    StructuredArrays.checksize(dims) -> len

yields the number of elements of an array of size `dims` throwing an error if
any dimension is invalid.

"""
function checksize(dims::NTuple{N,Integer}) where {N}
    len = 1
    flag = true
    @inbounds for i in 1:N
        dim = Int(dims[i])
        flag &= dim ≥ 0
        len *= dim
    end
    flag || bad_dimension_length()
    return len
end

@noinline not_all_elements() =
    error("all elements must be set at the same time")

@noinline bad_dimension_length() = error("invalid dimension length")

end # module
