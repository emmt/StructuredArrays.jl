#
# uniformarrays.jl --
#
# Implement arrays whose elements all have the same value.
#
module UniformArrays

export
    AbstractUniformArray,
    AbstractUniformMatrix,
    AbstractUniformVector,
    MutableUniformArray,
    MutableUniformMatrix,
    MutableUniformVector,
    UniformArray,
    UniformMatrix,
    UniformVector

import Base: @propagate_inbounds

"""
    UniformArray(val, siz) -> A

yields an array `A` which behaves as an immutable array of size `siz` whose
values are all `val`.  The storage requirement is `O(1)` instead of
`O(prod(siz))` for a usual array.  The array dimensions may be specified as
multiple arguments.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val`
for all linear indices `i` in the range `1:length(A)`.

A statement like `A[i] = val` is not implemented as uniform arrays are
considered as immutable.  Call `MutableUniformArray(val,siz)` to create a
uniform array whose element value can be changed.

""" UniformArray

"""
    MutableUniformArray(val, siz) -> A

yields an array `A` which behaves as a mutable array of size `siz` whose
values are all `val`.  The storage requirement is `O(1)` instead of
`O(prod(siz))` for a usual array.  The array dimensions may be specified as
multiple arguments.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val`
for all linear indices `i` in the range `1:length(A)`.

A statement like `A[i] = val` is allowed but changes the value of all the
elements of `A`.  Call `UniformArray(val,siz)` to create an immutable uniform
array whose element value cannot be changed.

""" MutableUniformArray

abstract type AbstractUniformArray{T,N} <: AbstractArray{T,N} end

struct UniformArray{T,N} <: AbstractUniformArray{T,N}
    len::Int
    siz::NTuple{N,Int}
    val::T
    function UniformArray{T,N}(val, siz::NTuple{N,Int}) where {T,N}
        len = 1
        @inbounds for i in 1:N
            (dim = siz[i]) ≥ 0 || bad_dimension_length()
            len *= dim
        end
        return new{T,N}(len, siz, val)
    end
end

mutable struct MutableUniformArray{T,N} <: AbstractUniformArray{T,N}
    len::Int
    siz::NTuple{N,Int}
    val::T
    function MutableUniformArray{T,N}(val, siz::NTuple{N,Int}) where {T,N}
        len = 1
        @inbounds for i in 1:N
            (dim = siz[i]) ≥ 0 || bad_dimension_length()
            len *= dim
        end
        return new{T,N}(len, siz, val)
    end
end

const AbstractUniformVector{T} = AbstractUniformArray{T,1}
const MutableUniformVector{T} = MutableUniformArray{T,1}
const UniformVector{T} = UniformArray{T,1}

const AbstractUniformMatrix{T} = AbstractUniformArray{T,2}
const MutableUniformMatrix{T} = MutableUniformArray{T,2}
const UniformMatrix{T} = UniformArray{T,2}

# Constructors for UniformArray.
UniformArray(val::T, siz::Integer...) where {T} =
    UniformArray{T}(val, siz)
UniformArray(val::T, siz::NTuple{N,Integer}) where {T,N} =
    UniformArray{T,N}(val, siz)
UniformArray{T}(val, siz::Integer...) where {T} =
    UniformArray{T}(val, siz)
UniformArray{T}(val, siz::NTuple{N,Integer}) where {T,N} =
    UniformArray{T,N}(val, siz)
UniformArray{T,N}(val, siz::Integer...) where {T,N} =
    UniformArray{T,N}(val, siz)
UniformArray{T,N}(val, siz::NTuple{N,Integer}) where {T,N} =
    UniformArray{T,N}(val, map(Int, siz))

# Constructors for MutableUniformArray.
MutableUniformArray(val::T, siz::Integer...) where {T} =
    MutableUniformArray{T}(val, siz)
MutableUniformArray(val::T, siz::NTuple{N,Integer}) where {T,N} =
    MutableUniformArray{T,N}(val, siz)
MutableUniformArray{T}(val, siz::Integer...) where {T} =
    MutableUniformArray{T}(val, siz)
MutableUniformArray{T}(val, siz::NTuple{N,Integer}) where {T,N} =
    MutableUniformArray{T,N}(val, siz)
MutableUniformArray{T,N}(val, siz::Integer...) where {T,N} =
    MutableUniformArray{T,N}(val, siz)
MutableUniformArray{T,N}(val, siz::NTuple{N,Integer}) where {T,N} =
    MutableUniformArray{T,N}(val, map(Int, siz))

Base.eltype(::AbstractUniformArray{T}) where {T} = T

Base.ndims(::AbstractUniformArray{T,N}) where {T,N} = N

Base.length(A::AbstractUniformArray) = A.len

Base.size(A::AbstractUniformArray) = A.siz
Base.size(A::AbstractUniformArray{T,N}, i::Integer) where {T,N} =
    (i < 1 ? bad_dimension_index() : i ≤ N ? A.siz[i] : 1)

Base.axes(A::AbstractUniformArray) = map(Base.OneTo, size(A))
Base.axes(A::AbstractUniformArray, i::Integer) = Base.OneTo(size(A, i))

Base.has_offset_axes(::AbstractUniformArray) = false

Base.IndexStyle(::Type{<:AbstractUniformArray}) = IndexLinear()

@inline function Base.getindex(A::AbstractUniformArray, i::Int)
    @boundscheck checkbounds(A, i)
    A.val
end

@inline function Base.setindex!(A::MutableUniformArray, x, i::Int)
    @boundscheck checkbounds(A, i)
    A.val = x
end

@noinline bad_dimension_index() = error("out of range dimension index")

@noinline bad_dimension_length() = error("invalid dimension length")

end # module
