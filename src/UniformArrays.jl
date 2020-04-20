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
UniformArray(val, siz::Integer...) =
    UniformArray(val, siz)
UniformArray{T}(val, siz::Integer...) where {T} =
    UniformArray{T}(val, siz)
UniformArray{T,N}(val, siz::Integer...) where {T,N} =
    UniformArray{T,N}(val, siz)

UniformArray(val::T, siz::NTuple{N,Integer}) where {T,N} =
    UniformArray{T,N}(val, to_size(siz))
UniformArray{T}(val, siz::NTuple{N,Integer}) where {T,N} =
    UniformArray{T,N}(val, to_size(siz))
UniformArray{T,N}(val, siz::NTuple{N,Integer}) where {T,N} =
    UniformArray{T,N}(val, to_size(siz))

# Constructors for MutableUniformArray.
MutableUniformArray(val, siz::Integer...) =
    MutableUniformArray(val, siz)
MutableUniformArray{T}(val, siz::Integer...) where {T} =
    MutableUniformArray{T}(val, siz)
MutableUniformArray{T,N}(val, siz::Integer...) where {T,N} =
    MutableUniformArray{T,N}(val, siz)

MutableUniformArray(val::T, siz::NTuple{N,Integer}) where {T,N} =
    MutableUniformArray{T,N}(val, to_size(siz))
MutableUniformArray{T}(val, siz::NTuple{N,Integer}) where {T,N} =
    MutableUniformArray{T,N}(val, to_size(siz))
MutableUniformArray{T,N}(val, siz::NTuple{N,Integer}) where {T,N} =
    MutableUniformArray{T,N}(val, to_size(siz))

Base.eltype(::AbstractUniformArray{T}) where {T} = T

Base.ndims(::AbstractUniformArray{T,N}) where {T,N} = N

Base.length(A::AbstractUniformArray) = A.len

Base.size(A::AbstractUniformArray) = A.siz
Base.size(A::AbstractUniformArray{T,N}, i::Integer) where {T,N} =
    (i < 1 ? bad_dimension_index() : i ≤ N ? A.siz[i] : 1)

Base.axes1(A::AbstractUniformArray{T,0}) where {T} = Base.OneTo(1)
Base.axes1(A::AbstractUniformArray{T,N}) where {T,N} = Base.OneTo(A.siz[1])
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
    (i == length(A) == 1) || not_all_elements()
    A.val = x
end

@inline function Base.setindex!(A::MutableUniformArray, x,
                                i::AbstractUnitRange{<:Integer})
    (first(i) == 1 && last(i) == length(A)) || not_all_elements()
    A.val = x
end

@inline function Base.setindex!(A::MutableUniformArray, x, ::Colon)
    A.val = x
end

@noinline not_all_elements() = error("all elements must be set at the same time")

@noinline bad_dimension_index() = error("out of range dimension index")

@noinline bad_dimension_length() = error("invalid dimension length")

# Methods to convert size argument to canonic form.
to_size(siz::Tuple{Vararg{Int}}) = siz
to_size(siz::Tuple{Vararg{Integer}}) = map(to_int, siz)
to_size(siz::Integer) = (to_int(siz),)

# Convert to integer type suitable for indexing.
to_int(i::Int) = i
to_int(i::Integer) = Int(i)

end # module
