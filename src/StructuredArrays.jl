#
# StructuredArrays.jl --
#
# Implement arrays whose elements have values that only depend on the indices.
# Exemple of such arrays are uniform arrays with constant values, structured
# arrays with boolean values indicating whether an entry is significant.
#
module StructuredArrays

export
    AbstractStructuredArray,
    AbstractStructuredMatrix,
    AbstractStructuredVector,
    AbstractUniformArray,
    AbstractUniformMatrix,
    AbstractUniformVector,
    MutableUniformArray,
    MutableUniformMatrix,
    MutableUniformVector,
    StructuredArray,
    StructuredMatrix,
    StructuredVector,
    UniformArray,
    UniformMatrix,
    UniformVector

import Base: @propagate_inbounds

abstract type AbstractStructuredArray{T,N,S<:IndexStyle} <:
    AbstractArray{T,N} end

abstract type AbstractUniformArray{T,N} <:
    AbstractStructuredArray{T,N,IndexLinear} end

"""
    UniformArray(val, siz) -> A

yields an array `A` which behaves as an immutable array of size `siz` whose
values are all equal to `val`.  The storage requirement is `O(1)` instead of
`O(prod(siz))` for a usual array.  The array dimensions may be specified as
multiple arguments.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val`
for all linear indices `i` in the range `1:length(A)`.

A statement like `A[i] = val` is not implemented as uniform arrays are
considered as immutable.  Call `MutableUniformArray(val,siz)` to create a
uniform array whose element value can be changed.

""" UniformArray

struct UniformArray{T,N} <: AbstractUniformArray{T,N}
    len::Int
    siz::NTuple{N,Int}
    val::T
    UniformArray{T,N}(val, siz::NTuple{N,Int}) where {T,N} =
        new{T,N}(checksize(siz), siz, val)
end

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

mutable struct MutableUniformArray{T,N} <: AbstractUniformArray{T,N}
    len::Int
    siz::NTuple{N,Int}
    val::T
    MutableUniformArray{T,N}(val, siz::NTuple{N,Int}) where {T,N} =
        new{T,N}(checksize(siz), siz, val)
end

"""
    StructuredArray([S = IndexCartesian,] fnc, siz) -> A

yields an array `A` which behaves as an array of size `siz` whose values are a
given function, here `fnc`, of its indices: `A[i]` is computed as `fnc(i)`.
The storage requirement is `O(1)` instead of `O(prod(siz))` for a usual array.
The array dimensions may be specified as multiple arguments.

The optional argument `S` may be used to specifiy another index style than the
default `IndexCartesian`, for instance `IndexLinear`.  If specified, `S` may be
a sub-type of `IndexStyle` or an instance of such a sub-type.  If `S` is
`IndexCartesian` (the default), the function `fnc` will be called with `N`
integer arguments, `N` being the number of dimensions.  If `S` is
`IndexCartesian`, the function `fnc` will be called with a single integer
argument.

For instance, the structure of a lower triangular matrix of size `m×n` would be
given by:

    StructuredArray((i,j) -> (i ≥ j), m, n)

but with a constant small storage requirement whatever the size of the matrix.

Although the callable object `fnc` may not be a *pure function*, its return
type shall be stable and structured arrays are considered as immutable in the
sense that a statement like `A[i] = val` is not implemented.  The type of the
elements of structured array is guessed by applying `fnc` to the unit index.
The element type, say `T`, may also be explicitely specified:

    StructuredArray{T}([S = IndexCartesian,] fnc, siz)

""" StructuredArray

struct StructuredArray{T,N,S,F} <: AbstractStructuredArray{T,N,S}
    len::Int
    siz::NTuple{N,Int}
    fnc::F
    StructuredArray{T,N,S,F}(fnc, siz::NTuple{N,Int}) where {T,N,S,F} =
        new{T,N,S,F}(checksize(siz), siz, fnc)
end

const AbstractStructuredVector{T,S} = AbstractStructuredArray{T,1,S}
const StructuredVector{T,S,F} = StructuredArray{T,1,S,F}
const AbstractUniformVector{T} = AbstractUniformArray{T,1}
const MutableUniformVector{T} = MutableUniformArray{T,1}
const UniformVector{T} = UniformArray{T,1}

const AbstractStructuredMatrix{T,S} = AbstractStructuredArray{T,2,S}
const StructuredMatrix{T,S,F} = StructuredArray{T,2,S,F}
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

# All constructors for StructuredArray are based on the 2 first ones (only
# depending on whether parameter T is provided or not) so the other constructors
# just make sure and arguments have correct type.

function StructuredArray(::Type{S},
                         fnc::F,
                         siz::NTuple{N,Int}) where {N,S<:IndexStyle,F}
    T = guess_eltype(fnc, S, Val(N))
    StructuredArray{T,N,S,F}(fnc, siz)
end

function StructuredArray{T}(::Type{S},
                            fnc::F,
                            siz::NTuple{N,Int}) where {T,N,S<:IndexStyle,F}
    StructuredArray{T,N,S,F}(fnc, siz)
end

guess_eltype(fnc, ::Type{IndexLinear}, ::Val{N}) where {N} =
    typeof(fnc(one(Int)))

guess_eltype(fnc, ::Type{IndexCartesian}, ::Val{N}) where {N} =
    typeof(fnc(ntuple(i -> one(Int), Val(N))...))

# Index style specified and size specified as a tuple or by trailing arguments.

function StructuredArray(::Union{S,Type{S}}, fnc,
                         siz::Integer...) where {S<:IndexStyle}
    StructuredArray(S, fnc, siz)
end
function StructuredArray(::Union{S,Type{S}}, fnc,
                         siz::NTuple{N,Integer}) where {N,S<:IndexStyle}
    StructuredArray(S, fnc, to_size(siz))
end

function StructuredArray{T}(::Union{S,Type{S}}, fnc,
                            siz::Integer...) where {T,S<:IndexStyle}
    StructuredArray{T}(S, fnc, siz)
end
function StructuredArray{T}(::Union{S,Type{S}}, fnc,
                            siz::NTuple{N,Integer}) where {T,N,S<:IndexStyle}
    StructuredArray{T}(S, fnc, to_size(siz))
end

function StructuredArray{T,N}(::Union{S,Type{S}}, fnc,
                              siz::Integer...) where {T,N,S<:IndexStyle}
    StructuredArray{T,N}(S, fnc, siz) # keep the N to check
end
function StructuredArray{T,N}(::Union{S,Type{S}}, fnc,
                              siz::NTuple{N,Integer}) where {T,N,S<:IndexStyle}
    StructuredArray{T}(S, fnc, to_size(siz))
end

function StructuredArray{T,N,S}(::Union{S,Type{S}}, fnc,
                                siz::Integer...) where {T,N,S<:IndexStyle}
    StructuredArray{T,N}(S, fnc, siz) # keep the N to check
end
function StructuredArray{T,N,S}(::Union{S,Type{S}}, fnc,
                                siz::NTuple{N,Integer}) where {T,N,S<:IndexStyle}
    StructuredArray{T}(S, fnc, to_size(siz))
end

function StructuredArray{T,N,S}(fnc,
                                siz::Integer...) where {T,N,S<:IndexStyle}
    StructuredArray{T,N}(S, fnc, siz) # keep the N to check
end
function StructuredArray{T,N,S}(fnc,
                                siz::NTuple{N,Integer}) where {T,N,S<:IndexStyle}
    StructuredArray{T}(S, fnc, to_size(siz))
end

# Index style not specified and size specified as a tuple or by trailing
# arguments.

StructuredArray(fnc, siz::Integer...) =
    StructuredArray(fnc, siz)
StructuredArray(fnc, siz::NTuple{N,Integer}) where {N} =
    StructuredArray(IndexCartesian, fnc, to_size(siz))

StructuredArray{T}(fnc, siz::Integer...) where {T} =
    StructuredArray{T}(fnc, siz)
StructuredArray{T}(fnc, siz::NTuple{N,Integer}) where {T,N} =
    StructuredArray{T}(IndexCartesian, fnc, to_size(siz))

StructuredArray{T,N}(fnc, siz::Integer...) where {T,N} =
    StructuredArray{T,N}(fnc, siz) # keep the N to check
StructuredArray{T,N}(fnc, siz::NTuple{N,Integer}) where {T,N} =
    StructuredArray{T}(fnc, siz)

Base.eltype(::AbstractStructuredArray{T}) where {T} = T

Base.ndims(::AbstractStructuredArray{T,N}) where {T,N} = N

for X in (:StructuredArray, :UniformArray, :MutableUniformArray)
    @eval begin
        Base.length(A::$X) = getfield(A, :len)
        Base.size(A::$X) = getfield(A, :siz)
        Base.size(A::$X{T,N}, i::Integer) where {T,N} =
            (i < 1 ? bad_dimension_index() : i ≤ N ? @inbounds(size(A)[i]) : 1)
        Base.axes1(A::$X{T,0}) where {T} = Base.OneTo(1)
        Base.axes1(A::$X{T,N}) where {T,N} = Base.OneTo(size(A)[1])
        Base.axes(A::$X) = map(Base.OneTo, size(A))
        Base.axes(A::$X, i::Integer) = Base.OneTo(size(A, i))
        Base.has_offset_axes(::$X) = false
    end
end

# See comments near `getindex` in `abstractarray.jl` for explanations about how
# `getindex` and `setindex!` methods are expected to be specialized depending
# on the indexing style.

Base.IndexStyle(::Type{<:AbstractStructuredArray{T,N,S}}) where {T,N,S} = S()

@inline function Base.getindex(A::StructuredArray{T,N,IndexLinear},
                               i::Int) :: T where {T,N}
    @boundscheck checkbounds(A, i)
    A.fnc(i)
end

@inline function Base.getindex(A::StructuredArray{T,N,IndexCartesian},
                               inds::Vararg{Int, N}) :: T where {T,N}
    @boundscheck checkbounds(A, inds...)
    A.fnc(inds...)
end

@inline function Base.getindex(A::X,
                               i::Int) where {X<:Union{<:UniformArray,
                                                       <:MutableUniformArray}}
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

"""
    checksize(dims) -> len

yields the number of elements of an array of size `dims` throwing an error if
any dimension is invalid.

"""
function checksize(dims::Dims{N}) where {N}
    len = 1
    @inbounds for i in 1:N
        (dim = dims[i]) ≥ 0 || bad_dimension_length()
        len *= dim
    end
    return len
end

@noinline not_all_elements() =
    error("all elements must be set at the same time")

@noinline bad_dimension_index() = error("out of range dimension index")

@noinline bad_dimension_length() = error("invalid dimension length")

# Methods to convert size argument to canonic form.
to_size(siz::Dims) = siz
to_size(siz::Tuple{Vararg{Integer}}) = map(to_int, siz)
to_size(siz::Integer) = (to_int(siz),)

# Convert to integer type suitable for indexing.
to_int(i::Int) = i
to_int(i::Integer) = Int(i)

end # module
