"""
    StructuredArray([S = IndexCartesian,] func, args...) -> A
    StructuredArray{T}([S = IndexCartesian,] func, args...) -> A
    StructuredArray{T,N}([S = IndexCartesian,] func, args...) -> A
    StructuredArray{T,N,S}(func, args...) -> A

build a `N`-dimensional array `A` whose values at index `i` are computed as `func(i)` and
whose shape is specified by `args...`. The storage requirement is `O(1)` instead of
`O(lenght(A))` for an ordinary array. Optional parameters `T`, `N`, and `S` are to specify
the element type, the number of dimensions, and the type of the indexing style of the
array. If specified as an argument, not as a type parameter, `S` may also be an instance
of `IndexStyle`.

The function `func` is called with the index specified as in base method `getindex`: if
indexing style is `IndexCartesian` (the default), the function `func` will be called with
`N` integer arguments (a `Vararg{Int,N}`); if `S` is `IndexLinear`, the function `func`
will be called with a single integer argument (an `Int`).

For example, the structure of a lower triangular matrix of size `m×n` could be given by:

    StructuredArray((i,j) -> (i ≥ j), m, n)

but with a constant small storage requirement whatever the size of the matrix.

Although the callable object `func` may not be a *pure function*, its return type shall be
stable and structured arrays are considered as immutable in the sense that a statement
like `A[i] = x` is not implemented. If parameter `T` is not specified in the call to the
constructor, the type of the elements of structured array is inferred by applying `func`
to the unit index.

""" StructuredArray

# The following is the only outer constructor that calls the inner constructor. It takes care of
# checking and converting the array shape arguments.
function StructuredArray{T,N,S}(func, inds::NTuple{N,AxisLike}) where {T,N,S<:ConcreteIndexStyle}
    check_shape(inds)
    return StructuredArray{T,N,S}(BareBuild(), func, as_shape(inds))
end

# If element type `T` is unspecified, all constructors call the following one.
function StructuredArray(::Union{S,Type{S}}, func, inds::NTuple{N,AxisLike}) where {N,S<:ConcreteIndexStyle}
    T = guess_eltype(func, S, Val(N))
    return StructuredArray{T,N,S}(func, inds)
end

guess_eltype(func, ::Type{IndexLinear}, ::Val{N}) where {N} =
    Base.promote_op(func, Int)

guess_eltype(func, ::Type{IndexCartesian}, ::Val{N}) where {N} =
    Base.promote_op(func, ntuple(Returns(Int), Val(N))...)

# Other outer constructors for `StructuredArray` objects.
for I in (:(NTuple{N,AxisLike}), :(Vararg{AxisLike,N}))
    @eval begin
        # Default index style is `IndexCartesian` for `StructuredArray` objects.
        StructuredArray(     func, inds::$I) where {  N} = StructuredArray(    IndexCartesian, func, inds)
        StructuredArray{T}(  func, inds::$I) where {T,N} = StructuredArray{T,N,IndexCartesian}(func, inds)
        StructuredArray{T,N}(func, inds::$I) where {T,N} = StructuredArray{T,N,IndexCartesian}(func, inds)
        # Constructors with specified element type `T` and index style `S`.
        StructuredArray{T}(    ::Union{S,Type{S}}, func, inds::$I) where {T,N,S<:ConcreteIndexStyle} = StructuredArray{T,N,S}(func, inds)
        StructuredArray{T,N}(  ::Union{S,Type{S}}, func, inds::$I) where {T,N,S<:ConcreteIndexStyle} = StructuredArray{T,N,S}(func, inds)
        StructuredArray{T,N,S}(::Union{S,Type{S}}, func, inds::$I) where {T,N,S<:ConcreteIndexStyle} = StructuredArray{T,N,S}(func, inds)
    end
    I.args[1] === :Vararg || continue
    @eval begin
        # These constructors are defined elsewhere when shape is specified as a tuple.
        StructuredArray(::Union{S,Type{S}}, func, inds::$I) where {  N,S<:ConcreteIndexStyle} = StructuredArray(    S, func, inds)
        StructuredArray{T,N,S}(             func, inds::$I) where {T,N,S<:ConcreteIndexStyle} = StructuredArray{T,N,S}(func, inds)
    end
end

# Constructors for structured vectors and matrices.
for (A, N) in ((:Vector, 1), (:Matrix, 2)), I in (Vararg{AxisLike,N}, NTuple{N,AxisLike})
    @eval begin
        $(Symbol(:Structured,A))(                       func, inds::$I)                                 = StructuredArray(      func, inds)
        $(Symbol(:Structured,A))(   ::Union{S,Type{S}}, func, inds::$I) where {  S<:ConcreteIndexStyle} = StructuredArray(   S, func, inds)
        $(Symbol(:Structured,A)){T}(                    func, inds::$I) where {T}                       = StructuredArray{T}(   func, inds)
        $(Symbol(:Structured,A)){T}(::Union{S,Type{S}}, func, inds::$I) where {T,S<:ConcreteIndexStyle} = StructuredArray{T}(S, func, inds)
    end
end

# See comments near `getindex` in `abstractarray.jl` for explanations about how `getindex`
# `getindex` and `setindex!` methods are expected to be specialized depending on the
# indexing style: `i:Int` for linear indexing style or `i::Vararg{Int,N}` for `N`
# dimensional Cartesian indexing style.

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
