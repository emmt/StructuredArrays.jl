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

# Constructors for structured vectors and matrices.
for (A, N) in ((:Vector, 1), (:Matrix, 2))
    @eval begin
        $(Symbol("Structured",A))(args...) = $(Symbol("Structured",A))(IndexCartesian, args...)
        function $(Symbol("Structured",A))(::Union{S,Type{S}}, func, args...) where {S<:ConcreteIndexStyle}
            T = guess_eltype(func, S, Val($N))
            return StructuredArray{T,$N,S}(func, args...)
        end
        $(Symbol("Structured",A)){T}(args...) where {T} =
            StructuredArray{T,$N,IndexCartesian}(args...)
        $(Symbol("Structured",A)){T}(::Union{S,Type{S}}, args...) where {T,S<:ConcreteIndexStyle} =
            StructuredArray{T,$N,S}(args...)
        $(Symbol("Structured",A)){T,S}(args...) where {T,S<:ConcreteIndexStyle} =
            StructuredArray{T,$N,S}(args...)
    end
end

# For structured arrays, the index style may be specified as the first argument.
let cls = :StructuredArray
    # 0-dimensional case.
    @eval begin
        $cls(       ::Union{S,Type{S}}, func) where {  S<:ConcreteIndexStyle} = $cls(    S, func, ())
        $cls{T}(    ::Union{S,Type{S}}, func) where {T,S<:ConcreteIndexStyle} = $cls{T}( S, func, ())
        $cls{T,0}(  ::Union{S,Type{S}}, func) where {T,S<:ConcreteIndexStyle} = $cls{T,0,S}(func, ())
        $cls{T,0,S}(::Union{S,Type{S}}, func) where {T,S<:ConcreteIndexStyle} = $cls{T,0,S}(func, ())
        $cls{T,0,S}(                    func) where {T,S<:ConcreteIndexStyle} = $cls{T,0,S}(func, ())
    end
    # N-dimensional cases with N ≥ 1.
    for type in (:(Vararg{Union{Integer,AbstractRange{<:Integer}},N}),
                 :(NTuple{N,Union{Integer,AbstractRange{<:Integer}}}))
        @eval begin
            $cls(       ::Union{S,Type{S}}, func, args::$type) where {  N,S<:ConcreteIndexStyle} = $cls(    S, func, to_inds(args))
            $cls{T}(    ::Union{S,Type{S}}, func, args::$type) where {T,N,S<:ConcreteIndexStyle} = $cls{T}( S, func, to_inds(args))
            $cls{T,N}(  ::Union{S,Type{S}}, func, args::$type) where {T,N,S<:ConcreteIndexStyle} = $cls{T,N,S}(func, to_inds(args))
            $cls{T,N,S}(::Union{S,Type{S}}, func, args::$type) where {T,N,S<:ConcreteIndexStyle} = $cls{T,N,S}(func, to_inds(args))
            $cls{T,N,S}(                    func, args::$type) where {T,N,S<:ConcreteIndexStyle} = $cls{T,N,S}(func, to_inds(args))
        end
    end
    # Default index style is Cartesian for StructuredArray.
    @eval begin
        $cls(     func, args::Inds{N}) where {  N} = $cls(     IndexCartesian, func, args)
        $cls{T}(  func, args::Inds{N}) where {T,N} = $cls{T,N}(IndexCartesian, func, args)
        $cls{T,N}(func, args::Inds{N}) where {T,N} = $cls{T,N}(IndexCartesian, func, args)
    end
end

# Constructors that manage to call the inner constructors.
function StructuredArray(::Union{S,Type{S}}, func, args::Inds{N}) where {N,S<:ConcreteIndexStyle}
    T = guess_eltype(func, S, Val(N))
    return StructuredArray{T,N,S}(func, args)
end
StructuredArray{T}(::Union{S,Type{S}}, func, args::Inds{N}) where {T,N,S<:ConcreteIndexStyle} =
    StructuredArray{T,N,S}(func, args)
StructuredArray{T,N}(::Union{S,Type{S}}, func, args::Inds{N}) where {T,N,S<:ConcreteIndexStyle} =
    StructuredArray{T,N,S}(func, args)

guess_eltype(func, ::Type{IndexLinear}, ::Val{N}) where {N} =
    Base.promote_op(func, Int)

guess_eltype(func, ::Type{IndexCartesian}, ::Val{N}) where {N} =
    Base.promote_op(func, ntuple(Returns(Int), Val(N))...)

# See comments near `getindex` in `abstractarray.jl` for explanations about how `getindex`
# and `setindex!` methods are expected to be specialized depending on the indexing style:
# `i:Int` for linear indexing style or `i::Vararg{Int,N}` for `N` dimensional Cartesian
# indexing style.

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
