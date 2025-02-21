# Specialize base abstract array methods for structured and uniform arrays.
for cls in (:StructuredArray, :FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        shape(A::$cls) = getfield(A, :inds)
        Base.length(A::$cls) = prod(size(A))
        Base.size(A::$cls) = as_array_size(shape(A))
        Base.size(A::$cls, i::Integer) =
            i > ndims(A) ? 1 :
            i > zero(i) ? as_array_dim(shape(A)[i]) : throw(BoundsError(size(A), i))
        Base.axes(A::$cls) = as_array_axes(shape(A))
        Base.axes(A::$cls, i::Integer) =
            i > ndims(A) ? Base.OneTo(1) :
            i > zero(i) ? as_array_axis(shape(A)[i]) : throw(BoundsError(axes(A), i))
     end
end
Base.has_offset_axes(A::AbstractStructuredArray{T,N,S,Dims{N}}) where {T,N,S} = false

# `copy(A)` and `deepcopy(A)` simply yield `A` if it is immutable.
for cls in (:StructuredArray, :FastUniformArray, :UniformArray)
    @eval begin
        Base.copy(A::$cls) = A
        Base.deepcopy(A::$cls) = A
    end
end
Base.copy(A::MutableUniformArray{T}) where {T} =
    MutableUniformArray{T}(BareBuild(), value(A), shape(A))
Base.deepcopy(A::MutableUniformArray) = copy(A)

function Base.Array(A::AbstractStructuredArray{T,N,S,Dims{N}}) where {T,N,S}
    B = Array{T,N}(undef, size(A))
    if A isa AbstractUniformArray
        fill!(B, value(A))
    else
        @inbounds for i in eachindex(A, B)
            B[i] = A[i]
        end
    end
    return B
end

function OffsetArrays.OffsetArray(A::AbstractStructuredArray{T,N}) where {T,N}
    X = Array{T,N}(undef, size(A))
    B = OffsetArray(X, axes(A))
    if A isa AbstractUniformArray
        fill!(X, value(A))
    else
        @inbounds for i in eachindex(A, B)
            B[i] = A[i]
        end
    end
    return B
end

"""
    StructuredArrays.shape(A)

yields the shape of array `A`, the result is similar to `axes(A)` except that `Base.OneTo`
axes are replaced by their lengths. Hence, for an ordinary arrays, `shape(A) === size(A)`
holds. `AbstractStructuredArray` objects directly store their shape.

"""
shape(A::Array) = size(A)
shape(A::AbstractArray) = checked_shape(axes(A))

shape_type(A::AbstractStructuredArray) = shape_type(typeof(A))
shape_type(::Type{<:AbstractStructuredArray{T,N,S,I}}) where {T,N,S,I} = I

"""
    StructuredArrays.checked_shape(x)

converts `x` as a proper array shape that is an array dimension length, a unit-step array
axis, or a tuple of these if `x` is a tuple. Instances of `Base.OneTo` are replaced by
their length. All integers are converted to `Int`s if needed. The result is a tuple of
`Union{Int,AbstractUnitRange{Int}`.

"""
checked_shape(inds::Tuple{}) = ()
checked_shape(inds::Vararg{AxisLike}) = checked_shape(inds)
@inline checked_shape(inds::Tuple{Vararg{AxisLike}}) = map(checked_shape_elem, inds)

@noinline checked_shape(x...) = throw(ArgumentError(
    "invalid argument of type `$(typeof(x))` for array shape"))

checked_shape_elem(dim::Integer) = dim ≥ zero(dim) ? as(Int, dim) : throw_bad_dimension(dim)
checked_shape_elem(rng::Base.OneTo) = checked_shape_elem(length(rng))
checked_shape_elem(rng::AbstractUnitRange{Int}) = rng
checked_shape_elem(rng::AbstractUnitRange{<:Integer}) = as(AbstractUnitRange{Int}, rng)
checked_shape_elem(rng::AbstractRange{<:Integer}) =
    isone(step(rng)) ? checked_shape_elem(first(rng):last(rng)) : throw_nonunit_step(rng)

@noinline throw_bad_dimension(dim::Integer) = throw(ArgumentError(
    "array dimension must be nonnegative"))

@noinline throw_nonunit_step(rng::AbstractRange) = throw(ArgumentError(
    "range has non-unit step"))

# Similar to `checked_shape` but more strict and just check indices.
check_shape_strict(inds::Tuple{}) = nothing
check_shape_strict(inds::Tuple{Vararg{AbstractUnitRange{Int}}}) = nothing
check_shape_strict(inds::Inds) = map(check_shape_elem_strict, inds)

check_shape_elem_strict(dim::Int) = dim ≥ zero(dim) ? nothing : throw_bad_dimension(dim)
check_shape_elem_strict(rng::AbstractUnitRange{Int}) = nothing

print_axis(io::IO, rng::Base.OneTo) = print(io, length(rng))
print_axis(io::IO, rng::AbstractUnitRange{<:Integer}) = print(io, first(rng), ':', last(rng))

print_axes(io::IO, A::AbstractArray; kwds...) = print_axes(io, axes(A); kwds...)
function print_axes(io::IO, rngs::NTuple{N,AbstractUnitRange{<:Integer}};
                    as_tuple::Bool=false) where {N}
    as_tuple && print(io, "(")
    flag = false
    for rng in rngs
        flag && print(io, ", ")
        print_axis(io, rng)
        flag = true
    end
    as_tuple && print(io, N == 1 ? ",)" : ")")
end

"""
    unrolled_mapfoldl(f::Symbol, op::Symbol, x::Symbol, [from::Int = 1], to::Int)

yields an expression corresponding to the unrolled code of `mapfoldl(f,op,x)` for the
entries of `x` at indices `from:to`.

"""
unrolled_mapfoldl(f::Symbol, op::Symbol, x::Symbol, n::Integer) =
    unrolled_mapfoldl(f, op, x, 1, as(Int, n))
function unrolled_mapfoldl(f::Symbol, op::Symbol, x::Symbol, i::Int, j::Int)
    b = :($f($x[$j]))
    i < j || return b
    a = unrolled_mapfoldl(f, op, x, i, j - 1)
    return :($op($a, $b))
end

"""
    unrolled_mapfoldr(f::Symbol, op::Symbol, x::Symbol, [from::Int = 1], to::Int)

yields an expression corresponding to the unrolled code of `mapfoldr(f,op,x)` for the
entries of `x` at indices `from:to`.

"""
unrolled_mapfoldr(f::Symbol, op::Symbol, x::Symbol, n::Integer) =
    unrolled_mapfoldr(f, op, x, 1, as(Int, n))
function unrolled_mapfoldr(f::Symbol, op::Symbol, x::Symbol, i::Int, j::Int)
    a = :($f($x[$i]))
    i < j || return a
    b = unrolled_mapfoldr(f, op, x, i + 1, j)
    return :($op($a, $b))
end
