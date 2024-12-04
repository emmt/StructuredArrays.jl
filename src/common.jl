# Specialize base abstract array methods for structured and uniform arrays.
for cls in (:StructuredArray, :FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        indices(A::$cls) = getfield(A, :inds)
        Base.length(A::$cls) = prod(size(A))
        Base.size(A::$cls) = as_array_size(indices(A))
        Base.size(A::$cls, i::Integer) =
            i > ndims(A) ? 1 :
            i > zero(i) ? as_array_dim(indices(A)[i]) : throw(BoundsError(size(A), i))
        Base.axes(A::$cls) = as_array_axes(indices(A))
        Base.axes(A::$cls, i::Integer) =
            i > ndims(A) ? Base.OneTo(1) :
            i > zero(i) ? as_array_axis(indices(A)[i]) : throw(BoundsError(axes(A), i))
     end
end
Base.has_offset_axes(A::AbstractStructuredArray{T,0,S,Tuple{}}) where {T,S} = false
Base.has_offset_axes(A::AbstractStructuredArray{T,N,S,Dims{N}}) where {T,N,S} = false

# `copy(A)` and `deepcopy(A)` simply yield `A` if it is immutable.
Base.copy(A::UniformArray) = A
Base.copy(A::StructuredArray) = A
Base.copy(A::FastUniformArray) = A
Base.copy(A::MutableUniformArray{T}) where {T} =
    MutableUniformArray{T}(BareBuild(), value(A), indices(A))

Base.deepcopy(A::UniformArray) = A
Base.deepcopy(A::StructuredArray) = A
Base.deepcopy(A::FastUniformArray) = A
Base.deepcopy(A::MutableUniformArray) = copy(A)

"""
    as_shape(x)

converts `x` as a proper array shape that is an array dimension length, a unit-step array
axis, or a tuple of these if `x` is a tuple. Instances of `Base.OneTo` are replaced by
their length. All integers are converted to `Int`s if needed. The result is an instance
(or a tuple) of `Union{Int,AbstractUnitRange{Int}`.

Call `as_shape(Tuple, x)` to ensure that the shape is returned as a tuple.

""" as_shape
# NOTE `map(f,x)` with `x` a tuple yields good code for `@code_warntype` and `@benchmark`
#      provided `f` is a simple function whose output can be inferred. Branching or
#      throwing in `f` breaks this, so we cannot check for argument validity while
#      converting array indices to a proper shape and the checking of the specified
#      indices is done in a separate function `check_shape`.
as_shape(dim::Integer) = as(Int, dim)
as_shape(rng::Base.OneTo{<:Integer}) = as(Int, length(rng))
as_shape(rng::AbstractUnitRange{<:Integer}) = as(AbstractUnitRange{Int}, rng)
as_shape(inds::Tuple{}) = ()
as_shape(inds::Tuple{AxisLike, Vararg{AxisLike}}) = map(as_shape, inds)
as_shape(Tuple, inds::Tuple{Vararg{AxisLike}}) = as_shape(inds)
as_shape(Tuple, x::AxisLike) = (as_shape(x),)

"""
    check_shape(x)

throws an exception if `x` has invalid array indices such that `as_shape(x)` would not
yield a proper array shape.

"""
check_shape(dim::Integer) = dim ≥ zero(dim) || throw_bad_dimension(dim)
check_shape(rng::AbstractUnitRange{<:Integer}) = nothing
check_shape(rng::AbstractRange{<:Integer}) = isone(step(rng)) || throw_nonunit_step(rng)
@noinline check_shape(x::Any) = throw(ArgumentError(
    "invalid argument of type `$(typeof(x))` for array shape"))

@noinline throw_bad_dimension(dim::Integer) = throw(ArgumentError(
    "array dimension must be nonnegative"))

@noinline throw_nonunit_step(rng::AbstractRange) = throw(ArgumentError(
    "range has non-unit step"))

# NOTE A loop such as `for x in inds; check_shape(x); end` is terrible in terms of
#      performances. `foreach` is much better, at least in recent versions of Julia (≥
#      1.8).
check_shape(inds::Tuple{}) = nothing
check_shape(inds::Tuple{AxisLike, Vararg{AxisLike}}) = foreach(check_shape, inds)

print_axis(io::IO, rng::Base.OneTo) = print(io, length(rng))
print_axis(io::IO, rng::AbstractUnitRange{<:Integer}) = print(io, first(rng), ':', last(rng))

print_axes(io::IO, A::AbstractArray) = print_axes(io, axes(A))
function print_axes(io::IO, rngs::NTuple{N,AbstractUnitRange{<:Integer}}) where {N}
    flag = false
    for rng in rngs
        flag && print(io, ", ")
        print_axis(io, rng)
        flag = true
    end
end
