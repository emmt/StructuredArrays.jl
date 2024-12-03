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
Base.has_offset_axes(A::AbstractUniformArray{T,0,Tuple{}}) where {T} = false
Base.has_offset_axes(A::AbstractUniformArray{T,N,Dims{N}}) where {T,N} = false
Base.has_offset_axes(A::StructuredArray{T,0,S,F,Tuple{}}) where {T,S,F} = false
Base.has_offset_axes(A::StructuredArray{T,N,S,F,Dims{N}}) where {T,N,S,F} = false

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

# Constructors that convert trailing argument(s) to array dimensions or axes. `arg1` is
# the value for a uniform array, the function for a structured array.
for cls in (:StructuredArray, :FastUniformArray, :UniformArray, :MutableUniformArray)
    # 0-dimensional case.
    @eval begin
        $cls(     arg1)             = $cls(     arg1, ())
        $cls{T}(  arg1) where {T}   = $cls{T}(  arg1, ())
        $cls{T,N}(arg1) where {T,N} = $cls{T,N}(arg1, ())
    end
    # N-dimensional cases with N ≥ 1.
    for type in (:(Vararg{Union{Integer,AbstractRange{<:Integer}},N}),
                 :(NTuple{N,Union{Integer,AbstractRange{<:Integer}}}))
        @eval begin
            $cls(     arg1, args::$type) where {  N} = $cls(     arg1, to_inds(args))
            $cls{T}(  arg1, args::$type) where {T,N} = $cls{T}(  arg1, to_inds(args))
            $cls{T,N}(arg1, args::$type) where {T,N} = $cls{T,N}(arg1, to_inds(args))
        end
    end
end

"""
    StructuredArrays.checked_indices(inds) -> inds

throws an `ArgumentError` exception if `inds` is not valid array dimensions or axes and
returns `inds` otherwise.

"""
checked_indices(inds::Tuple{Vararg{AbstractUnitRange{Int}}}) = inds # no needs to check
checked_indices(::Tuple{}) = () # no needs to check
@inline checked_indices(inds::Inds) =
    checked_indices(Bool, inds...) ? inds : throw(ArgumentError("invalid array dimensions or axes"))

#checked_indices(::Type{Bool}) = true
checked_indices(::Type{Bool}, a) = is_length_or_unit_range(a)
@inline checked_indices(::Type{Bool}, a, b...) =
    is_length_or_unit_range(a) & checked_indices(Bool, b...)

is_length_or_unit_range(x::Any) = false
is_length_or_unit_range(dim::Integer) = dim ≥ zero(dim)
is_length_or_unit_range(rng::AbstractUnitRange{<:Integer}) = true
is_length_or_unit_range(rng::AbstractRange{<:Integer}) = isone(step(rng))

to_dim_or_axis(arg::Integer) = as_array_dim(arg)
to_dim_or_axis(arg::AbstractRange{<:Integer}) = as_array_axis(arg)
to_dim_or_axis(arg::Base.OneTo{<:Integer}) = as_array_dim(arg)

to_inds(args::Tuple{Vararg{Union{Integer,AbstractRange{<:Integer}}}}) =
    map(to_dim_or_axis, args)

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
