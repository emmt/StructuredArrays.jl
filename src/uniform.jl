"""
    UniformArray(val, args...) -> A
    UniformArray{T}(val, args...) -> A
    UniformArray{T,N}(val, args...) -> A

build an immutable array `A` whose elements are all equal to `val` and whose axes are
specified by `args...`. The storage requirement is `O(1)` instead of `O(length(A))` for an
ordinary array `A`. Optional parameters `T` and `N` are to specify the element type and
the number of dimensions of the array.

Uniform arrays implement conventional linear indexing: `A[i]` yields `val` for all linear
indices `i` in the range `1:length(A)`.

A statement like `A[i] = x` is not implemented as uniform arrays are considered as
immutable. Call [`MutableUniformArray(val, dims)`](@ref) to create a mutable uniform
array.

""" UniformArray

"""
    FastUniformArray(val, args...) -> A
    FastUniformArray{T}(val, args...) -> A
    FastUniformArray{T,N}(val, args...) -> A

build an immutable uniform array `A` whose elements are all equal to `val` and whose axes
are specified by `args...`. The difference with an instance of [`UniformArray`](@ref) is
that `val` is part of the type signature so that `val` can be known at compile time. A
typical use is to create all true/false masks.

""" FastUniformArray

"""
    MutableUniformArray(val, args...) -> A
    MutableUniformArray{T}(val, args...) -> A
    MutableUniformArray{T,N}(val, args...) -> A

build a mutable array `A` whose elements are initially all equal to `val` and whose axes
are specified by `args...`. The difference with an instance of [`UniformArray`](@ref) is
that the uniform value can be changed.

A statement like `A[i] = x` is allowed to change the value of all the elements of `A`;
hence `i` must represent all indices of `A`. Call [`UniformArray(val, dims)`](@ref) to
create an immutable uniform array whose element value cannot be changed.

""" MutableUniformArray

# Constructors for uniform arrays. For each sub-type, all constructors call the last one
# that checks and converts the shape arguments and then calls the inner constructor.
for cls in (:FastUniformArray, :UniformArray, :MutableUniformArray)
    @eval begin
        $cls(     val::T, inds::Vararg{AxisLike,N}) where {T,N} = $cls{T}(val, inds)
        $cls{T}(  val,    inds::Vararg{AxisLike,N}) where {T,N} = $cls{T}(val, inds)
        $cls{T,N}(val,    inds::Vararg{AxisLike,N}) where {T,N} = $cls{T}(val, inds)
        $cls(     val::T, inds::NTuple{N,AxisLike}) where {T,N} = $cls{T}(val, inds)
        $cls{T,N}(val,    inds::NTuple{N,AxisLike}) where {T,N} = $cls{T}(val, inds)
        function $cls{T}(val, inds::NTuple{N,AxisLike}) where {T,N}
            check_shape(inds)
            return $cls{T}(BareBuild(), val, as_shape(inds))
        end
    end
end

# Constructors for uniform vectors and matrices.
for K in (:Uniform, :MutableUniform, :FastUniform), (A, N) in ((:Vector, 1), (:Matrix, 2))
    @eval begin
        $(Symbol(K,A))(val::T, inds::Vararg{AxisLike,$N}) where {T} = $(Symbol(K,:Array)){T}(val, inds)
        $(Symbol(K,A)){T}(val, inds::Vararg{AxisLike,$N}) where {T} = $(Symbol(K,:Array)){T}(val, inds)
        $(Symbol(K,A))(val::T, inds::NTuple{$N,AxisLike}) where {T} = $(Symbol(K,:Array)){T}(val, inds)
        $(Symbol(K,A)){T}(val, inds::NTuple{$N,AxisLike}) where {T} = $(Symbol(K,:Array)){T}(val, inds)
    end
end

function Base.reverse(A::AbstractUniformArray; dims::Union{Colon,Integer,Tuple{Vararg{Integer}},
                                                           AbstractVector{<:Integer}} = :)
    if !(dims isa Colon)
        for d in dims
            d ≥ one(d) || throw(ArgumentError("dimension must be ≥ 1, got $d"))
        end
    end
    return A
end

Base.unique(A::AbstractUniformArray; dims::Union{Colon,Integer} = :) = _unique(A, dims)

_unique(A::AbstractUniformArray, ::Colon) = [value(A)]
_unique(A::AbstractUniformArray, dim::Integer) =
    UniformArray(value(A), reduce_shape1(shape(A), dim))

# Optimize base reduction methods for uniform arrays.
for func in (:all, :any,
             :minimum, :maximum, :extrema,
             :count, :sum, :prod,
             :findmin, :findmax)
    _func = Symbol("_",func)
    @eval begin
        Base.$(func)(A::AbstractUniformArray; dims = :) = $(_func)(identity, A, dims)
        Base.$(func)(f::Function, A::AbstractUniformArray; dims = :) = $(_func)(f, A, dims)
    end
    if func ∈ (:all, :any)
        @eval begin
            function $(_func)(f, A::AbstractUniformArray, ::Colon)
                if isempty(A)
                    return _empty_result($(func))
                else
                    val = f(value(A))
                    return val isa Bool ? val : _unexpected_non_boolean(val)
                end
            end
            function $(_func)(f, A::AbstractUniformArray, dims)
                inds = reduce_shape(shape(A), dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    val = f(value(A))
                    val isa Bool || _unexpected_non_boolean(val)
                    return UniformArray(val, inds)
                end
            end
        end
    elseif func ∈ (:minimum, :maximum)
        @eval begin
            $(_func)(f, A::AbstractUniformArray, ::Colon) =
                isempty(A) ? _empty_result($(func)) : f(value(A))
            function $(_func)(f, A::AbstractUniformArray, dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    inds = reduce_shape(shape(A), dims)
                    val = f(value(A))
                    return UniformArray(val, inds)
                end
            end
        end
    elseif func === :extrema
        @eval begin
            function $(_func)(f, A::AbstractUniformArray, ::Colon)
                if isempty(A)
                    return _empty_result($(func))
                else
                    val = f(value(A))
                    return (val, val)
                end
            end
            function $(_func)(f, A::AbstractUniformArray, dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    inds = reduce_shape(shape(A), dims)
                    val = f(value(A))
                    return UniformArray((val,val), inds)
                end
            end
        end
    elseif func ∈ (:count, :sum, :prod)
        op = Symbol(_func,"_num_val")
        @eval begin
            function $(_func)(f, A::AbstractUniformArray, ::Colon)
                if isempty(A)
                    return _empty_result($(func))
                else
                    num = length(A)
                    val = f(value(A))
                    return $(op)(num, val)
                end
            end
            function $(_func)(f, A::AbstractUniformArray, dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    inds = reduce_shape(shape(A), dims)
                    num = div(length(A), prod(as_array_size(inds)))
                    val = f(value(A))
                    return UniformArray($(op)(num, val), inds)
                end
            end
        end
    elseif func ∈ (:findmin, :findmax)
        @eval begin
            function $(_func)(f, A::AbstractUniformArray, ::Colon)
                if isempty(A)
                    return _empty_result($(func))
                else
                    idx = CartesianIndex(map(first, axes(A)))
                    val = f(value(A))
                    return val, idx
                end
            end
            function $(_func)(f, A::AbstractUniformArray, dims)
                if isempty(A)
                    return _empty_result($(func))
                else
                    inds = reduce_shape(shape(A), dims)
                    val = f(value(A))
                    rngs = map(as_array_axis, inds)
                    I = CartesianIndices(rngs)
                    if shape_type(A) <: Dims
                        return UniformArray(val, size(I)), I
                    else
                        return UniformArray(val, inds), OffsetArray(I, rngs)
                    end
                end
            end
        end
    end
end

# `reduce_shape(inds,region)` reduces the dimensions in `region` for the array shape
# `inds`. For type-stability in reduction operations, `reduce_shape` shall return an
# object of the same type as `inds` when `region` is an integer or a list of integers.
# This is achieved by having `reduce_axis` return an object of the same tayp as its
# argument.
reduce_shape(inds::Inds, ::Colon) = ()
function reduce_shape(inds::Inds{N}, dim::Integer, reduce=reduce_axis) where {N}
    check_reduce_dim(minimum(dim))
    return ntuple(d -> d == dim ? reduce(inds[d]) : inds[d], Val(N))
end
reduce_shape(inds::Inds, dims::Tuple{}) = inds
function reduce_shape(inds::Inds{N},
                       dims::Union{AbstractVector{<:Integer},
                                   Tuple{Integer,Vararg{Integer}}}) where {N}
    check_reduce_dim(minimum(dims))
    return ntuple(d -> d ∈ dims ? reduce_axis(inds[d]) : inds[d], Val(N))
end
reduce_shape(inds::Inds, dims) = throw(ArgumentError(
    "region dimension(s) must be a colon, an integer, or a vector/tuple of integers"))

check_reduce_dim(dim::Integer) = dim ≥ one(dim) || throw_bad_reduce_dim(dim)
@noinline throw_bad_reduce_dim(dim::Integer) =
    throw(ArgumentError("region dimension(s) must be ≥ 1, got $dim"))

# `reduce_axis` reduces a single dimension. Its input is either an array dimension or an
# index range. For type-stability in reduction operations, `reduce_axis` shall return an
# object of the same type.
reduce_axis(dim::Integer) = one(dim)
reduce_axis(rng::Base.OneTo{T}) where {T} = Base.OneTo{T}(1)
reduce_axis(rng::AbstractUnitRange) = (i = first(rng); oftype(rng, i:i))

# `reduce_shape1` and `reduce_axis1` are for reduction functions like `unique` that do not
# preserve axis offsets.
reduce_shape1(inds::Inds, dim::Integer) = reduce_shape(inds, dim, reduce_axis1)
reduce_axis1(rng::AbstractUnitRange) = oftype(rng, 1:1)
reduce_axis1(x) = reduce_axis(x)

# Yield a uniform array with given value and axes/size or a scalar.
_uniform(val, ::Tuple{}) = val
_uniform(val, inds::Tuple) = UniformArray(val, inds)

_sum_num_val(num::Integer, val) = num*val
_prod_num_val(num::Integer, val) = val^num
_count_num_val(num::Integer, val::Bool) = val ? num : zero(num)
_count_num_val(num::Integer, val) = _unexpected_non_boolean(val)
@noinline _unexpected_non_boolean(::Union{T,Type{T}}) where {T} =
    throw(ArgumentError("unexpected non-boolean ($T) result"))

# Result for an empty array for some base reduction functions.
_empty_result(::typeof(all)) = true
_empty_result(::typeof(any)) = false
@noinline _empty_result(f::Function) =
    throw(ArgumentError("reducing by `$(typename(f))` over an empty region is not allowed"))

"""
    StructuredArrays.value(A::AbstractUniformArray)

yields the value of the elements of the uniform array `A`.

"""
value(A::FastUniformArray{T,N,V}) where {T,N,V} = V
value(A::UniformArray) = getfield(A, :val)
value(A::MutableUniformArray) = getfield(A, :val)

setvalue!(A::MutableUniformArray{T}, x) where {T} = setfield!(A, :val, as(T, x))

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
    i:i == eachindex(IndexLinear(), A) || not_all_elements()
    setvalue!(A, x)
    return A
end

@inline function Base.setindex!(A::MutableUniformArray, x, rng::AbstractUnitRange{<:Integer})
    rng == eachindex(IndexLinear(), A) || not_all_elements()
    setvalue!(A, x)
    return A
end

@inline function Base.setindex!(A::MutableUniformArray, x, ::Colon)
    setvalue!(A, x)
    return A
end

@noinline not_all_elements() =
    error("all elements must be set at the same time")

@inline function subarray(A::AbstractUniformArray, I::Vararg{Any})
    # NOTE: Since we are calling the `subarray` method to produce an object
    # that is independent from `A`, it is not needed to unalias the result of
    # `to_indices` from `A` as is done in Julia base code (in `subarray.jl`) to
    # build views.
    J = Base.to_indices(A, I)
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

Base.show(io::IO, A::AbstractUniformArray) = show(io, MIME"text/plain"(), A)
function Base.show(io::IO, ::MIME"text/plain", A::AbstractUniformArray)
    print(io, parameterless(typeof(A)), "{", eltype(A), ",", ndims(A), "}(",
          value(A), ", ")
    print_axes(io, A; as_tuple=true)
    print(io, ")")
    nothing
end
