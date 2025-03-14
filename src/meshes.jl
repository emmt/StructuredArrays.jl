"""
    org = origin(A)

yields the index at origin of coordinates in the Cartesian mesh or array `A`. The origin
is defined as the index where all coordinates of the nodes are zero. The returned value
may be `nothing` if the origin is at index `(0,0,...)`, a single array index if the origin
is at the same index for all dimensions, or a tuple of array indices. Indices may be
integers and/or reals for fractional indices.

Call `origin(Tuple, A)` to retrieve a `N`-dimensional (possibly fractional) index in all
cases.

""" function origin end

"""
    mesh = CartesianMesh{N}(step, origin = nothing)

yields a callable object that generates the coordinates of the nodes of a `N`-dimensional
Cartesian mesh with given `step` between consecutive nodes and `origin` of coordinates. If
any of `step` or `origin` is a `N`-tuple, the parameter `N` may be omitted. Calling the
mesh object with `i = (i1,i2,...)`, the `N` indices of a node, yields:

    mesh(i) = step .* i             # if `origin` is `nothing`
    mesh(i) = step .* (i .- origin) # else

Thanks to broadcasting rules, each of `step` and `origin` may be specified as a scalar to
assume that this parameters is the same for all dimensions, as a `N`-tuple otherwise.
`origin` may also be `nothing` (the default), to assume that the origin of the mesh is at
index `(0,0,...)`. In the implementation, the exact formula used to compute the
coordinates of the nodes is optimized for the different possible cases. As a consequence,
specifying `origin` as `0` or as a `N`-tuple of `0`s yields a mesh with the same
coordinates but computed with more overheads than with `origin = nothing`.

`step` may have units (possibly different for each dimension) but `origin` must be
unitless (integer or real). The values of the mesh parameters can be retrieved by calling
`step(mesh)` or [`origin(mesh)`](@ref StructuredArrays.Meshes.origin). Call
`step(Tuple,mesh)` or `origin(Tuple,mesh)` to retrieve a `N`-dimensional *step* and
*origin* in all cases. The values of `step` and `origin` stored by the mesh may be
converted at construction time to reduce the number of operations when computing
coordinates. This optimization assumes that all indices are specified as `Int`s but
fractional indices (i.e. reals) are also accepted.

A typical usage is to wrap a Cartesian mesh in a `StructuredArray` to build an abstract
array whose values are the coordinates of the nodes of a finite size Cartesian mesh. For
example:

    grid = StructuredArray(CartesianMesh{N}(step, origin), inds...)

with `inds...` the *shape* (indices and/or dimensions) of the mesh. This can also be done
by calling [`CartesianMeshArray`](@ref StructuredArrays.Meshes.CartesianMeshArray):

    grid = CartesianMeshArray(inds...; step, origin=nothing)

""" CartesianMesh

CartesianMesh(step::Number, origin::Union{Nothing,Real} = nothing) =
    CartesianMesh{1}(step, origin)

CartesianMesh(step::NTuple{N,Number}, origin::Union{Nothing,Real} = nothing) where {N} =
    CartesianMesh{N}(step, origin)

CartesianMesh(step::Union{Number,NTuple{N,Number}}, origin::NTuple{N,Real}) where {N} =
    CartesianMesh{N}(step, origin)

function CartesianMesh{N}(step::Union{Number,NTuple{N,Number}},
                          origin::Union{Nothing,Real,NTuple{N,Real}} = nothing) where {N}
    # The rules to promote the types of the mesh step and of the origin are:
    #
    # - Each coordinate is computed by the formula `(idx - org)*stp` with `idx` the node
    #   index (an `Int`), `org` the (fractional) index of the origin (assumed to be 0 if
    #   origin is `nothing`), and `stp` the step.
    #
    # - The step may have units, possibly different along all dimensions, but the origin
    #   is unitless (nothing, integer, or real).
    #
    # - The computed node coordinates all have the same numerical precision of bare real
    #   type `R`. This type may be integer but must be signed.
    #
    # - To optimize computations, for each node coordinate, at most one conversion shall
    #   be done (no conversion is needed if `R` is `Int`). This may be the conversion of
    #   `idx` to `R`, then the corresponding `org` must be of type `R`, or the conversion
    #   of the offset `idx - org` to `R`, then the corresponding `org` must be `nothing`
    #   or of type `Int`. To avoid further numerical conversions, all elements of the step
    #   have the same bare real type `R`.
    #
    # - As a simplification, we consider that any integer `org` can be converted to an
    #   `Int`.
    #
    R =  typeof(zero(mesh_offset_index_type(origin))*zero(mesh_step_real_type(step)))
    stp = convert_mesh_step(R, step)
    org = convert_mesh_origin(R, origin)
    return CartesianMesh{N,typeof(stp),typeof(org)}(stp, org)
end

mesh_offset_index_type(org::Nothing) = Int
mesh_offset_index_type(org::Integer) = Int
mesh_offset_index_type(org::Real   ) = promote_type(typeof(org), Int)
@inline mesh_offset_index_type(org::Tuple{Real,Vararg{Real}}) =
    promote_type(map(mesh_offset_index_type, org)...)

convert_mesh_origin(::Type{R}, org::Nothing) where {R<:Real} = nothing
convert_mesh_origin(::Type{R}, org::Integer) where {R<:Real} = as(Int, org)
convert_mesh_origin(::Type{R}, org::Real   ) where {R<:Real} = as(R, org)
@inline convert_mesh_origin(::Type{R}, org::Tuple{Real,Vararg{Real}}) where {R<:Real} =
    map(Converter(convert_mesh_origin, R), org)

mesh_step_real_type(stp::Number) = real_type(stp)
@inline mesh_step_real_type(stp::Tuple{Number,Vararg{Number}}) = real_type(stp...)

convert_mesh_step(::Type{R}, stp::Number) where {R<:Real} = convert_real_type(R, stp)
@inline convert_mesh_step(::Type{R}, stp::Tuple{Number,Vararg{Number}}) where {R<:Real} =
    map(convert_real_type(R), stp)

# Evaluators.
@inline (f::CartesianMesh{N})(I::CartesianIndex{N}) where {N} = f(Tuple(I))
@inline (f::CartesianMesh{N})(I::Vararg{Real,N}) where {N} = f(I)
@inline (f::CartesianMesh{N,S,Nothing})(I::NTuple{N,Real}) where {N,S} = I .* step(f)
@inline (f::CartesianMesh{N,S,O})(I::NTuple{N,Real}) where {N,S,O} = (I .- origin(f)) .* step(f)

"""
    A = CartesianMeshArray(inds...; step, origin=nothing)

builds an abstract array `A` representing a `N`-dimensional Cartesian mesh with shape
`inds...` and given `step` and `origin`. The syntax `A[i1,i2,...]` yields the coordinates
of the node at index `(i1,i2,...)`. Node indices may also be specified as a tuple of
`Int`s or as a `CartesianIndex`. The coordinates are lazily computed and the storage is
`O(1)`.

Also see [`CartesianMesh`](@ref StructuredArrays.Meshes.CartesianMesh) for a description
of arguments `step` and `origin`.

"""
CartesianMeshArray(inds::Union{Integer,AbstractUnitRange{<:Integer}}...; kwds...) =
    CartesianMeshArray(inds; kwds...)

function CartesianMeshArray(inds::NTuple{N,Union{Integer,AbstractUnitRange{<:Integer}}};
                            step, origin=nothing) where {N}
    return StructuredArray(IndexCartesian, CartesianMesh{N}(step, origin), inds)
end

# Extend some base methods.
Base.ndims(A::CartesianMesh) = ndims(typeof(A))
Base.ndims(::Type{<:CartesianMesh{N}}) where {N} = N

Base.eltype(A::CartesianMesh) = eltype(typeof(A))
Base.eltype(::Type{<:CartesianMesh{N,S}}) where {N,S<:Tuple} = S
@generated Base.eltype(::Type{<:CartesianMesh{N,S}}) where {N,S<:Number} = quote
    $(Tuple{ntuple(Returns(S), Val{N}())...})
end

# Accessors.
Base.step(A::CartesianMeshArray) = step(A.func)
Base.step(A::CartesianMesh) = getfield(A, :step)
Base.step(::Type{Tuple}, A::CartesianMeshArray) = step(Tuple, A.func)
Base.step(::Type{Tuple}, A::CartesianMesh{N,<:Number}) where {N} = ntuple(Returns(step(A)), Val(N))
Base.step(::Type{Tuple}, A::CartesianMesh{N,<:Tuple}) where {N} = step(A)

origin(A::CartesianMeshArray) = origin(A.func)
origin(A::CartesianMesh) = getfield(A, :origin)
origin(::Type{Tuple}, A::CartesianMeshArray) = origin(Tuple, A.func)
origin(::Type{Tuple}, A::CartesianMesh{N,S,Nothing}) where {N,S} = ntuple(Returns(0), Val(N))
origin(::Type{Tuple}, A::CartesianMesh{N,S,<:Real}) where {N,S} = ntuple(Returns(origin(A)), Val(N))
origin(::Type{Tuple}, A::CartesianMesh{N,S,<:Tuple}) where {N,S} = origin(A)

Base.show(io::IO, A::CartesianMeshArray) = show(io, MIME"text/plain"(), A)
function Base.show(io::IO, ::MIME"text/plain", A::CartesianMeshArray)
    print(io, "CartesianMeshArray{", eltype(A), ",", ndims(A), "}(")
    print_axes(io, A)
    print(io, "; step=", step(A), ", origin=", repr(origin(A)), ")")
    nothing
end

# Unary plus and minus.
Base.:(+)(A::CartesianMesh) = A
Base.:(-)(A::CartesianMesh{N}) where {N} = CartesianMesh{N}(map(-, step(A)), origin(A))

# Multiplication by a scalar.
Base.:(*)(A::CartesianMesh, x::Number) = x*A
Base.:(*)(x::Number, A::CartesianMesh) = CartesianMesh(x .* step(A), origin(A))

# Division by a scalar.
Base.:(\)(x::Number, A::CartesianMesh) = A/x
Base.:(/)(A::CartesianMesh, x::Number) = CartesianMesh(x .\ step(A), origin(A))

# Shift mesh.
Base.:(+)(x::NTuple{N,Number}, A::CartesianMesh{N}) where {N} = A + x
Base.:(+)(A::CartesianMesh{N}, x::NTuple{N,Number}) where {N} =
    CartesianMesh{N}(step(A), adjust_origin(A, -, map(Converter(convert, Real), x./step(A))))
Base.:(-)(x::NTuple{N,Number}, A::CartesianMesh{N}) where {N} = -(A - x)
Base.:(-)(A::CartesianMesh{N}, x::NTuple{N,Number}) where {N} =
    CartesianMesh{N}(step(A), adjust_origin(A, +, map(Converter(convert, Real), x./step(A))))

adjust_origin(A::CartesianMesh{N},  op::PlusOrMinus, adj::NTuple{N,Real}) where {N} = _adjust_origin(origin(A), op, adj)
_adjust_origin(org::Nothing,        op::PlusOrMinus, adj::NTuple{N,Real}) where {N} = map(op, adj)
_adjust_origin(org::Real,           op::PlusOrMinus, adj::NTuple{N,Real}) where {N} = map(Base.Fix1(op, org), adj)
_adjust_origin(org::NTuple{N,Real}, op::PlusOrMinus, adj::NTuple{N,Real}) where {N} = map(op, org, adj)

# Equality is tested in the sense that the meshes have the same node coordinates. Since
# neither step nor origin can contain `missing`, `isequal` and `==` are the same except
# for NaNs.
for f in (:(==), :isequal)
    @eval begin
        Base.$f(x::CartesianMesh{Nx}, y::CartesianMesh{Ny}) where {Nx,Ny} = false
        Base.$f(x::CartesianMesh{N}, y::CartesianMesh{N}) where {N} =
            x === y || (eq_step($f, x, y) && eq_origin($f, x, y))
    end
end

eq_step(f, x::CartesianMesh, y::CartesianMesh) = _eq_step(f, step(x), step(y))
_eq_step(f, x::Number, y::Number) = f(x, y)
_eq_step(f, x::Tuple,  y::Tuple ) = f(x, y)
_eq_step(f, x::Number, y::Tuple ) = all(Base.Fix1(f, x), y)
_eq_step(f, x::Tuple,  y::Number) = all(Base.Fix2(f, y), x)

eq_origin(f, x::CartesianMesh, y::CartesianMesh) = _eq_origin(f, origin(x), origin(y))
_eq_origin(f, x::Nothing, y::Nothing) = true
_eq_origin(f, x::Nothing, y::Real   ) = iszero(y)
_eq_origin(f, x::Nothing, y::Tuple  ) = all(iszero, y)
_eq_origin(f, x::Real,    y::Nothing) = iszero(x)
_eq_origin(f, x::Real,    y::Real   ) = f(x, y)
_eq_origin(f, x::Real,    y::Tuple  ) = all(Base.Fix1(f, x), y)
_eq_origin(f, x::Tuple,   y::Nothing) = all(iszero, x)
_eq_origin(f, x::Tuple,   y::Real   ) = all(Base.Fix2(f, y), x)
_eq_origin(f, x::Tuple,   y::Tuple  ) = f(x, y)
