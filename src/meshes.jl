module Meshes

export
    CartesianMesh,
    CartesianMeshArray,
    CartesianMeshMatrix,
    CartesianMeshVector,
    origin

using TypeUtils

import ..StructuredArrays: StructuredArray
using ..StructuredArrays
using ..StructuredArrays: Returns

"""
    using StructuredArrays.Meshes
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
    using StructuredArrays.Meshes
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

"""
struct CartesianMesh{N,S,O} <: Function
    step::S
    origin::O
end

CartesianMesh(step::Number, origin::Union{Nothing,Real} = nothing) =
    CartesianMesh{1}(step, origin)

CartesianMesh(step::NTuple{N,Number}, origin::Union{Nothing,Real} = nothing) where {N} =
    CartesianMesh{N}(step, origin)

CartesianMesh(step::Union{Number,NTuple{N,Number}}, origin::NTuple{N,Real}) where {N} =
    CartesianMesh{N}(step, origin)

function CartesianMesh{N}(step::Union{Number,NTuple{N,Number}},
                          origin::Union{Nothing,Real,NTuple{N,Real}} = nothing) where {N}
    R = real_type_of_coords(step, origin)
    stp = fix_step(R, step)
    org = fix_origin(R, origin)
    return CartesianMesh{N,typeof(stp),typeof(org)}(stp, org)
end

# Convert the mesh step to the real numeric type `R` that has been inferred by
# `real_type_of_coords`.
fix_step(::Type{R}, stp::Number) where {R<:Real} = convert_real_type(R, stp)
fix_step(::Type{R}, stp::NTuple{N,Number}) where {R<:Real,N} =
    map(Base.Fix1(fix_step, R), stp)

# Convert the mesh origin to limit the number of conversions when computing node
# coordinates with numeric type `R` that has been inferred by `real_type_of_coords`.
fix_origin(::Type{R}, org::Union{Nothing,Int}) where {R<:Real} = org
fix_origin(::Type{R}, org::Integer) where {R<:Real} = as(Int, org)
fix_origin(::Type{R}, org::Real) where {R<:Real} = as(R, org)
fix_origin(::Type{R}, org::NTuple{N,Real}) where {R<:Real,N} =
    map(Base.Fix1(fix_origin, R), org)

# Yield the real numeric type of coordinate(s).
real_type_of_coords(stp::Number, org::Union{Real,Nothing}) =
    # This version is called by all the others of when the step and origin are the same
    # for all dimensions.
    real_type_of_coords(real_type(stp), typeof(org))
real_type_of_coords(stp::Number, org::NTuple{N,Real}) where {N} =
    # This version is called when the step is the same for all dimensions but the origin
    # is different for each dimension.
    real_type(ntuple(d -> real_type_of_coords(stp, org[d]), Val(N))...)
real_type_of_coords(stp::NTuple{N,Number}, org::Union{Real,Nothing}) where {N} =
    # This version is called when the step is different for each dimension but the origin
    # is the same for all dimensions.
    real_type(ntuple(d -> real_type_of_coords(stp[d], org), Val(N))...)
real_type_of_coords(stp::NTuple{N,Number}, org::NTuple{N,Real}) where {N} =
    # This version is called when the step and the origin are different for each
    # dimension.
    real_type(ntuple(d -> real_type_of_coords(stp[d], org[d]), Val(N))...)


real_type_of_coords(::Type{R}, ::Type{S}) where {R<:Real,S<:Nothing} =
    typeof(one(R)*one(Int))
real_type_of_coords(::Type{R}, ::Type{S}) where {R<:Real,S<:Real} =
    typeof(one(R)*(one(Int) - zero(S)))

# Evaluators.
@inline (f::CartesianMesh{N})(I::CartesianIndex{N}) where {N} = f(Tuple(I))
@inline (f::CartesianMesh{N})(I::Vararg{Real,N}) where {N} = f(I)
@inline (f::CartesianMesh{N})(I::NTuple{N,Real}) where {N} = mesh_node(step(f), I, origin(f))

@inline function mesh_node(stp::Union{Number,NTuple{N,Number}}, I::NTuple{N,Real},
                           org::Nothing = nothing) where {N}
    return stp .* I
end

@inline function mesh_node(stp::Union{Number,NTuple{N,Number}}, I::NTuple{N,Real},
                           org::Union{Real,NTuple{N,Real}}) where {N}
    return stp .* (I .- org)
end

const CartesianMeshArray{T,N} = StructuredArray{T,N,IndexCartesian,<:CartesianMesh}
const CartesianMeshVector{T} = CartesianMeshArray{T,1}
const CartesianMeshMatrix{T} = CartesianMeshArray{T,2}

"""
    using StructuredArrays.Meshes
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
Base.eltype(::Type{<:CartesianMesh{N,S}}) where {N,S<:Number} =
    Tuple{ntuple(Returns(S), Val(N))...}

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
    join(io, size(A), "×")
    print(io, " CartesianMeshArray{", eltype(A), ",", ndims(A), "}(")
    print_axes(io, A)
    print(io, "; step=", step(A), ", origin=", origin(A), ")")
    nothing
end

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

end # module
