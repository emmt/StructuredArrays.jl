const ConcreteIndexStyle = Union{IndexLinear,IndexCartesian}

# Type alias for array size or axes.
const Inds{N} = NTuple{N,Union{Int,AbstractUnitRange{Int}}}

abstract type AbstractStructuredArray{T,N,S<:ConcreteIndexStyle,I<:Inds{N}} <: AbstractArray{T,N} end

abstract type AbstractUniformArray{T,N,I<:Inds{N}} <: AbstractStructuredArray{T,N,IndexLinear,I} end

# Singleton type used to avoid the overhead of `checked_indices` in constructors.
struct BareBuild end

struct UniformArray{T,N,I<:Inds{N}} <: AbstractUniformArray{T,N,I}
    inds::I
    val::T
    UniformArray{T}(val, inds::I) where {T,N,I<:Inds{N}} =
        new{T,N,I}(checked_indices(inds), val)
end

struct FastUniformArray{T,N,V,I<:Inds{N}} <: AbstractUniformArray{T,N,I}
    inds::I
    FastUniformArray{T}(val, inds::I) where {T,N,I<:Inds{N}} =
        new{T,N,as(T,val),I}(checked_indices(inds))
end

mutable struct MutableUniformArray{T,N,I<:Inds{N}} <: AbstractUniformArray{T,N,I}
    inds::I
    val::T
    MutableUniformArray{T}(val, inds::I) where {T,N,I<:Inds{N}} =
        new{T,N,I}(checked_indices(inds), val)
    MutableUniformArray{T}(::BareBuild, val, inds::I) where {T,N,I<:Inds{N}} =
        new{T,N,I}(inds, val)
end

struct StructuredArray{T,N,S,F,I<:Inds{N}} <: AbstractStructuredArray{T,N,S,I}
    inds::I
    func::F
    StructuredArray{T,N,S}(func::F, inds::I) where {T,N,S<:ConcreteIndexStyle,F,I<:Inds{N}} =
        new{T,N,S,F,I}(checked_indices(inds), func)
end

# Define aliases.
for (A, N) in ((:Vector, 1), (:Matrix, 2))
    @eval begin
        const $(Symbol("AbstractStructured",A)){T,S,I} = AbstractStructuredArray{T,$N,S,I}
        const $(Symbol("AbstractUniform",A)){T,I} = AbstractUniformArray{T,$N,I}
        const $(Symbol("MutableUniform",A)){T,I} = MutableUniformArray{T,$N,I}
        const $(Symbol("Uniform",A)){T,I} = UniformArray{T,$N,I}
        const $(Symbol("FastUniform",A)){T,V,I} = FastUniformArray{T,$N,V,I}
        const $(Symbol("Structured",A)){T,S,F,I} = StructuredArray{T,$N,S,F,I}
    end
end
