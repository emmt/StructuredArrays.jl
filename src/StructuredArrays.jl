"""

Package `StructuredArrays` implements arrays whose elements have values that only depend
on the indices. Examples of such arrays are uniform arrays with constant values,
structured arrays with Boolean values indicating whether an entry is significant.

"""
module StructuredArrays

export
    AbstractStructuredArray,
    AbstractStructuredMatrix,
    AbstractStructuredVector,
    AbstractUniformArray,
    AbstractUniformMatrix,
    AbstractUniformVector,
    CartesianMesh,
    CartesianMeshArray,
    CartesianMeshMatrix,
    CartesianMeshVector,
    FastUniformArray,
    FastUniformMatrix,
    FastUniformVector,
    MutableUniformArray,
    MutableUniformMatrix,
    MutableUniformVector,
    StructuredArray,
    StructuredMatrix,
    StructuredVector,
    UniformArray,
    UniformMatrix,
    UniformVector,
    origin

using OffsetArrays
using TypeUtils
using TypeUtils: Converter, @public

import Base: @propagate_inbounds, front, tail, Fix1, Fix2

include("compat.jl")
include("types.jl")
include("common.jl")
include("structured.jl")
include("uniform.jl")
include("meshes.jl")

end # module
