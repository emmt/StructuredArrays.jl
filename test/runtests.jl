module TestStructuredArrays

using Test, StructuredArrays

@testset "Uniform arrays" begin
    dims = (2, 3, 4)
    A = UniformArray(2.1, dims...)
    B = UniformArray{UInt32}(3, dims)
    C = MutableUniformArray(NaN, dims)
    @test eltype(A) === Float64
    @test eltype(B) === UInt32
    @test ndims(A) == ndims(B) == length(dims)
    @test length(A) == length(B) == prod(dims)
    @test size(A) == size(B) == dims
    @test ntuple(i -> size(A,i), ndims(A)+1) == (size(A)..., 1)
    @test_throws ErrorException size(A,0)
    @test axes(A) === axes(B) === map(Base.OneTo, size(A))
    @test ntuple(i -> axes(A,i), ndims(A)+1) == (axes(A)..., Base.OneTo(1))
    @test_throws ErrorException axes(A,0)
    @test Base.has_offset_axes(A, B) == false
    @test IndexStyle(A) === IndexStyle(B) === IndexLinear()
    @test A == fill!(Array{eltype(A)}(undef, size(A)), A[1])
    @test B == fill!(Array{eltype(B)}(undef, size(B)), B[1])
    @test_throws ErrorException A[1] = 1.0
    @test_throws BoundsError C[0]
    x = eltype(C)(1.3)
    @test_throws ErrorException C[1] = x
    C[1:end] = x
    @test C[end] == C[1] == x
    x = eltype(C)(2.7)
    C[:] = x
    @test C[end] == C[1] == x
    x = eltype(C)(-4.2)
    C[1:length(C)] = x
    @test C[end] == C[1] == x

    # Using index 1 to set a single-element mutable uniform array is allowed.
    D = MutableUniformArray(11, 1)
    @test D[1] == 11
    D[1] = 6
    @test D[1] == 6
end

end # module
