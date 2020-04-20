module TestStructuredArrays

using Test, StructuredArrays

@testset "Uniform arrays" begin
    dims = (2, 3, 4)
    N = length(dims)
    vals = (Float64(2.1), UInt32(7), UInt16(11),
            Float32(-6.2), Float64(pi), Int16(-4))
    for k in 1:6
        x = vals[k]
        T = typeof(x)

        A = (k == 1 ? UniformArray(x, dims) :
             k == 2 ? UniformArray(x, dims...) :
             k == 3 ? UniformArray{T}(x, dims) :
             k == 4 ? UniformArray{T}(x, dims...) :
             k == 5 ? UniformArray{T,N}(x, dims) :
             k == 6 ? UniformArray{T,N}(x, dims...) : break)
        @test eltype(A) === T
        @test ndims(A) == N
        @test size(A) == dims
        @test ntuple(i -> size(A,i), ndims(A)+1) == (size(A)..., 1)
        @test_throws ErrorException size(A,0)
        @test axes(A) === map(Base.OneTo, size(A))
        @test ntuple(i -> axes(A,i), ndims(A)+1) == (axes(A)..., Base.OneTo(1))
        @test_throws ErrorException axes(A,0)
        @test Base.has_offset_axes(A, A) == false
        @test IndexStyle(A) === IndexLinear()
        @test A == fill!(Array{T}(undef, size(A)), A[1])
        @test_throws ErrorException A[1] = zero(T)
        @test_throws BoundsError A[0]

        B = (k == 1 ? MutableUniformArray(x, dims) :
             k == 2 ? MutableUniformArray(x, dims...) :
             k == 3 ? MutableUniformArray{T}(x, dims) :
             k == 4 ? MutableUniformArray{T}(x, dims...) :
             k == 5 ? MutableUniformArray{T,N}(x, dims) :
             k == 6 ? MutableUniformArray{T,N}(x, dims...) : break)
        @test eltype(B) === T
        @test ndims(B) == N
        @test size(B) == dims
        @test ntuple(i -> size(B,i), ndims(B)+1) == (size(B)..., 1)
        @test_throws ErrorException size(B,0)
        @test axes(B) === map(Base.OneTo, size(B))
        @test ntuple(i -> axes(B,i), ndims(B)+1) == (axes(B)..., Base.OneTo(1))
        @test_throws ErrorException axes(B,0)
        @test Base.has_offset_axes(B, B) == false
        @test IndexStyle(B) === IndexLinear()
        @test B == fill!(Array{T}(undef, size(B)), B[1])
        @test_throws ErrorException B[1] = zero(T)
        @test_throws BoundsError B[0]
        x = x
        x -= one(x)
        B[1:end] = x
        @test B[end] == B[1] == x
        x -= one(x)
        B[:] = x
        @test B[end] == B[1] == x
        x -= one(x)
        B[1:length(B)] = x
        @test B[end] == B[1] == x
    end

    # Using index 1 to set a single-element mutable uniform array is allowed.
    D = MutableUniformArray(11, 1)
    @test D[1] == 11
    D[1] = 6
    @test D[1] == 6
end

@testset "Structured arrays" begin
    dims = (3, 4)
    N = length(dims)
    f1(i) = -2i
    f2(i,j) = (-1 ≤ i - j ≤ 2)
    T1 = typeof(f1(1))
    T2 = typeof(f2(1,1))
    S1 = IndexLinear
    S2 = IndexCartesian
    for k in 1:16
        A = (k ==  1 ? StructuredArray(S2, f2, dims) :
             k ==  2 ? StructuredArray(S2(), f2, dims...) :
             k ==  3 ? StructuredArray{T2}(S2(), f2, dims) :
             k ==  4 ? StructuredArray{T2}(S2, f2, dims...) :
             k ==  5 ? StructuredArray{T2,N}(S2, f2, dims) :
             k ==  6 ? StructuredArray{T2,N}(S2(), f2, dims...) :
             k ==  7 ? StructuredArray{T2,N,S2}(f2, dims) :
             k ==  8 ? StructuredArray{T2,N,S2}(f2, dims...) :
             k ==  9 ? StructuredArray{T2,N,S2}(S2, f2, dims) :
             k == 10 ? StructuredArray{T2,N,S2}(S2(), f2, dims...) :
             k == 11 ? StructuredArray(f2, dims) :
             k == 12 ? StructuredArray(f2, dims...) :
             k == 13 ? StructuredArray{T2}(f2, dims) :
             k == 14 ? StructuredArray{T2}(f2, dims...) :
             k == 15 ? StructuredArray{T2,N}(f2, dims) :
             k == 16 ? StructuredArray{T2,N}(f2, dims...) : break)
        @test eltype(A) === T2
        @test ndims(A) == N
        @test size(A) == dims
        @test ntuple(i -> size(A,i), ndims(A)+1) == (size(A)..., 1)
        @test_throws ErrorException size(A,0)
        @test axes(A) === map(Base.OneTo, size(A))
        @test ntuple(i -> axes(A,i), ndims(A)+1) == (axes(A)..., Base.OneTo(1))
        @test_throws ErrorException axes(A,0)
        @test Base.has_offset_axes(A, A) == false
        @test IndexStyle(A) === IndexCartesian()
        @test A == [f2(i,j) for i in 1:dims[1], j in 1:dims[2]]
        #@test_throws ErrorException A[1,1] = zero(T)
        @test_throws BoundsError A[0,1]
    end
    for k in 1:16
        A = (k ==  1 ? StructuredArray(S1, f1, dims) :
             k ==  2 ? StructuredArray(S1(), f1, dims...) :
             k ==  3 ? StructuredArray{T1}(S1(), f1, dims) :
             k ==  4 ? StructuredArray{T1}(S1, f1, dims...) :
             k ==  5 ? StructuredArray{T1,N}(S1, f1, dims) :
             k ==  6 ? StructuredArray{T1,N}(S1(), f1, dims...) :
             k ==  7 ? StructuredArray{T1,N,S1}(f1, dims) :
             k ==  8 ? StructuredArray{T1,N,S1}(f1, dims...) :
             k ==  9 ? StructuredArray{T1,N,S1}(S1, f1, dims) :
             k == 10 ? StructuredArray{T1,N,S1}(S1(), f1, dims...) : break)
        I = LinearIndices(A)
        @test eltype(A) === T1
        @test ndims(A) == N
        @test size(A) == dims
        @test ntuple(i -> size(A,i), ndims(A)+1) == (size(A)..., 1)
        @test_throws ErrorException size(A,0)
        @test axes(A) === map(Base.OneTo, size(A))
        @test ntuple(i -> axes(A,i), ndims(A)+1) == (axes(A)..., Base.OneTo(1))
        @test_throws ErrorException axes(A,0)
        @test Base.has_offset_axes(A, A) == false
        @test IndexStyle(A) === IndexLinear()
        @test A == [f1(I[CartesianIndex(i,j)]) for i in 1:dims[1], j in 1:dims[2]]
        #@test_throws ErrorException A[1,1] = zero(T)
        @test_throws BoundsError A[0,1]
    end
end

end # module
