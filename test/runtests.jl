module TestStructuredArrays

using Test, ArrayTools, StructuredArrays

@testset "Utilities        " begin
    dims = (Int8(2), Int16(3), Int32(4), Int64(5), 6)
    @test to_size(dims) === map(Int, dims)
    @test to_size(dims) === map(to_int, dims)
    @test to_size(dims[2]) === (to_int(dims[2]),)
end

# Error type when setindex! is not implemented. This depends on Julia version.
const NoSetIndexMethod = isdefined(Base, :CanonicalIndexError) ? CanonicalIndexError : ErrorException

@testset "Uniform arrays   " begin
    dims = (Int8(2), Int16(3), Int32(4))
    @test to_size(dims) === map(Int, dims)
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
        @test Base.has_offset_axes(A) == false
        @test IndexStyle(A) === IndexLinear()
        @test IndexStyle(typeof(A)) === IndexLinear()
        @test A == fill!(Array{T}(undef, size(A)), A[1])
        @test_throws NoSetIndexMethod A[1] = zero(T)
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
        @test Base.has_offset_axes(B) == false
        @test IndexStyle(B) === IndexLinear()
        @test IndexStyle(typeof(B)) === IndexLinear()
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

        C = (k == 1 ? FastUniformArray(x, dims) :
             k == 2 ? FastUniformArray(x, dims...) :
             k == 3 ? FastUniformArray{T}(x, dims) :
             k == 4 ? FastUniformArray{T}(x, dims...) :
             k == 5 ? FastUniformArray{T,N}(x, dims) :
             k == 6 ? FastUniformArray{T,N}(x, dims...) : break)
        @test eltype(C) === T
        @test ndims(C) == N
        @test size(C) == dims
        @test ntuple(i -> size(C,i), ndims(C)+1) == (size(C)..., 1)
        @test_throws ErrorException size(C,0)
        @test axes(C) === map(Base.OneTo, size(C))
        @test ntuple(i -> axes(C,i), ndims(C)+1) == (axes(C)..., Base.OneTo(1))
        @test_throws ErrorException axes(C,0)
        @test Base.has_offset_axes(C) == false
        @test IndexStyle(C) === IndexLinear()
        @test IndexStyle(typeof(C)) === IndexLinear()
        @test C == fill!(Array{T}(undef, size(C)), C[1])
        @test_throws NoSetIndexMethod C[1] = zero(T)
        @test_throws BoundsError C[0]
    end

    # All true and all false uniform arrays.
    let A = UniformArray(true, dims), B = FastUniformArray(true, dims)
        @test all(A) === true
        @test all(B) === true
        @test count(A) == length(A)
        @test count(B) == length(B)
    end
    let A = UniformArray(false, dims), B = FastUniformArray(false, dims)
        @test all(A) === false
        @test all(B) === false
        @test count(A) == 0
        @test count(B) == 0
    end

    # Array with zero dimensions.
    C = UniformArray{Int,0}(17)
    @test C[1] == 17
    @test Base.axes1(C) == 1:1
    @test Base.has_offset_axes(C) == false

    # Using index 1 to set a single-element mutable uniform array is allowed.
    D = MutableUniformArray(11, 1)
    @test D[1] == 11
    D[1] = 6
    @test D[1] == 6

    # Call constructors with illegal dimension.
    @test_throws ErrorException UniformArray{Int}(17, 1, -1)
    @test_throws ErrorException MutableUniformArray{Int}(17, 1, -1)
end

@testset "Structured arrays" begin
    dims = (Int8(3), Int16(4))
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
        @test Base.has_offset_axes(A) == false
        @test IndexStyle(A) === IndexCartesian()
        @test IndexStyle(typeof(A)) === IndexCartesian()
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
        @test Base.has_offset_axes(A) == false
        @test IndexStyle(A) === IndexLinear()
        @test IndexStyle(typeof(A)) === IndexLinear()
        @test A == [f1(I[CartesianIndex(i,j)])
                    for i in 1:dims[1], j in 1:dims[2]]
        #@test_throws ErrorException A[1,1] = zero(T)
        @test_throws BoundsError A[0,1]
    end

    # Call constructors with illegal dimension.
    @test_throws ErrorException StructuredArray{Bool}((i,j) -> i ≥ j, 1, -1)
end

end # module

nothing
