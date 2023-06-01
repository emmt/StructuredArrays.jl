module TestStructuredArrays

using Test, ArrayTools, StructuredArrays
using StructuredArrays: checksize, subarraysize

@testset "StructuredArrays package" begin

    @testset "Utilities" begin
        dims = (Int8(2), Int16(3), Int32(4), Int64(5), 6)

        @test to_size(dims) === map(Int, dims)
        @test to_size(dims) === map(to_int, dims)
        @test to_size(dims[2]) === (to_int(dims[2]),)

        @test checksize(()) == 1
        @test checksize((0x4, Int16(11))) == 44
        @test checksize((0x4, Int16(11), 0)) == 0
        @test_throws ArgumentError checksize((0x4, Int16(-11)))

        @test () === @inferred subarraysize((), (), ()...)
        @test () === @inferred subarraysize((), (2,), (1,)...)
        @test (3,4) === @inferred subarraysize((), (2,3,4), (1,:,:)...)
        @test (3,4) === @inferred subarraysize((), (2,3,4), (1,:,1:4)...)
        @test (2,4) === @inferred subarraysize((), (2,3,4), (:,2,1:4)...)
        @test (1,3) === @inferred subarraysize((), (2,3,4), (2:2,2,2:4)...)
        @test (2,3,4) === @inferred subarraysize((), (2,3,4), (:,:,:)...)
    end

    @testset "Uniform arrays ($K)" for K in (UniformArray,
                                             FastUniformArray,
                                             MutableUniformArray)
        dims = (Int8(2), Int16(3), Int32(4))
        @test to_size(dims) === map(Int, dims)
        N = length(dims)
        vals = (Float64(2.1), UInt32(7), UInt16(11),
                Float32(-6.2), Float64(pi), Int16(-4))
        for k in 1:6
            x = vals[k]
            T = typeof(x)
            A = k == 1 ? K(x, dims) :
                k == 2 ? K(x, dims...) :
                k == 3 ? K{T}(x, dims) :
                k == 4 ? K{T}(x, dims...) :
                k == 5 ? K{T,N}(x, dims) :
                k == 6 ? K{T,N}(x, dims...) : break
            @test eltype(A) === T
            @test ndims(A) == N
            @test size(A) == dims
            @test ntuple(i -> size(A,i), ndims(A)+1) == (size(A)..., 1)
            @test_throws BoundsError size(A,0)
            @test axes(A) === map(Base.OneTo, size(A))
            @test ntuple(i -> axes(A,i), ndims(A)+1) == (axes(A)..., Base.OneTo(1))
            @test_throws BoundsError axes(A,0)
            @test Base.has_offset_axes(A) == false
            @test IndexStyle(A) === IndexLinear()
            @test IndexStyle(typeof(A)) === IndexLinear()
            @test A == fill!(Array{T}(undef, size(A)), A[1])
            @test_throws Exception A[1] = zero(T)
            @test_throws BoundsError A[0]
            x -= one(x)
            if K <: MutableUniformArray
                A[:] = x
                @test all(isequal(x), A)
                x -= one(x)
                A[1:end] = x
                @test A[end] == A[1] == x
                x -= one(x)
                A[1:length(A)] = x
                @test A[end] == A[1] == x
            else
                @test_throws Exception A[:] = x
            end
        end

        # Check conversion of value.
        let A = K{Float32}(π, dims)
            @test eltype(A) === Float32
            @test first(A) === Float32(π)
        end

        # All true and all false uniform arrays.
        let A = K(true, dims), B = K(false, dims)
            @test all(A) === true
            @test count(A) == length(A)
            if A isa MutableUniformArray
                A[:] = false
                B = A
            else
                B = K(false, dims)
            end
            @test all(B) === false
            @test count(B) == 0
        end

        # Colon and range indexing of uniform arrays.
        V = K === UniformArray        ? UniformVector :
            K === FastUniformArray    ? FastUniformVector :
            K === MutableUniformArray ? MutableUniformVector : Nothing
        x = K === UniformArray        ? π :
            K === FastUniformArray    ? 3.1 :
            K === MutableUniformArray ? Int8(5) : nothing
        A = K(x, dims)
        @test A isa K{typeof(x)}
        @test A[:] isa V{eltype(A)}
        @test length(A[:]) == length(A)
        @test first(A[:]) === first(A)
        @test_throws BoundsError A[firstindex(A)-1:2:lastindex(A)]
        @test_throws BoundsError A[firstindex(A):2:lastindex(A)+1]
        r = firstindex(A):2:lastindex(A)
        @test A[r] isa V{eltype(A)}
        @test length(A[r]) == length(r)
        @test first(A[r]) == first(A)
        r = firstindex(A):firstindex(A)-1
        @test A[r] isa V{eltype(A)}
        @test length(A[r]) == 0

        # Sub-uniform array is a uniform array (or a scalar).
        let b = @inferred A[1,2,3]
            @test b isa eltype(A)
            @test b === first(A)
            @test b === A[0x1,2,3]
        end
        let B = @inferred A[:,:,:]
            @test B isa typeof(A)
            @test size(B) == size(A)
            @test last(B) === first(A)
        end
        let B = A[:,2,2:end]
            @test B isa K{eltype(A),ndims(A)-1}
            @test size(B) == (size(A,1), size(A,3) - 1)
            @test last(B) === first(A)
        end
        let B = A[1:end,2:end,3:end]
            @test B isa K{eltype(A),ndims(A)}
            @test size(B) == (size(A,1), size(A,2) - 1, size(A,3) - 2)
            @test last(B) === first(A)
        end

        # Check ambiguities.
        let B = A[:] # a uniform vector
            @test eltype(B) === eltype(A)
            @test ndims(B) == 1
            @test length(B) == length(A)
            @test all(isequal(first(B)), B)
            @test B[0x1] === B[1]
            C = @inferred B[:]
            @test typeof(C) === typeof(B)
            @test length(C) == length(B)
            @test all(isequal(first(C)), C)
        end

        # Array with zero dimensions.
        C = K{Int,0}(17)
        @test C[1] == 17
        @test Base.axes1(C) == 1:1
        @test Base.has_offset_axes(C) == false

        if K <: MutableUniformArray
            # Using index 1 to set a single-element mutable uniform array is allowed.
            D = K(11, 1)
            @test D[1] == 11
            D[1] = 6
            @test D[1] == 6
        end

        # Call constructors with illegal dimension.
        @test_throws ArgumentError K{Int}(17, 1, -1)
    end

    @testset "Structured arrays (StructuredArray)" begin
        dims = (Int8(3), Int16(4))
        N = length(dims)
        f1(i) = -2i
        f2(i,j) = (-1 ≤ i - j ≤ 2)
        T1 = typeof(f1(1))
        T2 = typeof(f2(1,1))
        S1 = IndexLinear
        S2 = IndexCartesian
        for k in 1:16
            A = k ==  1 ? StructuredArray(S2, f2, dims) :
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
                k == 16 ? StructuredArray{T2,N}(f2, dims...) : break
            @test eltype(A) === T2
            @test ndims(A) == N
            @test size(A) == dims
            @test ntuple(i -> size(A,i), ndims(A)+1) == (size(A)..., 1)
            @test_throws BoundsError size(A,0)
            @test axes(A) === map(Base.OneTo, size(A))
            @test ntuple(i -> axes(A,i), ndims(A)+1) == (axes(A)..., Base.OneTo(1))
            @test_throws BoundsError axes(A,0)
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
            @test_throws BoundsError size(A,0)
            @test axes(A) === map(Base.OneTo, size(A))
            @test ntuple(i -> axes(A,i), ndims(A)+1) == (axes(A)..., Base.OneTo(1))
            @test_throws BoundsError axes(A,0)
            @test Base.has_offset_axes(A) == false
            @test IndexStyle(A) === IndexLinear()
            @test IndexStyle(typeof(A)) === IndexLinear()
            @test A == [f1(I[CartesianIndex(i,j)])
                        for i in 1:dims[1], j in 1:dims[2]]
            #@test_throws ErrorException A[1,1] = zero(T)
            @test_throws BoundsError A[0,1]
        end

        # Call constructors with illegal dimension.
        @test_throws ArgumentError StructuredArray{Bool}((i,j) -> i ≥ j, 1, -1)
    end
end

end # module

nothing
