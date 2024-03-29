module TestStructuredArrays

using Test, TypeUtils, StructuredArrays
using StructuredArrays: checked_size, parameterless, to_dim, to_size, to_axis, to_axes
using Base: OneTo

@testset "StructuredArrays package" begin

    @testset "Utilities" begin
        @test parameterless(Array) === Array
        @test parameterless(AbstractVector) === AbstractArray
        @test parameterless(DenseMatrix) === DenseArray
        @test parameterless(Array{Float32}) === Array
        @test parameterless(DenseArray{Float32,3}) === DenseArray
        @test parameterless(Matrix{Float64}) === Array

        @test checked_size(()) == ()
        @test checked_size((2,)) == (2,)
        @test checked_size((4, 0, 1,)) == (4, 0, 1,)
        @test_throws MethodError checked_size((0x4, Int16(11)))
        @test_throws ArgumentError checked_size((4, -1, 1,))

        dims = (Int8(2), Int16(3), Int32(4), Int64(5), 6)
        inds = (Int8(2):Int8(3), OneTo{Int8}(3), Int16(-2):Int16(1), 1:5, OneTo{Int16}(6))

        @test @inferred(to_size(())) === ()
        @test @inferred(to_size(dims)) === map(Int, dims)
        @test @inferred(to_size(dims)) === map(as(Int), dims)
        @test @inferred(to_size(dims)) === Dims(dims)
        @test @inferred(to_size(dims[2])) === (as(Int, dims[2]),)
        @test @inferred(to_size(Dims(dims))) === Dims(dims)
        @test @inferred(to_size(inds)) === Dims(dims)

        @test @inferred(to_axis(3)) === OneTo(3)
        @test @inferred(to_axis(0x5)) === OneTo(5)
        @test @inferred(to_axis(0:2)) === 0:2
        @test @inferred(to_axis(0x0:0x7)) === 0:7
        @test @inferred(to_axis(OneTo(0x8))) === OneTo(8)
        @test @inferred(to_axis(OneTo(5))) === OneTo(5)

        @test @inferred(to_axes(())) === ()
        @test @inferred(to_axes((2, 4, 7,))) === (OneTo(2), OneTo(4), OneTo(7),)
        @test @inferred(to_axes((0x2, 4, Int16(7),))) === (OneTo(2), OneTo(4), OneTo(7),)
        @test @inferred(to_axes((OneTo(2), 0:5, 0x1:0x9))) === (OneTo(2), 0:5, 1:9,)
        @test @inferred(to_axes((OneTo(2), 0:5, 1:9,))) === (OneTo(2), 0:5, 1:9,)
    end

    @testset "Uniform arrays ($K)" for K in (UniformArray,
                                             FastUniformArray,
                                             MutableUniformArray)
        dims = (Int8(2), Int16(3), Int32(4))
        inds = (Int8(0):Int8(1), OneTo{Int16}(3), Int16(-2):Int16(1),)
        @test to_size(dims) === map(Int, dims)
        @test to_size(inds) === to_size(dims)
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
            B = k == 1 ? K(x, inds) :
                k == 2 ? K(x, inds...) :
                k == 3 ? K{T}(x, inds) :
                k == 4 ? K{T}(x, inds...) :
                k == 5 ? K{T,N}(x, inds) :
                k == 6 ? K{T,N}(x, inds...) : break
            @test eltype(A) === T
            @test eltype(B) === T
            @test ndims(A) == N
            @test ndims(B) == N
            @test size(A) == dims
            @test size(B) == dims
            @test ntuple(i -> size(A,i), ndims(A)+1) == (size(A)..., 1)
            @test ntuple(i -> size(B,i), ndims(B)+1) == (size(B)..., 1)
            @test_throws BoundsError size(A,0)
            @test_throws BoundsError size(B,0)
            @test axes(A) === map(Base.OneTo{Int}, dims)
            @test axes(B) === map(x -> convert_eltype(Int, x), inds)
            @test ntuple(i -> axes(A,i), ndims(A)+1) == (axes(A)..., Base.OneTo(1))
            @test ntuple(i -> axes(B,i), ndims(B)+1) == (axes(B)..., Base.OneTo(1))
            @test_throws BoundsError axes(A,0)
            @test_throws BoundsError axes(B,0)
            @test Base.has_offset_axes(A) == false
            @test Base.has_offset_axes(B) == true
            @test IndexStyle(A) === IndexLinear()
            @test IndexStyle(B) === IndexLinear()
            @test IndexStyle(typeof(A)) === IndexLinear()
            @test IndexStyle(typeof(B)) === IndexLinear()
            @test A == fill!(Array{T}(undef, size(A)), A[1])
            @test_throws Exception A[1] = zero(T)
            @test_throws Exception B[1] = zero(T)
            @test_throws BoundsError A[0]
            @test_throws BoundsError B[0]
            x -= one(x)
            if K <: MutableUniformArray
                A[:] = x
                @test all(isequal(x), A)
                B[:] = x
                @test all(isequal(x), B)
                x -= one(x)
                A[1:end] = x
                @test A[end] == A[1] == x
                B[1:end] = x
                @test B[end] == B[1] == x
                x -= one(x)
                A[1:length(A)] = x
                @test A[end] == A[1] == x
                B[firstindex(B):lastindex(B)] = x
                @test B[end] == B[firstindex(B)] == x
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
        @test A[:] isa Array{eltype(A)}
        @test length(A[:]) == length(A)
        @test first(A[:]) === first(A)
        @test_throws BoundsError A[firstindex(A)-1:2:lastindex(A)]
        @test_throws BoundsError A[firstindex(A):2:lastindex(A)+1]
        r = firstindex(A):2:lastindex(A)
        @test A[r] isa Array{eltype(A)}
        @test length(A[r]) == length(r)
        @test first(A[r]) == first(A)
        r = firstindex(A):firstindex(A)-1
        @test A[r] isa Array{eltype(A)}
        @test length(A[r]) == 0

        # Sub-indexing a uniform array yields uniform array (or a scalar).
        let B = Array(A)
            @test B isa Array{eltype(A),ndims(A)}
            funcs = A isa MutableUniformArray ? (getindex,) : (getindex, view)
            @testset "sub-indexing ($f)" for f in funcs
                R = f === getindex ? Array : K
                let I = (1,2,3), X = f(A, I...), Y = f(B, I...)
                    # should yield a scalar when sub-indexing
                    @test X isa (f === getindex ? eltype(A) : K{eltype(A),ndims(Y)})
                    @test X == Y
                end
                let I = (1,2,3,1,1), X = f(A, I...), Y = f(B, I...)
                    # should yield a scalar when sub-indexing
                    @test X isa (f === getindex ? eltype(A) : K{eltype(A),ndims(Y)})
                    @test X == Y
                end
                # result is an array
                let I = (1,2:2,3,1), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(Y)}
                    @test X == Y
                end
                let I = (1,2,3,1,1:1), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(Y)}
                    @test X == Y
                end
                let I = (1,2:2,3,1,1:1), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(Y)}
                    @test X == Y
                end
                let I = (:,), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(Y)}
                    @test X == Y
                end
                let I = (:,:,:), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(Y)}
                    @test axes(X) === axes(A)
                    @test X == Y
                end
                let I = ntuple(i -> i:size(A,i), ndims(A)), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(Y)}
                    @test size(X) == map(length, I)
                    @test X == Y
                end
                let I = (:,2,2:4), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(A)-1}
                    @test X == Y
                end
                let I = (:,[true, false, true],:), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(Y)}
                    @test X == Y
                end
                if v"1.6" ≤ VERSION ≤ v"1.10"
                    let I = (:,[true false true],:), X = f(A, I...), Y = f(B, I...)
                        @test X isa R{eltype(A),ndims(Y)}
                        @test X == Y
                    end
                end
                let I = (:,[2, 1, 2, 3],:), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(Y)}
                    @test X == Y
                end
                let I = (falses(size(A)),), X = f(A, I...), Y = f(B, I...)
                    @test X isa R{eltype(A),ndims(Y)}
                    @test X == Y
                end
            end
        end

        # Check ambiguities.
        let B = A[:] # a uniform vector
            @test eltype(B) === eltype(A)
            @test ndims(B) == 1
            @test length(B) == length(A)
            @test all(isequal(first(B)), B)
            @test B[0x1] === B[1]
            C = B[:]
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

    val_list = (true, false, 0, 3, 3.0)
    dims_list = (:, 1, 2, 3, (2,), [3,3], (2,1), [3,2],
                 # NOTE: Specifying dimensions larger than number of dimensions
                 #       is only supported in Julia ≥ 1.8
                 VERSION ≥ v"1.8" ? [1,3,4] : [1,3],
                 1:3)
    @testset "Optimized methods for uniform arrays (val=$val)" for val in val_list
        A = @inferred(UniformArray(val, (2, 3, 4)))
        B = Array(A)
        @test typeof(B) === Array{eltype(A),ndims(A)}
        @testset "... with `dims=$dims`" for dims in dims_list
            f(x) = x > zero(x)
            if dims isa Colon
                @test all(f,  B) == @inferred(all(f,  A))
                @test any(f,  B) == @inferred(any(f,  A))
                @test extrema(B) == @inferred(extrema(A))
                @test findmax(B) == @inferred(findmax(A))
                @test findmin(B) == @inferred(findmin(A))
                @test maximum(B) == @inferred(maximum(A))
                @test minimum(B) == @inferred(minimum(A))
                @test prod(   B) == @inferred(prod(   A))
                @test sum(    B) == @inferred(sum(    A))
                @test reverse(B) == @inferred(reverse(A))
                @test unique( B) == @inferred(unique( A))
            end
            @test all(f,  B; dims=dims) == all(f,  A; dims=dims)
            @test any(f,  B; dims=dims) == any(f,  A; dims=dims)
            @test extrema(B; dims=dims) == extrema(A; dims=dims)
            @test findmax(B; dims=dims) == findmax(A; dims=dims)
            @test findmin(B; dims=dims) == findmin(A; dims=dims)
            @test maximum(B; dims=dims) == maximum(A; dims=dims)
            @test minimum(B; dims=dims) == minimum(A; dims=dims)
            @test prod(   B; dims=dims) == prod(   A; dims=dims)
            @test sum(    B; dims=dims) == sum(    A; dims=dims)

            if dims isa Integer
                @test reverse(B; dims=dims) == @inferred(reverse(A; dims=dims))
                @test unique( B; dims=dims) == @inferred(unique( A; dims=dims))
            end
        end
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
        for A in (StructuredArray(S2, f2, dims),
                  StructuredArray(S2(), f2, dims...),
                  StructuredArray{T2}(S2(), f2, dims),
                  StructuredArray{T2}(S2, f2, dims...),
                  StructuredArray{T2,N}(S2, f2, to_size(dims)),
                  StructuredArray{T2,N}(S2(), f2, dims...),
                  StructuredArray{T2,N,S2}(f2, dims),
                  StructuredArray{T2,N,S2}(f2, to_size(dims)...),
                  StructuredArray{T2,N,S2}(S2, f2, dims),
                  StructuredArray{T2,N,S2}(S2(), f2, dims...),
                  StructuredArray(f2, dims),
                  StructuredArray(f2, dims...),
                  StructuredArray{T2}(f2, dims),
                  StructuredArray{T2}(f2, dims...),
                  StructuredArray{T2,N}(f2, dims),
                  StructuredArray{T2,N}(f2, dims...),)
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
        for A in (StructuredArray(S1, f1, dims),
                  StructuredArray(S1(), f1, dims...),
                  StructuredArray{T1}(S1(), f1, dims),
                  StructuredArray{T1}(S1, f1, dims...),
                  StructuredArray{T1,N}(S1, f1, dims),
                  StructuredArray{T1,N}(S1(), f1, dims...),
                  StructuredArray{T1,N,S1}(f1, dims),
                  StructuredArray{T1,N,S1}(f1, dims...),
                  StructuredArray{T1,N,S1}(S1, f1, dims),
                  StructuredArray{T1,N,S1}(S1(), f1, dims...),)
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
