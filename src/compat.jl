# `Returns` does not exist until Julia 1.7 and is not provided by `Compat`.
if !@isdefined(Returns)
    struct Returns{T}
        val::T
        Returns{T}(val) where {T} = new{T}(val)
    end
    Returns(val) = Returns{_stable_typeof(val)}(val)
    (f::Returns)(@nospecialize(args...); @nospecialize(kwds...)) = f.val
    _stable_typeof(x) = typeof(x)
    _stable_typeof(::Type{T}) where {T} = @isdefined(T) ? Type{T} : DataType
end

# `foreach` is very inefficient for Julia < 1.8. Since the beginning, there is a generic
# version of `foreach` for any iterable (in `abstractarray.jl`):
#
#     foreach(f) = (f(); nothing)
#     foreach(f, itr) = (for x in itr; f(x); end; nothing)
#     foreach(f, itrs...) = (for z in zip(itrs...); f(z...); end; nothing)
#
# In Julia 1.10 appears a version specialized for tuples and based on `foldl` (in `tuple.jl`):
#
#     foreach(f, itr::Tuple) = foldl((_, x) -> (f(x); nothing), itr, init=nothing)
#     foreach(f, itrs::Tuple...) = foldl((_, xs) -> (f(xs...); nothing), zip(itrs...), init=nothing)
#
if VERSION < v"1.8"
    # For tuples, the following implementation of `foreach` use unrolling (see `tuple.jl`
    # and `reduce.jl` in Julia base code).
    foreach(f, t::Tuple{}) = nothing
    @inline foreach(f, t::Tuple{Any})         = (f(t[1]); nothing)
    @inline foreach(f, t::Tuple{Any,Any})     = (f(t[1]); f(t[2]); nothing)
    @inline foreach(f, t::Tuple{Any,Any,Any}) = (f(t[1]); f(t[2]); f(t[3]); nothing)
    @inline foreach(f, t::Tuple)              = (f(t[1]); foreach(f, Base.tail(t)))
    # stop inlining after some number of arguments to avoid code blowup
    const Any32{N} = Tuple{Any,Any,Any,Any,Any,Any,Any,Any,
                           Any,Any,Any,Any,Any,Any,Any,Any,
                           Any,Any,Any,Any,Any,Any,Any,Any,
                           Any,Any,Any,Any,Any,Any,Any,Any,
                           Vararg{Any,N}}
    const All32{T,N} = Tuple{T,T,T,T,T,T,T,T,
                             T,T,T,T,T,T,T,T,
                             T,T,T,T,T,T,T,T,
                             T,T,T,T,T,T,T,T,
                             Vararg{T,N}}
    function foreach(f, t::Any32)
        for i in 1:length(t)
            f(t[i])
        end
        nothing
    end
    # For other arguments, call the base version.
    foreach(args...; kwds...) = Base.foreach(args...; kwds...)
end
