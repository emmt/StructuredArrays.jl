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
