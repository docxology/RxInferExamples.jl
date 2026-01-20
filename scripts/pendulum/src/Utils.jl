using Dates

const RXINF_LOGFILE = joinpath(@__DIR__, "..", "rxinfer.log")

function rxlog(level::AbstractString, msg)
    ts = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    open(RXINF_LOGFILE, "a") do io
        println(io, "[$ts] $(uppercase(level)) - $msg")
    end
end

# Returns a variable argument function, that shifts a vector and put a new value at the end
function shift(vector, value)
    return (args...) -> begin 
        @inbounds for i in firstindex(vector):lastindex(vector)-1
            vector[i] = vector[i + 1]
        end
        vector[end] = value[]
        return vector
    end
end

function shift(vector, subject::AbstractSubject)
    return (_) -> begin 
        subscribe!(subject |> take(1), (value) -> shift(vector, value)())
        return vector
    end
end

export rxlog, shift
