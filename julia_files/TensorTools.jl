using ITensors
using JSON


function tuple_to_complex(data)
    if typeof(data[1]) <: Float64
        return complex(data[1], data[2])
    else
        return map(tuple_to_complex, data)
    end
end


function flatten_and_convert_to_matrix(data)
    flattened_data = [x isa Vector ? flatten_and_convert_to_matrix(x) : x for x in data]
    matrix_data = hcat(flattened_data...)
    return matrix_data
end

function main(path)
    data = JSON.parsefile(path)
    max_index = maximum(data[end]["ind"])
    index_set = [Index(2, "i_$x") for x in 1:(max_index+1)]
    tensor_list = []
    svd_list = []
    shared_indices = Dict(i => 0 for i in 1:max_index)

    for d in data
        d["data"]
        tensor_indices = [index_set[i] for i in d["ind"]]
        tensor_data = tuple_to_complex(d["data"])
        tensor_data = flatten_and_convert_to_matrix(tensor_data)
        println(tensor_data)
        tensor = ITensor(tensor_data, tensor_indices)
        push!(tensor_list, tensor)

        for index in d["ind"]
            shared_indices[index] += 1
        end

        U,S,V = svd(tensor, tensor_indices, cutoff=1E-3)
        # S IS MISSING
        push!(svd_list, [U.tensor,V.tensor])

    end

    for tensor in tensor_list
        println(tensor)
    end

    shared_indices = [k for (k, v) in shared_indices if v > 1]

    for item in svd_list
        println(item)
    end

    JSON.open("./qc_out.json", "w") do io
        JSON.print(io, svd_list)
    end

end



main(ARGS[1])
