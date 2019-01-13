module FullyConnected

export FullyConnectedNetwork

struct FullyConnectedNetwork
    numLayers::UInt
    layerSizes::Array{UInt, 1}
    #hello() = println("hello world")
    #FullyConnected() = new()
end  # struct

FullyConnectedNetwork(layerSizes::Array{UInt, 1}) = begin
    FullyConnectedNetwork(length(layerSizes), layerSizes)
end # begin

end # module FullyConnected
