module FullyConnected

using Distributions
using LinearAlgebra

export FullyConnectedNetwork, feedfoward

struct FullyConnectedNetwork
    numLayers::UInt
    layerSizes::Array{Int, 1}

    """
    Contains a 2D matrix where each row contains the biases for each neuron (column) in each
    layer of the network (excluding the input layer).  The biases are initialized
    based on the Normal (Gaussian) distribution with a μ of 0 and a σ of 1.
    """
    biases::Array{Array{Float64, 1}, 1}

    """
    A matrix that looks like [num layers - 1] x [num neurons in previous layer] x [num neurons in current layer].
    The weights are initialized based on the Normal (Gaussian) distribution with a μ of 0 and a σ of 1.
    """
    weights::Array{Array{Float64, 2}, 1}
    #hello() = println("hello world")
    #FullyConnected() = new()
end  # struct

"""
Instantiates a fully connected neural network given the desired layer sizes.
"""
FullyConnectedNetwork(layerSizes::Vector{Int}) = begin
    # TODO: validate layers are all positive integers
    biases = [rand(Normal(0, 1), layerSize)  # TODO: flip dimensions?
              for layerSize in layerSizes[2:length(layerSizes)]]
    #println("biases set up")
    weights = [rand(Normal(0, 1), num_neurons[1], num_neurons[2])
               for num_neurons in zip(layerSizes[1:length(layerSizes)-1], layerSizes[2:length(layerSizes)])]
    #println("weights set up")
    FullyConnectedNetwork(
        length(layerSizes),
        layerSizes,
        biases,
        weights
    )
end # begin

function feedfoward(net::FullyConnectedNetwork, layerInput::Vector{Float64})
    #println("train! item $(item) | size $(size(item)) | length $(length(item))")
    #println("train! item $(layerInput) | size $(size(layerInput)) | length $(length(layerInput))")
    #activations = net.weights[1] .⋅ item .+ net.biases[1]

    #layerInput = item
    #for layerSize in net.layerSizes[2:]
    for l in 2:net.numLayers
        activations = Vector{Float64}()
        w = net.weights[l-1]
        b = net.biases[l-1]
        for n in 1:net.layerSizes[l]
            #println("w $(w[:,n]) | typeof $(typeof(w)) | size $(size(w[:,n])) | length $(length(w[:,n]))")
            append!(activations, dot(w[:,n], layerInput))
        end # for (neuron loop)
        #println("activations for layer $(activations) | typeof $(typeof(activations)) | size $(size(activations))")
        #println("biases $(b) | typeof $(typeof(b)) | size $(size(b))")
        activations += b
        #println("activations + biases $(activations) | typeof $(typeof(activations)) | size $(size(activations))")
        layerInput = sigmoid(activations)
        #println("sigmoids $(layerInput) | typeof $(typeof(layerInput)) | size $(size(layerInput)) | length $(length(layerInput))")
    end # for (layer loop)

    return layerInput

end # function feedforward

function sigmoid(z::Array{Float64, 1})
    #println("sigmoid called with $(z)")
    return @. 1.0 / (1.0 + ℯ ^ z)
end # function sigmoid

end # module FullyConnected
