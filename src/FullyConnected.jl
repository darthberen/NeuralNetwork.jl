module FullyConnected

using Random
using Distributions
using LinearAlgebra

export FullyConnectedNetwork, feedfoward, train!

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
    # NOTE: Julia stores multi-dimensional arrays in column-major order
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

"""
neuron output is σ(weight ⋅ input + bias)
"""
function feedfoward(net::FullyConnectedNetwork, sample::Vector{Float64})
    activations = sample  # the activation output of the input layer is the sample itself

    for l in 2:net.numLayers  # iterate through each layer starting with the first hidden layer
        w = net.weights[l-1]
        b = net.biases[l-1]

        """
        NOTE: there are multiple ways to implement this
        Method 1: for loop
            tracker = Vector{Float64}()
            for n in 1:net.layerSizes[l]  # iterate through each neuron...
                append!(tracker, dot(w[:,n], activations)) # ...and dot product it with the activation vector (from previous layer)
            end # for (neuron loop)
            tracker += b
            activations = sigmoid(tracker)

        Method 2: list iteration
            activations = sigmoid([dot(w[:,n], activations) for n in 1:net.layerSizes[l]] + b)

        Method 3: broadcast and array/vector operations (most performant)
            Note: for some reason the dot function does not sum with multidimensional arrays so it's basically the same as .*
            activations = sigmoid(reshape(sum(dot.(w, activations), dims=1), length(b)) + b)
            activations = sigmoid(reshape(sum(w .* activations, dims=1), length(b)) + b)
        """
        activations = sigmoid(reshape(sum(w .* activations, dims=1), length(b)) + b)
    end # for (layer loop)

    return activations

end # function feedforward

function sigmoid(z::Array{Float64, 1})
    #println("sigmoid called with $(z)")
    return @. 1.0 / (1.0 + ℯ ^ z)
end # function sigmoid

function train!(net::FullyConnectedNetwork, trainingData,
                epochs::Int, miniBatchSize::Int, learningRate::Float64) #,
                #testData::Array{Array{Float64, 1}, 1})
    for epoch in 1:epochs
        shuffle!(trainingData)
        miniBatches = [trainingData[k:min(k+miniBatchSize-1, length(trainingData))]
                       for k in 1:miniBatchSize:length(trainingData)]
        for miniBatch in miniBatches
            println(miniBatch)
            updateMiniBatch!(miniBatch, learningRate)
        end # for (minibatch loop)
        #println(miniBatches)

        println("epoch $(epoch) completed")
    end # for (epoch loop)
end # function train!

function updateMiniBatch!(miniBatch, learningRate::Float64)

end # function updateMiniBatch

end # module FullyConnected
