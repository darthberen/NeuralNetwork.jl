using Test
using NeuralNetwork
using Random

@testset "NeuralNetwork.fullyconnected" begin
    println("hello word from fc test")

    Random.seed!(101101)

    NeuralNetwork.greet()
    NeuralNetwork.testing()

    net = NeuralNetwork.FullyConnectedNetwork([2, 100, 3])
    #println(net.biases)
    #println("size: $(size(net.biases))")
    #println(net.weights)
    #println("weights: $(size(net.weights)) $(size(net.weights[1]))")

    @time actual = NeuralNetwork.feedfoward(net, [10.0, 100.0])
    println("result: ", actual)
    expected = [3.38075e-8, 0.95709, 5.87727e-6]
    @test actual ≈ expected atol=0.000001

    # actual timing run
    @time NeuralNetwork.feedfoward(net, [10.0, 100.0])

    trainingData = [
        [[1.0, 2.0], 1.0],
        [[3.0, 4.0], 2.0],
        [[5.0, 6.0], 3.0],
        [[7.0, 8.0], 1.0],
        [[9.0, 10.0], 2.0],
        [[11.0, 12.0], 2.0],
        [[13.0, 14.0], 3.0],
    ]
    NeuralNetwork.train!(net, trainingData, 1, 2, 0.4)#, Nothing)
end # begin (testset)
