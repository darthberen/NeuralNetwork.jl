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
    @time NeuralNetwork.feedfoward(net, [10.0, 100.0])
    expected = [3.38075e-8, 0.95709, 5.87727e-6]
    @test actual ≈ expected atol=0.000001
end # begin (testset)
