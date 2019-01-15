using Test
using NeuralNetwork

@testset "NeuralNetwork.fullyconnected" begin
    println("hello word from fc test")

    NeuralNetwork.greet()
    NeuralNetwork.testing()

    net = NeuralNetwork.FullyConnectedNetwork([2, 100, 3])
    println(net.biases)
    println("size: $(size(net.biases))")
    println(net.weights)
    println("weights: $(size(net.weights)) $(size(net.weights[1]))")

    @time NeuralNetwork.feedfoward(net, [10.0, 100.0])
    @time NeuralNetwork.feedfoward(net, [10.0, 100.0])
    @test 1 == 1
end # begin (testset)
