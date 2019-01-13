#include('NeuralNetwork')
import NeuralNetwork

NeuralNetwork.greet()
NeuralNetwork.testing()

#FullyConnected.hello() = println("hello")
net = NeuralNetwork.FullyConnectedNetwork(convert(Array{UInt, 1}, [2, 4, 1]))
println(isimmutable(net))
#net.hello()

println("num layers: $(net.numLayers)")
#net.numLayers = 11
println("num layers: $(net.numLayers)")

println("layers: $(size(net.layerSizes))")
#net.hello()
