using Test

testlist = [
    "fullyconnected"
]

@testset "NeuralNetwork" begin
    for f in testlist
        println("Running $(f) tests...")
        include("$(f).jl")
    end # for
end # begin (testset)
# println("hello world from runtests.jl")

#@test 1 == 1
