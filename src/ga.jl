module GeneticAlgorithms2
import Base.isless

using GeneticAlgorithms
using Persa

mutable struct Param{T}
    name::AbstractString
    values::Array{T}
end

mutable struct Gene <: Entity
    values::Array{Int}
    params::Array{Param}
    fitness
end

function Gene(params::Array{Param})
    values = Array{Int}(length(params))

    for i=1:length(values)
        values[i] = rand([1:length(params[i].values)...])
    end
    println("-- Create a Gene")
    return Gene(values, params, +Inf)
end

function Gene(parents::Array{Gene})
    child = Gene(parents[1].params)

    println("--> Crossover")

    num_parents = length(parents)

    for i=1:length(child.values)
        parent = (rand(UInt) % num_parents) + 1
        child.values[i] = parents[parent].values[i]
    end

    return child
end

function mutate(gene::Gene)
    # let's go crazy and mutate 20% of the time
    rand(Float64) < 0.8 && return

    i = rand(UInt) % length(gene.values) + 1
    gene.values[i] = rand([1:length(params[i].values)])

    return nothing
end

function create_entity(num)
    global parameters

    return Gene(parameters)
end

function group_entities(pop)
    println("BEST: ", pop[1])

    if pop[1].fitness == 0
        return
    end

    # simple naive groupings that pair the best entitiy with every other
    for i in 1:length(pop)
        produce([1, i])
    end
end

function isless(lhs::Gene, rhs::Gene)
    abs(lhs.fitness) > abs(rhs.fitness)
end

function crossover(group)
    return Gene(group)
end

function fitness(ent)
    global estfun2
    global evalfun2
    println("-- Fitness")
    values = Array{Any}(length(ent.params))
    for i=1:length(values)
        values[i] = ent.params[i].values[ent.values[i]]
    end

    ent.fitness = evalfun2(estfun2(values...))
end

parameters = Array{Param}(0)
estfun2 = Array{Param}(0)
evalfun2 = Array{Param}(0)

function run(dataset::Persa.CFDatasetAbstract, estfun::Function, params::Tuple{AbstractString, Any}...)
    global parameters = Array{Param}(length(params))

    for i=1:length(params)
        parameters[i] = Param(params[i][1], params[i][2])
    end

    ds_train, ds_test = Persa.get(Persa.HoldOut(dataset, 0.9))

    GeneticAlgorithms2.estfun2(args...) = estfun(ds_train, args...)
    GeneticAlgorithms2.evalfun2(model) = Persa.aval(model, ds_test).mae

    runga(GeneticAlgorithms2; initial_pop_size = 2)
end
end
