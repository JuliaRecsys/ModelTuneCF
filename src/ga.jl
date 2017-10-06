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

    gene = Gene(values, params, +Inf)

    println("-- Creating $gene")

    return gene
end

function Gene(parents::Array{Gene})
    child = Gene(parents[1].params)

    num_parents = length(parents)

    for i=1:num_parents
        println("-- Parent $i: $(parents[i])")
    end

    for i=1:length(child.values)
        parent = (rand(UInt) % num_parents) + 1
        child.values[i] = parents[parent].values[i]
    end

    println("Child: $child")

    return child
end

function Base.show(io::IO, gene::Gene)
    text = "Gene: fitness ($(gene.fitness)), values ("

    for i=1:length(gene.params)
        text = string(text, "($(gene.params[i].name), $(gene.params[i].values[gene.values[i]]))")
    end

    text = string(text, ")")

    print(io, text)
end

function mutate(gene::Gene)
    # let's go crazy and mutate 20% of the time
    rand(Float64) < 0.8 && return

    i = rand(UInt) % length(gene.values) + 1
    gene.values[i] = rand([1:length(gene.params[i].values)...])

    return nothing
end

function create_entity(num)
    return Gene(parameters)
end

function group_entities(pop)
    global generation
    global maxgen
    global best_fitness
    global dontchange_fitness
    global max_dontchange_fitness
    global best_values

    generation += 1

    if true
        println(best_gene)
        return
    end

    if pop[1].fitness == 0
        return
    end

    if pop[1].fitness < best_fitness
        best_fitness = pop[1].fitness
        dontchange_fitness = 0
        best_values = pop[1].values
    else
        dontchange_fitness += 1
        pop[1].values = best_values
        pop[1].fitness = best_fitness
    end

    print("BEST $generation/$maxgen: ", pop[1])

    if generation >= maxgen
        return
    end

    if dontchange_fitness > max_dontchange_fitness
        return
    end

    produce([1])

    # simple naive groupings that pair the best entitiy with every other
    for i in 1:(length(pop) - 1)
        ranks = rankroulette(length(pop))
        parent1 = rand(ranks)
        deleteat!(ranks, findin(ranks, [parent1]))
        parent2 = rand(ranks)
        println("-- Crossover: $parent1, $parent2")
        produce([parent1, parent2])
    end
end

function rankroulette(total::Int)
    elements = Array{Int}(convert(Int, (total .* (total + 1) ./ 2)))
    element = total
    j = 1
    for i=1:total
        for w=1:element
            elements[j] = element
            j += 1
        end
        element -= 1
    end
    return elements
end

function isless(lhs::Gene, rhs::Gene)
    abs(lhs.fitness) > abs(rhs.fitness)
end

function crossover(group)
    return Gene([group...])
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
best_values = Array{Int}(0)
generation = 0
maxgen = 2
best_fitness = Inf
dontchange_fitness = 0
max_dontchange_fitness = 3

function run(dataset::Persa.CFDatasetAbstract, estfun::Function, params::Tuple{AbstractString, Any}...; maxgenerations::Int = 2)
    global parameters = Array{Param}(length(params))
    global generation = 0
    global maxgen = maxgenerations
    global bestgen = Array{Gene}(maxgen)
    global best_fitness = Inf
    global dontchange_fitness = 0
    global max_dontchange_fitness = 3
    global best_values = Array{Int}(length(params))

    for i=1:length(params)
        parameters[i] = Param(params[i][1], params[i][2])
    end

    global best_gene = Gene(parameters)

    ds_train, ds_test = Persa.get(Persa.HoldOut(dataset, 0.9))

    GeneticAlgorithms2.estfun2(args...) = estfun(ds_train, args...)
    GeneticAlgorithms2.evalfun2(model) = Persa.aval(model, ds_test).mae

    runga(GeneticAlgorithms2; initial_pop_size = 3)
end
end
