module GA
import Base.isless

using GeneticAlgorithms
using Persa

using Memento

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

    debug(logger, "Creating $gene")

    return gene
end

function Gene(parents::Array{Gene})
    child = Gene(parents[1].params)

    num_parents = length(parents)

    for i=1:num_parents
        debug(logger, "Parent $i: $(parents[i])")
    end

    for i=1:length(child.values)
        parent = (rand(UInt) % num_parents) + 1
        child.values[i] = parents[parent].values[i]
    end

    debug(logger, "Child: $child")

    return child
end

function Base.show(io::IO, gene::Gene)
    text = "Gene: fitness ($(gene.fitness)), values ("

    for i=1:length(gene.params)
        text = string(text, "($(gene.params[i].name), $(gene.params[i].values[gene.values[i]]))")
    end

    text = string(text, ")")

    show(io, text)
end

function mutate(gene::Gene)
    # let's go crazy and mutate mutation_rate% of the time
    rand(Float64) < (1 - g_mutation_ratio) && return

    i = rand(UInt) % length(gene.values) + 1
    gene.values[i] = rand([1:length(gene.params[i].values)...])

    return nothing
end

function create_entity(num)
    return Gene(parameters)
end

function group_entities(pop)
    global generation += 1
    global best_genes
    global dontchange
    global g_population

    if pop[1].fitness == 0
        return
    end

    if pop[1].fitness < best_genes[1].fitness
        dontchange = 0
    else
        dontchange += 1
    end

    push!(pop, best_genes...)

    sort!(pop; lt = (x, y) -> x.fitness < y.fitness)

    best_genes = pop[1:g_elite]

    info(logger, "BEST $generation/$g_generations: $(pop[1])")

    if generation >= g_generations
        return
    end

    if dontchange > g_patience
        return
    end

    # simple naive groupings that pair the best entitiy with every other
    for i in 1:g_population
        ranks = rankroulette(length(pop))
        parent1 = rand(ranks)
        deleteat!(ranks, findin(ranks, [parent1]))
        parent2 = rand(ranks)

        debug(logger, "Crossover: $parent1, $parent2")
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
    debug(logger, "Training a model")

    values = params(ent)

    ent.fitness = evalfun(estfun(values...))
end

function params(gene::Gene)
    values = Array{Any}(length(gene.params))

    for i=1:length(values)
        values[i] = gene.params[i].values[gene.values[i]]
    end

    return values
end

function run(dataset::Persa.CFDatasetAbstract,
                createmodel::Function,
                params::Tuple{AbstractString, Any}...;
                generations::Int = 10,
                population::Int = 3,
                mutation_ratio::Float64 = 0.2,
                elite::Int = 2,
                k::Float64 = 0.9,
                verbose::Bool = true,
                patience::Int = 3
                )

    if verbose
        global logger = Memento.config("debug"; fmt="[{date} | {level}]: {msg}")
    else
        global logger = Memento.config("info"; fmt="[{date} | {level}]: {msg}")
    end

    info(logger, "Starting Genetic Algorithm")

    # Global Parameters
    global g_generations = generations
    global g_mutation_ratio = mutation_ratio
    global g_elite = elite
    global g_population = population

    # Global Variables
    global generation = 0

    global g_patience = patience
    global dontchange = 0

    global parameters = Array{Param}(length(params))

    for i=1:length(params)
        parameters[i] = Param(params[i][1], params[i][2])
    end

    global best_genes = Array{Gene}(elite)

    for i=1:elite
        best_genes[i] = Gene(parameters)
    end

    ds_train, ds_val = Persa.get(Persa.HoldOut(dataset, k))

    global estfun(args...) = createmodel(ds_train, args...)
    global evalfun(model) = Persa.aval(model, ds_val).mae

    gamodel = runga(GA; initial_pop_size = population)

    values = Array{Any}(length(best_genes[1].params))

    for i=1:length(values)
        values[i] = best_genes[1].params[i].values[best_genes[1].values[i]]
    end

    return (createmodel(dataset, values...), values, best_genes[1].fitness)
end
end
