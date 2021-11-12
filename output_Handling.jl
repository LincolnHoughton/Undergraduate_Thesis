using Plots
plotly()
println()

#This function will read in the data and split it up into usable vectors
function parseData() #Read in the data and split it up by each newline
    f = read("/Users/lincolnhoughton/Documents/PhysicsCode/Research/quantum/slurm.txt", String)
    fSplit = split(f, "\n")

    #Define the start location of each sample system
    outputNums = findall(x->occursin("samples", x), fSplit)
    global numOutputs = length(outputNums)

    global numSamps = zeros(Int64, numOutputs)
    global numFuncs = zeros(Int64, numOutputs)
    global radii = zeros(Float16, numOutputs)
    global numPredictions = zeros(Int64, numOutputs)
    global numErrors = zeros(Int64, numOutputs)
    global aveErrors = zeros(Float64, numOutputs)

    diffFuncs = 1
    diffRadii = 2
    diffPredicts = 3
    diffNumErrors = 4
    diffAveErrors = 5

    for (idx, val) in enumerate(outputNums)
        numSamps[idx] =       parse(Int64,   fSplit[val][37:end])
        numFuncs[idx] =       parse(Int64,   fSplit[val+diffFuncs][37:end])
        radii[idx] =          parse(Float16, fSplit[val+diffRadii][37:end])
        numPredictions[idx] = parse(Int64,   fSplit[val+diffPredicts][37:end])
        numErrors[idx] =      parse(Int64,   fSplit[val+diffNumErrors][37:end])
        aveErrors[idx] =      parse(Float64, fSplit[val+diffAveErrors][37:end])

    end
end


function plotErrorHeatmap(sampNum, errorList, scaleLim, errorTitle)
    indices = []
    for i = 1:numOutputs
        if numSamps[i] == sampNum
            push!(indices, i)
        end 
    end

    radiVals = Vector([1., 1.25, 1.5, 1.75, 2., 2.25])
    funcVals = Vector([20, 40, 60, 80, 100, 200, 300, 400, 500])
    data = zeros(Float64, (length(radiVals), length(funcVals)))
    for (i,x) in enumerate(radiVals)
        for (j,y) in enumerate(funcVals)
            for z in indices
                if radii[z] == x && numFuncs[z] == y
                    data[i,j] = errorList[z]
                end
            end
        end
    end

    xAxis = Vector(undef, length(radiVals))
    yAxis = Vector(undef, length(funcVals))
    for (i, v) in enumerate(radiVals)
        xAxis[i] = string(v)
    end
    for (i, v) in enumerate(funcVals)
        yAxis[i] = string(v)
    end

    #:bluesreds
    #:cividis
    #:vik
    heatmap(xAxis, yAxis, data', c = :bluesreds)
    heatmap!(xlabel="Cutoff Radius", ylabel="Number of basis functions",
             title="Training Set = $sampNum", colorbar_title="$errorTitle",
             clim=(0,scaleLim))
    display(current())
end

function bestModels(numErrorMax, aveErrorMax)
    for i in 1:numOutputs
        if numErrors[i] <= numErrorMax
            if aveErrors[i] <= aveErrorMax
                println("Testing Set: ", numSamps[i])
                println("Basis Funcs: ", numFuncs[i])
                println("Radius:      ", radii[i])
                println("Num Errors:  ", numErrors[i])
                println("Ave Error:   ", aveErrors[i])
                println()
            end
        end
    end
end

@time begin
    parseData()
    #for i = 200:200:1000
    #    plotErrorHeatmap(i, aveErrors, 10, "Average Errors")
    #    plotErrorHeatmap(i, numErrors, 20, "Large Errors")
    #end 
    bestModels(1, 2)
end
