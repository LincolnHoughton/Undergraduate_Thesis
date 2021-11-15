#Call libraries
using LinearAlgebra
using SpecialFunctions
using Distributions
using Plots
plotly()
println()

#This function will read in the data and split it up into usable vectors
function parseData()
    #Read in the data and split it up by each newline
    f = read("/Users/lincolnhoughton/Documents/PhysicsCode/Research/quantum/AgPtdata.txt", String)
    fSplit = split(f, "\n")

    #Define the start location of each sample system
    systemNums = findall(x->occursin("str", x), fSplit)
    global numSystems = length(systemNums)

    #Define vector for each system's label
    systemLabel = Vector(undef, numSystems)

    #Define vector for lattice parameters (a)
    global latticeParameter = zeros(Float64, numSystems)
    lpDiff = 1

    #Define vectors for the basis vectors (a1, a2, a3)
    global a1Vector = zeros(Float64, (numSystems, 3))
    global a2Vector = zeros(Float64, (numSystems, 3))
    global a3Vector = zeros(Float64, (numSystems, 3))
    bvDiff = 2

    #Define vector for number of Ag and Pt atoms in each sample system
    global numAg = zeros(Int8, numSystems)
    global numPt = zeros(Int8, numSystems)
    numAgPtDiff = 5

    #Loop to find the label, lattice parameters, basis vectors, and number of Ag/Pt atoms in each sample system
    for (index, value) in enumerate(systemNums)
        #System Label 
        systemLabel[index] = fSplit[value]

        #Lattice Parameter
        latticeParameter[index] = parse(Float64, fSplit[value+lpDiff])

        #Basis Vectors
        # [Set, Col]
        a1Vector[index,:] = [parse(Float64, fSplit[value+bvDiff][1:23]), parse(Float64, fSplit[value+bvDiff][24:45]), parse(Float64, fSplit[value+bvDiff][46:end])]
        a2Vector[index,:] = [parse(Float64, fSplit[value+bvDiff+1][1:23]), parse(Float64, fSplit[value+bvDiff+1][24:45]), parse(Float64, fSplit[value+bvDiff+1][46:end])]
        a3Vector[index,:] = [parse(Float64, fSplit[value+bvDiff+2][1:23]), parse(Float64, fSplit[value+bvDiff+2][24:45]), parse(Float64, fSplit[value+bvDiff+2][46:end])]

        a1Vector[index,:] *= latticeParameter[index]
        a2Vector[index,:] *= latticeParameter[index]
        a3Vector[index,:] *= latticeParameter[index]

        #Num Ag and Pt atoms
        numAg[index] = parse(Int8, fSplit[value+numAgPtDiff][1:5])
        numPt[index] = parse(Int8, fSplit[value+numAgPtDiff][6:end])
    end

    ## AG ATOMS
    #Define vector for positions of Ag atoms in terms of a1, a2, a3
    global posAgDirect = Vector(undef, numSystems)
    posAgDiff = 7
    #Loop to define posAg to have as many 3D vectors as there are Ag atoms in each system
    for (idx, val) in enumerate(numAg)
        posAgDirect[idx] = zeros(Float64, (val, 3))
    end
    global posAg = deepcopy(posAgDirect) #Creates a duplicate zero vector for the final Ag positions 
    #Loop to assign the position of each Ag atom in the unit cell for every sample system
    for (sysIdx, sysVal) in enumerate(systemNums) #Loop through each sample system
        for i = 1:numAg[sysIdx] #Loop through each Ag atom in the unit cell 
            # [Set][Row, Col]
            #Assign the direct coordinates to each Ag atom
            posAgDirect[sysIdx][i,:] = [parse(Float64, fSplit[sysVal+posAgDiff+(i-1)][1:20]), parse(Float64, fSplit[sysVal+posAgDiff+(i-1)][21:40]), parse(Float64, fSplit[sysVal+posAgDiff+(i-1)][41:end])]
            #Convert direct coordinates (a1, a2, a3) into conventional coordinates (i, j, k)
            posAg[sysIdx][i,:] += posAgDirect[sysIdx][i,1] * a1Vector[sysIdx,:] + posAgDirect[sysIdx][i,2] * a2Vector[sysIdx,:] + posAgDirect[sysIdx][i,3] * a3Vector[sysIdx,:]
        end
    end

    ## PT ATOMS 
    #Define vector for positions of Pt atoms in terms of a1, a2, a3
    global posPtDirect = Vector(undef, numSystems)
    posPtDiff = Int8.(ones(numSystems)*7 + numAg) #Calculate the starting positions for the Pt position vectors
    #Loop to define posPt to have as many 3D vectors as there are Pt atoms in each system
    for (idx, val) in enumerate(numPt)
        posPtDirect[idx] = zeros(Float64, (val, 3))
    end
    global posPt = deepcopy(posPtDirect) #Creates a duplicate zero vector for the final Pt positions 
    #Loop to assign the position of each Pt atom in the unit cell for every sample system
    for (sysIdx, sysVal) in enumerate(systemNums) #Loop through each sample system
        for i = 1:numPt[sysIdx] #Loop through each Pt atom in the unit cell 
            # [Set][Row, Col]
            #Assign the direct coordinates to each Pt atom
            posPtDirect[sysIdx][i,:] = [parse(Float64, fSplit[sysVal+posPtDiff[sysIdx]+(i-1)][1:20]), parse(Float64, fSplit[sysVal+posPtDiff[sysIdx]+(i-1)][21:40]), parse(Float64, fSplit[sysVal+posPtDiff[sysIdx]+(i-1)][41:end])]
            #Convert direct coordinates (a1, a2, a3) into conventional coordinates (i, j, k)
            posPt[sysIdx][i,:] += posPtDirect[sysIdx][i,1] * a1Vector[sysIdx,:] + posPtDirect[sysIdx][i,2] * a2Vector[sysIdx,:] + posPtDirect[sysIdx][i,3] * a3Vector[sysIdx,:]
        end
    end

    ## ENERGY VALUES 
    #Define a vector with the unit cell energies for each system
    enValsText = filter.(x->occursin(".", x), split.(filter(x->occursin(" Energy", x), fSplit),":"))
    global energyValues = reduce(vcat, [parse.(Float64, x) for x in enValsText])
end


#This function will return all the atoms that effect the central Pt atom passed in
function findCutoffPt(samp, centerAtom, rFactor) #Takes in the sample number, which Pt atom of interest, and a radius 
    rBasis = findmax([norm(a1Vector[samp,:]), norm(a2Vector[samp,:]), norm(a3Vector[samp,:])]) #Find the largest basis vector magnitude
    r = rBasis[1] * rFactor #Make the radius of influence the largest basis vector magnitude multiplied by a constant

    #Define the limits of iterations in each direction, positive and negative
    a1Max = div(r, norm(a1Vector[samp,:]), RoundUp) #How many iterations in the a1 direction
    a2Max = div(r, norm(a2Vector[samp,:]), RoundUp) #How many iterations in the a2 direction
    a3Max = div(r, norm(a3Vector[samp,:]), RoundUp) #How many iterations in the a3 direction
    a1Min = -a1Max
    a2Min = -a2Max
    a3Min = -a3Max

    spread = Int64((a1Max - a1Min + 1) * (a2Max - a2Min + 1) * (a3Max - a3Min + 1)) #How many total unit cell iterations to consider?

    #Initialize a vector for positions of each unique Ag atom in the unit cell
    posAgIterations = Vector(undef, numAg[samp])
    for i = 1:numAg[samp]
        posAgIterations[i] = zeros(Float64, (spread, 3)) #Each sample will need to contain however many Ag atoms there are in the unit cell
    end

    #Initialize a vector for positions of each unique Pt atom in the unit cell
    posPtIterations = Vector(undef, numPt[samp])
    for i = 1:numPt[samp]
        posPtIterations[i] = zeros(Float64, (spread, 3)) #Each sample will need to contain however many Pt atoms there are in the unit cell
    end

    l = 1 #Overall index
    for i=a1Min:a1Max #Loop through a1 vectors
        for j=a2Min:a2Max #a2 vectors
            for k=a3Min:a3Max #a3 vectors 
                for m=1:numAg[samp] #loop through each unique atom in the unit cell 
                    #starting position of each unique Ag atom + the coordinates of the iterated version
                    #[unique atom][iteration num, coordinate]
                    posAgIterations[m][l,:] = posAg[samp][m,:] + i*a1Vector[samp,:] + j*a2Vector[samp,:] + k*a3Vector[samp,:]
                end
                for n=1:numPt[samp]
                    #starting position of each unique Pt atom + the coordinates of the iterated version
                    posPtIterations[n][l,:] = posPt[samp][n,:] + i*a1Vector[samp,:] + j*a2Vector[samp,:] + k*a3Vector[samp,:]
                end
                l+=1 #Increase the overall index 
            end
        end
    end

    #Define a 2D vector to contain the positions of Ag atoms that affect our initial Pt atom 
    posAgImportant = Vector{Vector{Float64}}()
    for i = 1:length(posAgIterations[1][:,1]) #Loop through the number of iterated unit cells
        for j = 1:numAg[samp] #Loop through each unique atom in the unit cell 
            #An if-statement to determine if the point we are considering is inside our sphere of interest and is NOT the center atom
            if norm(posAgIterations[j][i,:] - posPt[samp][centerAtom,:]) <= r && norm(posAgIterations[j][i,:] - posPt[samp][centerAtom,:]) != 0
                push!(posAgImportant, posAgIterations[j][i,:]) #If so, add it to our list of important points 
            end
        end
    end

    #Define a 2D vector to contain the positions of Pt atoms that affect our initial Pt atom 
    posPtImportant = Vector{Vector{Float64}}()
    #An identical nested for-loop as the above, but for Pt atoms
    for i = 1:length(posPtIterations[1][:,1])
        for j = 1:numPt[samp]
            if norm(posPtIterations[j][i,:] - posPt[samp][centerAtom,:]) <= r && norm(posPtIterations[j][i,:] - posPt[samp][centerAtom,:]) != 0
                push!(posPtImportant, posPtIterations[j][i,:])
            end
        end
    end

    #Only return the positions of the Ag and Pt atoms inside the sphere of influence for our center Pt atom
    return posAgImportant, posPtImportant
end


#This function operates the same as findCutoffPt, but works for the center atom as Ag
#See findCutoffPt for a line-by-line explanation in the comments
function findCutoffAg(samp, centerAtom, rFactor)
    rBasis = findmax([norm(a1Vector[samp,:]), norm(a2Vector[samp,:]), norm(a3Vector[samp,:])])
    r = rBasis[1] * rFactor

    a1Max = div(r, norm(a1Vector[samp,:]), RoundUp)
    a2Max = div(r, norm(a2Vector[samp,:]), RoundUp)
    a3Max = div(r, norm(a3Vector[samp,:]), RoundUp)
    a1Min = -a1Max
    a2Min = -a2Max
    a3Min = -a3Max

    spread = Int64((a1Max - a1Min + 1) * (a2Max - a2Min + 1) * (a3Max - a3Min + 1))

    posAgIterations = Vector(undef, numAg[samp])
    for i = 1:numAg[samp]
        posAgIterations[i] = zeros(Float64, (spread, 3))
    end

    posPtIterations = Vector(undef, numPt[samp])
    for i = 1:numPt[samp]
        posPtIterations[i] = zeros(Float64, (spread, 3))
    end

    l = 1
    for i=a1Min:a1Max
        for j=a2Min:a2Max
            for k=a3Min:a3Max
                for m=1:numAg[samp]
                    posAgIterations[m][l,:] = posAg[samp][m,:] + i*a1Vector[samp,:] + j*a2Vector[samp,:] + k*a3Vector[samp,:]
                end
                for n=1:numPt[samp]
                    posPtIterations[n][l,:] = posPt[samp][n,:] + i*a1Vector[samp,:] + j*a2Vector[samp,:] + k*a3Vector[samp,:]
                end
                l+=1
            end
        end
    end

    posAgImportant = Vector{Vector{Float64}}()
    for i = 1:length(posAgIterations[1][:,1])
        for j = 1:numAg[samp]
            if norm(posAgIterations[j][i,:] - posAg[samp][centerAtom,:]) <= r && norm(posAgIterations[j][i,:] - posAg[samp][centerAtom,:]) != 0
                push!(posAgImportant, posAgIterations[j][i,:])
            end
        end
    end

    posPtImportant = Vector{Vector{Float64}}()
    for i = 1:length(posPtIterations[1][:,1])
        for j = 1:numPt[samp]
            if norm(posPtIterations[j][i,:] - posAg[samp][centerAtom,:]) <= r && norm(posPtIterations[j][i,:] - posAg[samp][centerAtom,:]) != 0
                push!(posPtImportant, posPtIterations[j][i,:])
            end
        end
    end

    return posAgImportant, posPtImportant
end


#This function will calculate all the separation vectors between each atom in the unit cell and all its atoms of interest
function calculateSepVecs(rFactor=1.2) #pass in a radius
    #Initialize vectors for the important Ag and Pt positions
    global posAgFinal = Vector(undef, numSystems)
    global posPtFinal = Vector(undef, numSystems)
    for samp = 1:numSystems #Loop through each system in our file
        #Fill each sample with a vector long enough to contain the number of Ag/Pt atoms in the unit cell
        posAgFinal[samp] = Vector(undef, numAg[samp])
        posPtFinal[samp] = Vector(undef, numPt[samp])
        for i = 1:numAg[samp] #Loop through each Ag atom in the unit cell
            posAgFinal[samp][i] = findCutoffAg(samp, i, rFactor) #Call the function to return all the atoms of interest
        end
        for i = 1:numPt[samp] #Loop through each Pt atom in the unit cell
            posPtFinal[samp][i] = findCutoffPt(samp, i, rFactor) #Call the function to return all the atoms of interest
        end
    end #For reference: posAgFinal[sample number][unique Ag atom][Ag or Pt atom list (1 or 2)][atom of effect][x/y/z coordinate]

    #Initialize vectors to contain the separation vectors for each type of interaction (Ag-Ag, Ag-Pt, and Pt-Pt)
    global sepVecsAgAg = Vector(undef, numSystems)
    global sepVecsAgPt = Vector(undef, numSystems)
    global sepVecsPtPt = Vector(undef, numSystems)
    for samp = 1:numSystems #Loop through each system in our data set 
        #Somewhat complicated inline, double for-loops to fill each vector
        #i: Loop through each atom in the sample, j: loop through each important atom in i's sphere of influence
        #Take the norm of the difference between the original atom in the unit cell and each of it's atoms of influence
        sepVecsAgAg[samp] = [norm(posAgFinal[samp][i][1][j] - posAg[samp][i,:]) for i=1:numAg[samp] for j=1:length(posAgFinal[samp][i][1])]
        sepVecsPtPt[samp] = [norm(posPtFinal[samp][i][2][j] - posPt[samp][i,:]) for i=1:numPt[samp] for j=1:length(posPtFinal[samp][i][2])]
        #Here we need to calculate the norms for the center Ag and it's Pt atoms of influence as well as the center Pt and it's Ag atoms of influence
        sepVecsAgPt[samp] = reduce(vcat, [[norm(posAgFinal[samp][i][2][j] - posAg[samp][i,:]) for i=1:numAg[samp] for j=1:length(posAgFinal[samp][i][2])], [norm(posPtFinal[samp][i][1][j] - posPt[samp][i,:]) for i=1:numPt[samp] for j=1:length(posPtFinal[samp][i][1])]])
    end
end


#This function will help visualizing our sphere of interest by only plotting the atoms of interest for a particular Ag atom
#NOTE: Must call the parseData() and calculateSepVecs() functions prior to this function
function plotAgAtomSphere(samp, atomNum) #pass in the sample number and which atom of interest
    plot(getindex.(posAgFinal[samp][atomNum][1], 1), getindex.(posAgFinal[samp][atomNum][1], 2), getindex.(posAgFinal[samp][atomNum][1], 3), st=:scatter, markersize=3)
    plot!(getindex.(posAgFinal[samp][atomNum][2], 1), getindex.(posAgFinal[samp][atomNum][2], 2), getindex.(posAgFinal[samp][atomNum][2], 3), st=:scatter, markersize=3)
end


#This function will show us a histogram of the data's unit cell energies
#NOTE: Must call the parseData() function prior to this function
function plotSampleEnergies()
    histogram(energyValues, title = "Crystal Configuration Energy Histogram", xlabel = "Energy (eV)", ylabel = "Configurations", legend=false)
    display(current())
end


#This function will show us the primitive unit cell atoms and borders
#NOTE: Must call the parseData() function prior to this function
function plotPrimitiveCell(samp)
    xVals = [0, a1Vector[samp,1], a1Vector[samp,1]+a2Vector[samp,1], a2Vector[samp,1], 0]
    yVals = [0, a1Vector[samp,2], a1Vector[samp,2]+a2Vector[samp,2], a2Vector[samp,2], 0]
    zVals = [0, a1Vector[samp,3], a1Vector[samp,3]+a2Vector[samp,3], a2Vector[samp,3], 0]
    xVals2 = [0, a1Vector[samp,1], a1Vector[samp,1]+a3Vector[samp,1], a3Vector[samp,1], 0]
    yVals2 = [0, a1Vector[samp,2], a1Vector[samp,2]+a3Vector[samp,2], a3Vector[samp,2], 0]
    zVals2 = [0, a1Vector[samp,3], a1Vector[samp,3]+a3Vector[samp,3], a3Vector[samp,3], 0]
    xVals3 = [0, a3Vector[samp,1], a2Vector[samp,1]+a3Vector[samp,1], a2Vector[samp,1], 0]
    yVals3 = [0, a3Vector[samp,2], a2Vector[samp,2]+a3Vector[samp,2], a2Vector[samp,2], 0]
    zVals3 = [0, a3Vector[samp,3], a2Vector[samp,3]+a3Vector[samp,3], a2Vector[samp,3], 0]
    xVals4 = [a3Vector[samp,1], a1Vector[samp,1]+a3Vector[samp,1], a1Vector[samp,1]+a3Vector[samp,1]+a2Vector[samp,1], a1Vector[samp,1]+a2Vector[samp,1]]
    yVals4 = [a3Vector[samp,2], a1Vector[samp,2]+a3Vector[samp,2], a1Vector[samp,2]+a3Vector[samp,2]+a2Vector[samp,2], a1Vector[samp,2]+a2Vector[samp,2]]
    zVals4 = [a3Vector[samp,3], a1Vector[samp,3]+a3Vector[samp,3], a1Vector[samp,3]+a3Vector[samp,3]+a2Vector[samp,3], a1Vector[samp,3]+a2Vector[samp,3]]
    xVals5 = [a3Vector[samp,1]+a2Vector[samp,1], a1Vector[samp,1]+a2Vector[samp,1]+a3Vector[samp,1]]
    yVals5 = [a3Vector[samp,2]+a2Vector[samp,2], a1Vector[samp,2]+a2Vector[samp,2]+a3Vector[samp,2]]
    zVals5 = [a3Vector[samp,3]+a2Vector[samp,3], a1Vector[samp,3]+a2Vector[samp,3]+a3Vector[samp,3]]

    plot(posAg[samp][:,1], posAg[samp][:,2], posAg[samp][:,3], st=:scatter, markersize=5, label="Ag")
    plot!(posPt[samp][:,1], posPt[samp][:,2], posPt[samp][:,3], st=:scatter, markersize=5, label="Pt")
    plot!(xVals, yVals, zVals, linewidth=5, color="black")
    plot!(xVals2, yVals2, zVals2, linewidth=5, color="black")
    plot!(xVals3, yVals3, zVals3, linewidth=5, color="black")
    plot!(xVals4, yVals4, zVals4, linewidth=5, color="black")
    plot!(xVals5, yVals5, zVals5, linewidth=5, color="black")
    display(current())
end


#This function will initialize some variables for our model building
function initializeVars(var1, var2)
    global sampleNum = var1 #How many systems from the data set we will be using 
    global funcNum = var2 * 3 #Basis functions for each interaction type (Ag-Ag, Ag-Pt, and Pt-Pt)
    global numFunc = var2 #This is how many general basis functions we have
    global y = zeros(Float64, sampleNum) #Vector to contain the unit cell energies
    global A = zeros(Float64, sampleNum, funcNum) #Matrix to contain the basis functions and sample systems 
    global besselYZeros = [0.893577, 3.95768, 7.08605, 10.2223, 13.3611, 16.5009, 19.6413, 22.782, 25.923, 29.064, 32.2052, 35.3465, 38.4878, 41.6291, 44.7705, 47.9119, 51.0533, 54.1948, 57.3362, 60.4777, 63.6192, 66.7607, 69.9022, 73.0437, 76.1853, 79.3268, 82.4683, 85.6099, 88.7514, 91.8929, 95.0345, 98.176, 101.318, 104.459, 107.601, 110.742, 113.884, 117.025, 120.167, 123.309, 126.45, 129.592, 132.733, 135.875, 139.016, 142.158, 145.3, 148.441, 151.583, 154.724, 157.866, 161.007, 164.149, 167.291, 170.432, 173.574, 176.715, 179.857, 182.998, 186.14, 189.282, 192.423, 195.565, 198.706, 201.848, 204.99, 208.131, 211.273, 214.414, 217.556, 220.697, 223.839, 226.981, 230.122, 233.264, 236.405, 239.547, 242.689, 245.83, 248.972, 252.113, 255.255, 258.396, 261.538, 264.68, 267.821, 270.963, 274.104, 277.246, 280.388, 283.529, 286.671, 289.812, 292.954, 296.096, 299.237, 302.379, 305.52, 308.662, 311.803, 314.945, 318.087, 321.228, 324.37, 327.511, 330.653, 333.795, 336.936, 340.078, 343.219, 346.361, 349.503, 352.644, 355.786, 358.927, 362.069, 365.21, 368.352, 371.494, 374.635, 377.777, 380.918, 384.06, 387.202, 390.343, 393.485, 396.626, 399.768, 402.91, 406.051, 409.193, 412.334, 415.476, 418.618, 421.759, 424.901, 428.042, 431.184, 434.325, 437.467, 440.609, 443.75, 446.892, 450.033, 453.175, 456.317, 459.458, 462.6, 465.741, 468.883, 472.025, 475.166, 478.308, 481.449, 484.591, 487.733, 490.874, 494.016, 497.157, 500.299, 503.44, 506.582, 509.724, 512.865, 516.007, 519.148, 522.29, 525.432, 528.573, 531.715, 534.856, 537.998, 541.14, 544.281, 547.423, 550.564, 553.706, 556.848, 559.989, 563.131, 566.272, 569.414, 572.555, 575.697, 578.839, 581.98, 585.122, 588.263, 591.405, 594.547, 597.688, 600.83, 603.971, 607.113, 610.255, 613.396, 616.538, 619.679, 622.821, 625.963, 629.104, 632.246, 635.387, 638.529, 641.67, 644.812, 647.954, 651.095, 654.237, 657.378, 660.52, 663.662, 666.803, 669.945, 673.086, 676.228, 679.37, 682.511, 685.653, 688.794, 691.936, 695.078, 698.219, 701.361, 704.502, 707.644, 710.786, 713.927, 717.069, 720.21, 723.352, 726.493, 729.635, 732.777, 735.918, 739.06, 742.201, 745.343, 748.485, 751.626, 754.768, 757.909, 761.051, 764.193, 767.334, 770.476, 773.617, 776.759, 779.901, 783.042, 786.184, 789.325, 792.467, 795.608, 798.75, 801.892, 805.033, 808.175, 811.316, 814.458, 817.6, 820.741, 823.883, 827.024, 830.166, 833.308, 836.449, 839.591, 842.732, 845.874, 849.016, 852.157, 855.299, 858.44, 861.582, 864.724, 867.865, 871.007, 874.148, 877.29, 880.431, 883.573, 886.715, 889.856, 892.998, 896.139, 899.281, 902.423, 905.564, 908.706, 911.847, 914.989, 918.131, 921.272, 924.414, 927.555, 930.697, 933.839, 936.98, 940.122, 943.263, 946.405, 949.547, 952.688, 955.83, 958.971, 962.113, 965.254, 968.396, 971.538, 974.679, 977.821, 980.962, 984.104, 987.246, 990.387, 993.529, 996.67, 999.812, 1002.95, 1006.1, 1009.24, 1012.38, 1015.52, 1018.66, 1021.8, 1024.94, 1028.09, 1031.23, 1034.37, 1037.51, 1040.65, 1043.79, 1046.94, 1050.08, 1053.22, 1056.36, 1059.5, 1062.64, 1065.79, 1068.93, 1072.07, 1075.21, 1078.35, 1081.49, 1084.63, 1087.78, 1090.92, 1094.06, 1097.2, 1100.34, 1103.48, 1106.63, 1109.77, 1112.91, 1116.05, 1119.19, 1122.33, 1125.48, 1128.62, 1131.76, 1134.9, 1138.04, 1141.18, 1144.33, 1147.47, 1150.61, 1153.75, 1156.89, 1160.03, 1163.17, 1166.32, 1169.46, 1172.6, 1175.74, 1178.88, 1182.02, 1185.17, 1188.31, 1191.45, 1194.59, 1197.73, 1200.87, 1204.02, 1207.16, 1210.3, 1213.44, 1216.58, 1219.72, 1222.87, 1226.01, 1229.15, 1232.29, 1235.43, 1238.57, 1241.71, 1244.86, 1248., 1251.14, 1254.28, 1257.42, 1260.56, 1263.71, 1266.85, 1269.99, 1273.13, 1276.27, 1279.41, 1282.56, 1285.7, 1288.84, 1291.98, 1295.12, 1298.26, 1301.4, 1304.55, 1307.69, 1310.83, 1313.97, 1317.11, 1320.25, 1323.4, 1326.54, 1329.68, 1332.82, 1335.96, 1339.1, 1342.25, 1345.39, 1348.53, 1351.67, 1354.81, 1357.95, 1361.1, 1364.24, 1367.38, 1370.52, 1373.66, 1376.8, 1379.94, 1383.09, 1386.23, 1389.37, 1392.51, 1395.65, 1398.79, 1401.94, 1405.08, 1408.22, 1411.36, 1414.5, 1417.64, 1420.79, 1423.93, 1427.07, 1430.21, 1433.35, 1436.49, 1439.63, 1442.78, 1445.92, 1449.06, 1452.2, 1455.34, 1458.48, 1461.63, 1464.77, 1467.91, 1471.05, 1474.19, 1477.33, 1480.48, 1483.62, 1486.76, 1489.9, 1493.04, 1496.18, 1499.33, 1502.47, 1505.61, 1508.75, 1511.89, 1515.03, 1518.17, 1521.32, 1524.46, 1527.6, 1530.74, 1533.88, 1537.02, 1540.17, 1543.31, 1546.45, 1549.59, 1552.73, 1555.87, 1559.02, 1562.16, 1565.3, 1568.44]
end


#This function is how we calculate our basis vectors using bessel functions of the second kind
function basis(n, m, r)
    zero = besselYZeros[m]  #besselj_zero(n, m) #Find the m-th zero of the zeroth bessel function 
    return sum([bessely(n, zero * x/10) for x in r])
end


#This function builds our model using the Ab=y equation
function buildModel(var1=10, var2=10)
    initializeVars(var1, var2) #Call the function to initialize our variables
    for i=1:var1 #Loop through however many samples we would like to use
        y[i] = energyValues[i] #Fill in the y vector with the unit cell energy values
        for j=1:numFunc #Loop through the number of basis functions for our A matrix
            A[i,j] = basis(0, j, sepVecsAgAg[i]) #Fill in the basis functions for Ag-Ag interactions
            A[i,j+numFunc] = basis(0, j, sepVecsPtPt[i]) #Fill in the basis functions for Pt-Pt interactions
            A[i,j+2*numFunc] = basis(0, j, sepVecsAgPt[i]) #Fill in the basis functions for Ag-Pt interactions
        end
    end
    global b = A \ y #Calculate our vector of coefficients
end


#This function will use our model to predict the unit cell energies of new systems
function modelTest()
    global numTests = numSystems - sampleNum #Run as many tests as we have systems remaining
    rowA = zeros(Float64, numTests, funcNum) #Create a new "row" of our A matrix (really several rows)
    for i=1:numTests #Loop through as many tests we would like to complete
        for j=1:numFunc #Loop through all the basis functions 
            #Calculate each basis function for each different interaction type
            rowA[i,j] = basis(0, j, sepVecsAgAg[sampleNum+i]) #AgAg interaction 
            rowA[i,j+numFunc] = basis(0, j, sepVecsPtPt[sampleNum+i]) #PtPt interaction
            rowA[i,j+2*numFunc] = basis(0, j, sepVecsAgPt[sampleNum+i]) #AgPt interaction 
        end
    end
    global predictedEnergies = rowA * b #The model's system energy prediction
end


#This funciton will print out pertinent information regarding the test run
function printErrorInfo(r) #pass in the radius
    predictionErrors = abs.(energyValues[sampleNum+1:sampleNum+numTests] - predictedEnergies) #Error on each prediction
    largeErrorNum = 0 #Counter for the number of errors greater than reasonable
    largeErrorVal = 1e2 #Limit of reasonable error
    for i = length(predictedEnergies):-1:1 #Loop through each predicted energy
        if abs(predictionErrors[i]) >= largeErrorVal #Is this error greater than the the limit?
            deleteat!(predictionErrors, i) #Delete the value from the list of prediction energies 
            largeErrorNum += 1 #Increase the counter
        end
    end #Print important information regarding the test run in an easy to read format
    averageError = sum(predictionErrors) / length(predictionErrors) #Calculate the average error
    println("Number of samples:                   ", sampleNum)
    println("Number of basis functions:           ", numFunc)
    println("Radius r:                            ", r)
    println("Number of energy predictions:        ", numTests)
    println("Number of errors greater than ", largeErrorVal, ": ", largeErrorNum)
    println("Average error:                       ", averageError)
    println()
end


#This function will plot the predicted energies vs true energies. This will help visualize the accuracy of the model 
function plotModelTests()
    plot(predictedEnergies, energyValues[sampleNum+1:sampleNum+numTests], st=:scatter, label="Predicted Energies")
    plot!(energyValues, energyValues, label = "True Energies")
    plot!(xlabel = "Predicted Energies (eV)", ylabel = "Actual Energies (eV)", legend=false)
    plot!(xlims = [-150,0], ylims = [-150,0])
    display(current())
end


#This function will build a model and run tests given one set of parameters
function testOnce(r, s, f) #Pass in the radius, sampleNum, and numFunc
    parseData() #Call function to parse the data file
    println("Calculating separation vectors...") #Update the user 
    calculateSepVecs(r) #Call function to calculate the separation vectors 
    println("Building Model...") #Update the user 
    buildModel(s, f) #Call function to build our model 
    println("Testing Model...") #Update the user 
    modelTest() #Call function to test our model 
    printErrorInfo(r) #Call function to print information of interest 
    plotModelTests() #Call function to visualize the accuracy of our model 
end


#This function will build and test models while varying the radius of effect of a given particle
function testRadius(rStart=1.0, rChange=.2, rEnd=2.0) #pass in looping parameters 
    s = 600 #Number of samples
    f = 3 #Number of basis functions
    parseData()
    println("Building and testing models...")
    for r = rStart:rChange:rEnd #Loop through various radii 
        calculateSepVecs(r)
        buildModel(s, f)
        modelTest()
        printErrorInfo(r)
        #Save a plot????
    end
end


#This function will build and test models while varying the number of samples in our A matrix
function testSampleNum(sStart=200, sChange=100, sEnd=1000)
    r = 1.0 #Radius
    f = 3 #Number of basis functions
    parseData()
    println("Calculating separation vectors...")
    calculateSepVecs(r)
    println("Building and testing models...")
    for s = sStart:sChange:sEnd #Loop through the sample sizes
        buildModel(s, f)
        modelTest()
        printErrorInfo(r)
        ## save a plot??
    end
end


#This function will build and test models while varying the number of basis functions in our A matrix
function testFunctionNum(fStart=20, fChange=20, fEnd=100)
    r = 1.0 #Radius 
    s = 600 #Number of samples
    parseData()
    println("Calculating separation vectors...")
    calculateSepVecs(r)
    println("Building and testing models...")
    for f = fStart:fChange:fEnd #Loop through the number of basis functions
        buildModel(s, f)
        modelTest()
        printErrorInfo(r)
        ## save a plot??
    end
end


#This function will loop through various values of r, s, and f to provide data on all reasonable models
function testAllVars()
    parseData()
    for r = 1.0:0.25:2.25
        calculateSepVecs(r)
        for s = 200:200:1000
            for f = 20:20:100
                buildModel(s, f)
                modelTest()
                printErrorInfo(r)
            end
            for f = 200:100:500
                buildModel(s, f)
                modelTest()
                printErrorInfo(r)
            end
        end
    end

end


#This is the main part of the code that will run and report the time taken
@time begin
    parseData()
    testOnce(1.5, 800, 500) #(radius, sample size, basis functions) or (r, s, f)
    #testRadius(1.0, 0.2, 2.6) #Loop parameters (a, b, c)--> r = a:b:c
    #testSampleNum(200, 200, 1000) #Loop parameters (a, b, c)--> s = a:b:c
    #testFunctionNum(200, 100, 500) #Loop parameters (a, b, c)--> f = a:b:c
    #testAllVars()

    ###Extra functions --> See functions above for descriptions
    #parseData()
    #plotPrimitiveCell(65)#65
    #plotSampleEnergies()
    #plotAgAtomSphere(3, 1)
end


## NOTES/QUESTIONS 
#
# 2) Normal accuracy plot for some of them
#
##