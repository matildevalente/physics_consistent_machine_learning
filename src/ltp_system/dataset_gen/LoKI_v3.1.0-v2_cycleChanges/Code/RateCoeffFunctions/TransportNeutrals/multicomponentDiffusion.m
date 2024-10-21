function [rateCoeff, dependent] = multicomponentDiffusion(~, densitiesAll, ~, reactionArray, reactionID, ...
  stateArray, workCond, ~, ~)
  % multicomponentDiffusion evaluates the transport rate coefficient of a particular species interacting with the 
  % wall, assuming a density profile that is zero at the walls (neglecting the wall reaction time) and assuming equal
  % probability for each destruction channel.
  % 
  % For more info check documentation.

  persistent neutralSpeciesIDs;
  persistent reducedMass;
  persistent sigma;
  persistent epsilon;
  persistent numberOfChannelsArray;
  persistent charDiffLengthSquared;
  persistent dependentInfo;
  
  % local save of the ID of the reactant species
  reactantID = reactionArray(reactionID).reactantArray(1).ID;
  
  % --- performance sensitive calculations ---
  
  % calculations performed once per simulation 
  if isempty(dependentInfo)
    % evaluate the first zero of the zero order bessel function
    firstBesselZero = fzero(@(x) besselj(0,x), [2.4 2.5]);
    % evaluate geometrical parameters for a cylinder (including infinitely long cylinder and slab limiting cases)
    if workCond.chamberLength == 0        % infinitely long cylinder
      charDiffLengthSquared = (workCond.chamberRadius/firstBesselZero)^2;
    elseif workCond.chamberRadius == 0    % infinitely wide cylinder (slab)
      charDiffLengthSquared = (workCond.chamberLength/pi)^2;
    else                                  % finite cylinder
      charDiffLengthSquared = 1/((firstBesselZero/workCond.chamberRadius)^2+(pi/workCond.chamberLength)^2);
    end
    % define dependencies of the rate coefficient
    dependentInfo = struct('onTime', false, 'onDensities', true, 'onGasTemperature', true, 'onElectronKinetics', false);
    % find IDs of volume neutral species that are going to be taken into account for the multicomponent transport
    % (childless neutral volume species)
    neutralSpeciesIDs = [];
    for i = 1:length(stateArray)
      if isempty(stateArray(i).ionCharg) && isempty(stateArray(i).childArray) && stateArray(i).isVolumeSpecies
        neutralSpeciesIDs(end+1) = i;
      end
    end
    % evaluate parameters needed for the calculation of the inverse of the binary diffusion coefficients
    reducedMass = zeros(length(stateArray));
    sigma = zeros(length(stateArray));
    epsilon = zeros(length(stateArray));
    for idx = 1:length(neutralSpeciesIDs)
      for jdx = idx+1:length(neutralSpeciesIDs)
        i = neutralSpeciesIDs(idx);
        j = neutralSpeciesIDs(jdx);
        % error checking
        if isempty(stateArray(i).gas.mass)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\nMass of %s not found.\n' ...
            'Please, fix the problem and run the code again'], reactionArray(reactionID).type, ...
            reactionArray(reactionID).description, stateArray(i).gas.name);
        elseif isempty(stateArray(j).gas.mass)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\nMass of %s not found.\n' ...
            'Please, fix the problem and run the code again'], reactionArray(reactionID).type, ...
            reactionArray(reactionID).description, stateArray(j).gas.name);
        elseif isempty(stateArray(i).gas.lennardJonesDistance)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
            '''lennardJonesDistance'' property of %s not found.\nPlease, fix the problem and run the code again'], ...
            reactionArray(reactionID).type, reactionArray(reactionID).description, stateArray(i).gas.name);
        elseif isempty(stateArray(j).gas.lennardJonesDistance)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
            '''lennardJonesDistance'' property of %s not found.\nPlease, fix the problem and run the code again'], ...
            reactionArray(reactionID).type, reactionArray(reactionID).description, stateArray(j).gas.name);
        elseif isempty(stateArray(i).gas.lennardJonesDepth)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
            '''lennardJonesDepth'' property of %s not found.\nPlease, fix the problem and run the code again'], ...
            reactionArray(reactionID).type, reactionArray(reactionID).description, stateArray(i).gas.name);
        elseif isempty(stateArray(j).gas.lennardJonesDepth)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
            '''lennardJonesDepth'' property of %s not found.\nPlease, fix the problem and run the code again'], ...
            reactionArray(reactionID).type, reactionArray(reactionID).description, stateArray(j).gas.name);
        end
        % evaluation of different auxiliary variables needed
        reducedMass(i,j) = stateArray(i).gas.mass*stateArray(j).gas.mass/(stateArray(i).gas.mass+stateArray(j).gas.mass);
        sigma(i,j) = 0.5*(stateArray(i).gas.lennardJonesDistance+stateArray(j).gas.lennardJonesDistance);
        epsilon(i,j) = sqrt(stateArray(i).gas.lennardJonesDepth*stateArray(j).gas.lennardJonesDepth);
        reducedMass(j,i) = reducedMass(i,j);
        sigma(j,i) = sigma(i,j);
        epsilon(j,i) = epsilon(i,j);
      end
    end 
    % initialize array of numberOfChannels (number of destruction channels at the wall for each species)
    numberOfChannelsArray = zeros(size(stateArray));
  end
  
  % evaluate number of destruction channels at the wall for the diffused species (only done once per lost species)
  if numberOfChannelsArray(reactantID) == 0
    % evaluate total wall reaction coefficient
    for reaction = stateArray(reactantID).reactionsDestruction
      if reaction.isTransport
        if strcmp(reaction.type, reactionArray(reactionID).type)
          numberOfChannelsArray(reactantID) = numberOfChannelsArray(reactantID) + 1;
        else % error checking
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
            'It has been found another transport reaction of a different type for the same species:\n%s.\nPlease, ' ...
            'fix the problem and run the code again'], reactionArray(reactionID).type, ...
            reactionArray(reactionID).description, reaction.description);
        end
      end
    end
  end

  % --- regular calculations ---
  
  % evaluate total neutral density (neutralSpeciesIDs only include childless neutral species)
  neutralDens = sum(densitiesAll(neutralSpeciesIDs));
  
  % evaluation of the inverse of the reduced binary diffusion coeficients
  inverseReducedBinaryDiffCoeff = zeros(size(densitiesAll));
  for i = neutralSpeciesIDs
    if i == reactantID
      continue;
    end
    Taux = workCond.gasTemperature*Constant.boltzmann/epsilon(i,reactantID);
    % fit to first order collision integral from (R.J. Kee, M.E. Coltrin, P. Glarborg "Chemically Reacting Flow:
    % Theory and Practice" John Wiley (2003), Pag. 492)
    collIntegral = 1.0548*Taux^-0.15504+(Taux+0.55909)^-2.1705;
    % actual evaluation of the inverse of the reduced binary diffusion coeficient between species i and j
    inverseReducedBinaryDiffCoeff(i) = 16/3*sqrt(pi*reducedMass(i,reactantID)/(2*Taux*epsilon(i,reactantID)))*...
      sigma(i,reactantID)^2*collIntegral;
  end

  % evaluate effective diffusion coefficient (using Wilke's formula)
  effDiffCoeff = (1-densitiesAll(reactantID)/neutralDens)/sum(densitiesAll.*inverseReducedBinaryDiffCoeff);
  
  % evaluate rate coefficient
  rateCoeff = (1/numberOfChannelsArray(reactantID)) * (effDiffCoeff/charDiffLengthSquared);

  % set function dependencies
  dependent = dependentInfo;

end
