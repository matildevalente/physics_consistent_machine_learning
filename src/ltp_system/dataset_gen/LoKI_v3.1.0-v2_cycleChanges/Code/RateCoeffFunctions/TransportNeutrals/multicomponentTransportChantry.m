function [rateCoeff, dependent] = multicomponentTransportChantry(~, densitiesAll, ~, reactionArray, reactionID, ...
  stateArray, workCond, ~, rateCoeffParams)
  % multicomponentTransportChantry evaluates the transport rate coefficient of a particular species interacting with the 
  % wall, taking into account both the multicomponent diffusion time (using a Blanc's law) and the wall reaction time 
  % assuming a flux reaching the wall (with a certain wall recombination/deactivation probability). 
  % NOTE: this implementation  of the multicomponent transport uses the "heuristic" formula provided by Chantry for the
  % effective diffusion lenght.
  %
  % Chantry, P. J. (1987). Journal of Applied Physics, 62(4), 1141?1148. https://doi.org/10.1063/1.339662
  % 
  % For more info check documentation.
  
  persistent neutralSpeciesIDs;
  persistent charDiffLengthSquared;
  persistent reducedMass;
  persistent sigma;
  persistent epsilon;
  persistent gammaSumArray;
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
    % initialize array of gammaSum (total loss wall probability)
    gammaSumArray = zeros(size(stateArray));
  end

  % evaluate total wall reaction probability for the reactant specie (only done once per lost species)
  if gammaSumArray(reactantID) == 0
    % evaluate total wall reaction coefficient
    for reaction = stateArray(reactantID).reactionsDestruction
      if reaction.isTransport
        if strcmp(reaction.type, reactionArray(reactionID).type)
          if ~isempty(reaction.rateCoeffParams) && ...
              isnumeric(reaction.rateCoeffParams{1}) && reaction.rateCoeffParams{1}<=1 && reaction.rateCoeffParams{1}>0
            gammaSumArray(reactantID) = gammaSumArray(reactantID) + reaction.rateCoeffParams{1};
          else % error checking
            error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n''wallCoefficient'' ' ...
              'not probided (or wrong value) in the corresponding ''.chem'' file.\nPlease, fix the problem and ' ...
              'run the code again'], reactionArray(reactionID).type, reaction.description);
          end
        else % error checking
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
            'It has been found another transport reaction of a different type for the same species:\n%s.\nPlease, ' ...
            'fix the problem and run the code again'], reactionArray(reactionID).type, ...
            reactionArray(reactionID).description, reaction.description);
        end
      end
    end
    % error checking for limiting values of gammaSumArray
    if gammaSumArray(reactantID) == 0
      error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
        'Total wall reaction probability must be different than zero.\nPlease, fix the problem and run the code ' ...
        'again'], reactionArray(reactionID).type, reactionArray(reactionID).description, reaction.description);
    elseif gammaSumArray(reactantID) > 1
      error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
        'Total wall reaction probability can not be larger than 1.\nPlease, fix the problem and run the code ' ...
        'again'], reactionArray(reactionID).type, reactionArray(reactionID).description, reaction.description);
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
  
  % evaluate thermal velocity
  temperatureStr = rateCoeffParams{2};
  if ~any(strcmp(temperatureStr, {'gasTemperature' 'nearWallTemperature' 'wallTemperature'}))
    error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n''%s'' is not a valid temperature ' ...
      'to evaluate the thermal velocity with.\nChoose one of the following: ''gasTemperature'', ''nearWallTemperature'' ' ...
      'or ''wallTemperature''.\nPlease, fix the problem and run the code again'], reactionArray(reactionID).type, ...
      reactionArray(reactionID).description, rateCoeffParams{2});
  elseif isempty(workCond.(temperatureStr))
    error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n''%s'' not found in the working ' ...
      'conditions object.\nPlease, fix the problem and run the code again'], reactionArray(reactionID).type, ...
      reactionArray(reactionID).description, rateCoeffParams{2});
  else
    thermalVelocity = sqrt(8*Constant.boltzmann*workCond.(temperatureStr)/(pi*stateArray(reactantID).gas.mass));
  end

  % evaluate wall reaction coefficient
  gamma = rateCoeffParams{1};
  
  % evaluate rate coefficient
  rateCoeff = (gamma/gammaSumArray(reactantID)) / (charDiffLengthSquared/effDiffCoeff + ...
    workCond.volumeOverArea*4*(1-gammaSumArray(reactantID)/2)/(thermalVelocity*gammaSumArray(reactantID)) );

  % set function dependencies
  dependent = dependentInfo;

end
