function [rateCoeff, dependent] = wallReactionSurface(~, ~, ~, reactionArray, reactionID, stateArray, workCond, ~, ...
  rateCoeffParams)
  % wallReactionSurface evaluates the transport rate coefficient of a particular species interacting with a wall, 
  % assuming a flux reaching the wall (neglecting diffusion time) and a certain wall recombination/deactivation 
  % probability. Additionally, the previous rate coefficient is modified to accomodate for the collision of the volume
  % species with ONE surface species (dividing by surfaceSiteDensity*AreaOverVolume).
  % 
  % For more info check documentation.
  
  persistent gammaSumArray
  persistent dependentInfo;

  % local save of the ID of the volume reactant species
  for i = 1:length(reactionArray(reactionID).reactantArray)
    if reactionArray(reactionID).reactantArray(i).isVolumeSpecies
      volumeReactantID = reactionArray(reactionID).reactantArray(i).ID;
    end
  end
  
  % --- performance sensitive calculations ---
  
  % calculations performed once per simulation 
  if isempty(dependentInfo)
    % define dependencies of the rate coefficient
    dependentInfo = struct('onTime', false, 'onDensities', true, 'onGasTemperature', true, 'onElectronKinetics', false);
    % initialize array of gammaSum (total loss wall probability) and thermal velocities
    gammaSumArray = zeros(size(stateArray));
  end
  % evaluate total wall reaction probability (total loss) and thermal velocities
  if gammaSumArray(volumeReactantID) == 0
    % evaluate total wall reaction coefficient
    for reaction = stateArray(volumeReactantID).reactionsDestruction
      if reaction.isTransport
        if strcmp(reaction.type, reactionArray(reactionID).type)
          if ~isempty(reaction.rateCoeffParams) && ...
              isnumeric(reaction.rateCoeffParams{1}) && reaction.rateCoeffParams{1}<=1 && reaction.rateCoeffParams{1}>0
            gammaSumArray(volumeReactantID) = gammaSumArray(volumeReactantID) + reaction.rateCoeffParams{1};
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
    if gammaSumArray(volumeReactantID) == 0
      error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
        'Total wall reaction probability must be different than zero.\nPlease, fix the problem and run the code ' ...
        'again'], reactionArray(reactionID).type, reactionArray(reactionID).description, reaction.description);
    elseif gammaSumArray(volumeReactantID) > 1
      error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
        'Total wall reaction probability can not be larger than 1.\nPlease, fix the problem and run the code ' ...
        'again'], reactionArray(reactionID).type, reactionArray(reactionID).description, reaction.description);
    end
    % error checking for species mass
    if isempty(reactionArray(reactionID).reactantArray(1).gas.mass)
      error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\nMass of %s not ' ...
        'found.\nPlease, fix the problem and run the code again'], reactionArray(reactionID).type, ...
        reactionArray(reactionID).description, reactionArray(reactionID).reactantArray(1).name);
    end
  end
    
  % --- regular calculations ---
  
  % evaluate wall reaction coefficient
  gamma = rateCoeffParams{1};

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

  % evaluate rate coefficient
  rateCoeff = workCond.areaOverVolume*thermalVelocity/4*gamma/(1-gammaSumArray(volumeReactantID)/2);
  rateCoeff = rateCoeff/(workCond.surfaceSiteDensity*workCond.areaOverVolume);
  
  % set function dependencies
  dependent = dependentInfo;

end