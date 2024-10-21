function [rateCoeff, dependent] = binaryTransportChantrySurface(~, ~, totalGasDensity, reactionArray, reactionID, ...
  stateArray, workCond, ~, rateCoeffParams)
  % binaryTransportChantrySurface evaluates the transport rate coefficient of a particular species interacting with the 
  % wall, taking into account both the binary diffusion time and the wall reaction time assuming a flux reaching the 
  % wall (with a certain wall recombination/deactivation probability). Additionally, the previous rate coefficient is 
  % modified to accomodate for the collision of the volume species with ONE surface species (dividing by 
  % surfaceSiteDensity*AreaOverVolume).
  % NOTE: this implementation  of the binary transport uses the "heuristic" formula provided by Chantry for the
  % effective diffusion lenght.
  %
  % Chantry, P. J. (1987). Journal of Applied Physics, 62(4), 1141?1148. https://doi.org/10.1063/1.339662
  % 
  % For more info check documentation.

  persistent charDiffLengthSquared;
  persistent gammaSumArray;
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
    % initialize array of gammaSum (total loss wall probability)
    gammaSumArray = zeros(size(stateArray));
  end

  % evaluate total wall reaction probability for the reactant specie (only done once per lost species)
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
              'run the code again'], reaction.type, reaction.description);
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
  end
  
  % calculations performed once per time step

  % --- regular calculations ---
  
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
  
  % evaluate diffusion rate coefficient
  redDiffCoeff = stateArray(volumeReactantID).evaluateReducedDiffCoeff(workCond);
  
  % evaluate wall reaction coefficient
  gamma = rateCoeffParams{1};
  % evaluate rate coefficient
  rateCoeff = (gamma/gammaSumArray(volumeReactantID)) / ( charDiffLengthSquared*totalGasDensity/redDiffCoeff + ...
    workCond.volumeOverArea*4*(1-gammaSumArray(volumeReactantID)/2)/(thermalVelocity*gammaSumArray(volumeReactantID)) );
  rateCoeff = rateCoeff/(workCond.surfaceSiteDensity*workCond.areaOverVolume);

  % set function dependencies
  dependent = dependentInfo;
  if ~dependent.onGasTemperature && ...
      ~isempty(stateArray(volumeReactantID).reducedDiffCoeffFunc)
    for param = stateArray(volumeReactantID).reducedDiffCoeffParams
      if strcmp(param{1}, 'gasTemperature')
        dependent.onGasTemperature = true;
      end
    end
  end

end
