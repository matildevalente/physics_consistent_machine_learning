function [rateCoeff, dependent] = binaryDiffusion(~, ~, totalGasDensity, reactionArray, reactionID, ...
  stateArray, workCond, ~, ~)
  % binaryDiffusion evaluates the transport rate coefficient of a particular species interacting with the 
  % wall, assuming a density profile that is zero at the walls (neglecting the wall reaction time) and assuming equal
  % probability for each destruction channel.
  % 
  % For more info check documentation.

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
    dependentInfo = struct('onTime', false, 'onDensities', true, 'onGasTemperature', false, 'onElectronKinetics', false);
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
  
  % evaluate diffusion rate coefficient
  redDiffCoeff = reactionArray(reactionID).reactantArray(1).evaluateReducedDiffCoeff(workCond);
  
  if isempty(redDiffCoeff)
    error(['Error found when evaluating the rate coefficient of the reaction:\n%s\n'...
      '''reducedDiffCoeff'' property of the state ''%s'' not found.\n'...
      'Please check your setup file'], reactionArray(reactionID).descriptionExtended, ...
      reactionArray(reactionID).reactantArray(1).name);
  end
  rateCoeff = (1/numberOfChannelsArray(reactantID)) * redDiffCoeff/(charDiffLengthSquared*totalGasDensity);

  % set function dependencies
  dependent = dependentInfo;
  if ~isempty(reactionArray(reactionID).reactantArray(1).reducedDiffCoeffFunc)
    for param = reactionArray(reactionID).reactantArray(1).reducedDiffCoeffParams
      if strcmp(param{1}, 'gasTemperature')
        dependent.onGasTemperature = true;
      end
    end
  end

end
