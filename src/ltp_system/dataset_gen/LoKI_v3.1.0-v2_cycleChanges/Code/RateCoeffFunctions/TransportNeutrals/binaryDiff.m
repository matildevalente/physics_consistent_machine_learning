function [rateCoeff, dependent] = binaryDiff(time, ~, totalGasDensity, reactionArray, reactionID, ~, workCond, ~, ...
  rateCoeffParams)
% binaryDiff evaluates the diffusion rate coefficient of a particular species, assuming a binary diffusion regime
% where the species diffuse in the gas (see documentation)
  
  persistent firstBesselZero;
  persistent initialTime;
  persistent characteristicLengthSquared;
  persistent dependentInfo;
  
  % --- performance sensitive calculations ---
  
  % initialize variables the first time the binaryDiff function is called
  if isempty(firstBesselZero)
    firstBesselZero = fzero(@(x) besselj(0,x), [2.4 2.5]);
    initialTime = time;
    % define dependencies of the rate coefficient
    dependentInfo = struct('onTime', false, 'onDensities', true, 'onGasTemperature', false, 'onElectronKinetics', false);
  end
  
  % --- time independent calculations (also performance sensitive) ---
  
  % evaluate the squared characteristic length (every initial time, performance sensitive)
  if time == initialTime
    if workCond.chamberLength == 0
      characteristicLengthSquared = (workCond.chamberRadius/firstBesselZero)^2;
    elseif workCond.chamberRadius == 0
      characteristicLengthSquared = (workCond.chamberLength/pi)^2;
    else
      characteristicLengthSquared = 1/((workCond.chamberRadius/firstBesselZero)^-2+(workCond.chamberLength/pi)^-2);
    end
  end
  
  % --- regular calculations ---
  
  % evaluate diffusion rate coefficient
  wallProbability = rateCoeffParams{1};
  redDiffCoeff = reactionArray(reactionID).reactantArray.evaluateReducedDiffCoeff(workCond);
  
  if isempty(redDiffCoeff)
    error(['Error found when evaluating the rate coefficient of the reaction:\n%s\n'...
      '''reducedDiffCoeff'' property of the state ''%s'' not found.\n'...
      'Please check your setup file'], reactionArray(reactionID).descriptionExtended, ...
      reactionArray(reactionID).reactantArray.name);
  end
  rateCoeff = wallProbability*redDiffCoeff/(characteristicLengthSquared*totalGasDensity);
  
  % set function dependencies
  dependent = dependentInfo;
  if ~dependent.onGasTemperature && ~isempty(reactionArray(reactionID).reactantArray.reducedDiffCoeffFunc)
    for param = reactionArray(reactionID).reactantArray.reducedDiffCoeffParams
      if strcmp(param{1}, 'gasTemperature')
        dependent.onGasTemperature = true;
      end
    end
  end   

end