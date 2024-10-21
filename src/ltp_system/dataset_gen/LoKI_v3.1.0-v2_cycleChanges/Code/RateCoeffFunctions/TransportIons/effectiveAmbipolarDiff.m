function [rateCoeff, dependent] = effectiveAmbipolarDiff(time, densitiesAll, totalGasDensity, reactionArray, ...
  reactionID, stateArray, workCond, eTransProp, rateCoeffParams)
% effectiveAmbipolarDiff evaluates the effective ambipolar diffusion rate coefficient of a particular positive ion,
% for a plasma with multiple positive ions and a single negative ion with low density,
% using the transport model described in
% P Coche, V Guerra and L L Alves, J. Phys. D: Appl. Phys. 49 (2016) 235207
% http://dx.doi.org/10.1088/0022-3727/49/23/235207

  persistent firstBesselZero;
  persistent positiveIonIDs;
  persistent negativeIonIDs;
  persistent positiveIonMasses;
  persistent initialTime;
  persistent characteristicLengthSquared;
  persistent auxiliary;
  persistent chi;
  persistent alpha;
  persistent abacusFunction;

  % --- performance sensitive calculations ---

  % initialize variables the first time the classicalAmbipolar function is called
  if isempty(firstBesselZero)
    % evaluate the first zero of the zero order bessel function
    firstBesselZero = fzero(@(x) besselj(0,x), [2.4 2.5]);
    % save initial time of the simulation
    initialTime = time;
    % evaluate the IDs of the positive ions (singly ionized and gas phase)
    for i = 1:length(stateArray)
      state = stateArray(i);
      if strcmp(state.ionCharg, '+') && state.isVolumeSpecies
        positiveIonIDs(end+1) = state.ID;
      elseif strcmp(state.ionCharg, '-') && state.isVolumeSpecies
        negativeIonIDs(end+1) = state.ID;
      end
    end
  end

  % --- time independent calculations (also performance sensitive) ---

  if time == initialTime
    % evaluate the squared characteristic length
    if workCond.chamberLength == 0
      characteristicLengthSquared = (workCond.chamberRadius/firstBesselZero)^2;
    elseif workCond.chamberRadius == 0
      characteristicLengthSquared = (workCond.chamberLength/pi)^2;
    else
      characteristicLengthSquared = 1/((workCond.chamberRadius/firstBesselZero)^-2+(workCond.chamberLength/pi)^-2);
    end
    for i=1:length(positiveIonIDs)
      positiveIonMasses(i) = stateArray(positiveIonIDs(i)).mass;
    end
  end

  % --- regular calculations ---

  % evaluate the reduced diffusion coefficient and reduced mobility for all positive ions (singly ionized),
  %  weigthed by the corresponding ion densities
  % evaluate the weigthed ion density-to-mass ratio, for all positive ions (singly ionized)
  ionRedDiffCoeffWeightedSum = 0;
  ionRedMobCoeffWeightedSum = 0;
  ionDensitySum = 0;
  ionMassWeightedSum = 0;
  for i = 1:length(positiveIonIDs)
    ionID = positiveIonIDs(i);
    ionDensity = densitiesAll(ionID);
    ionDensitySum = ionDensitySum + ionDensity;
    ionRedDiffCoeffWeightedSum = ionRedDiffCoeffWeightedSum + ...
      ionDensity*stateArray(ionID).evaluateReducedDiffCoeff(workCond);
    ionRedMobCoeffWeightedSum = ionRedMobCoeffWeightedSum + ...
      ionDensity*stateArray(ionID).evaluateReducedMobility(workCond);
    ionMassWeightedSum = ionMassWeightedSum + ionDensity/positiveIonMasses(i);
  end

  % obtain electron data
  electronDensity = workCond.electronDensity;
  electronRedDiffCoeff = eTransProp.reducedDiffCoeff;
  electronRedMobCoeff = eTransProp.reducedMobility;
  electronCharEnergy = electronRedDiffCoeff/electronRedMobCoeff;

  % evaluate auxiliary factor for the ion mobilities
  auxiliary = (ionRedDiffCoeffWeightedSum - electronDensity*electronRedDiffCoeff) / ...
    (ionRedMobCoeffWeightedSum + electronDensity*electronRedMobCoeff);

  % evaluate correction factor due to the presence of negative ion
  % (correction valid for a single negative ion, with low density, and at low pressure)
  includeNegativeIon = rateCoeffParams{1};
  if ~islogical(includeNegativeIon)
    error(['Error in the parameters of reaction:\n%s\nUnknown parameter for the ''effectiveAmbipolarDiff'' ' ...
      'function'], reactionArray(reactionID).description);
  end
  if ~includeNegativeIon
    chi = 1;
    alpha=0;
  else
    if length(negativeIonIDs) == 1
      alpha = densitiesAll(negativeIonIDs)/electronDensity;
      if alpha <= 0.1
        electronTemperatureInKelvin = workCond.electronTemperature / Constant.boltzmannInEV;
        chi = (1 + alpha * electronTemperatureInKelvin/workCond.gasTemperature) / (1 + alpha);
      else
        error(['The diffusion model in effectiveAmbipolarDiff is not valid for large negative ion density; '...
          'alpha = %f'],alpha);
      end
    elseif isempty(negativeIonIDs)
      error('There is no negative ion to include in the diffusion model in effectiveAmbipolarDiff');
    else
      negativeIonNames = '';
      for i=1:length(negativeIonIDs)
        negativeIonNames = [negativeIonNames ', ' stateArray(negativeIonIDs(i)).name];
      end
      error('The diffusion model in effectiveAmbipolarDiff is not valid for more than one negative ion (%s)', ...
        negativeIonNames(3:end));
    end
  end

  % evaluate ratio of diffusion length to ion mean-free-path (undetermined expression when ionDensitySum=0)
  lengthRatio = totalGasDensity/ionRedMobCoeffWeightedSum*sqrt(characteristicLengthSquared * ...
    Constant.electronCharge*ionMassWeightedSum*ionDensitySum/(3*electronCharEnergy))*sqrt(chi);
  if includeNegativeIon && lengthRatio > 1
    error('The lengthRatio (%f) is too large for the validity of the diffusion model in effectiveAmbipolarDiff', ...
      lengthRatio);
  end
  if (lengthRatio < 1e-3 || lengthRatio > 100000)
    error('Check the lengthRatio (%f) for the diffusion model in effectiveAmbipolarDiff', lengthRatio);
  end

  % evaluate Self and Ewald abacus function for translating the diffusion coefficient from ambipolar into effective
  if ionDensitySum == 0
    abacusFunction = 0;
  else
    abacusFunction = tanh(lengthRatio^0.36)^2.9;
  end
  if (abacusFunction > 1)
    error('The abacusFunction in effectiveAmbipolarDiff is larger than unit');
  end

  % evaluate effective diffusion rate coefficient
  redDiffCoeff = reactionArray(reactionID).reactantArray.evaluateReducedDiffCoeff(workCond);
  redMobility = reactionArray(reactionID).reactantArray.evaluateReducedMobility(workCond);
  rateCoeff = (redDiffCoeff-redMobility*auxiliary)*abacusFunction*(1+alpha)/chi/...
    (characteristicLengthSquared*totalGasDensity);

  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', true, 'onGasTemperature', true, 'onElectronKinetics', true);

end
