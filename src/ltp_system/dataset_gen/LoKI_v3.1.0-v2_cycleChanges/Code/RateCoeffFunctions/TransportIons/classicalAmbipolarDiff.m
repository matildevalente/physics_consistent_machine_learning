function [rateCoeff, dependent] = classicalAmbipolarDiff(time, densitiesAll, totalGasDensity, reactionArray, ...
  reactionID, stateArray, workCond, eTransProp, rateCoeffParams)
% classicalAmbipolar evaluates the diffusion rate coefficient of a particular ion, assuming the classical ambipolar
% regime (see documentation)
  
  persistent firstBesselZero;
  persistent positiveIonIDs;
  persistent initialTime;
  persistent characteristicLengthSquared;
  persistent correctionFactor;
  persistent dependentInfo;
  
  % --- performance sensitive calculations ---
  
  % initialize variables the first time the classicalAmbipolar function is called
  if isempty(firstBesselZero)
    % evaluate the first zero of the zero order bessel function
    firstBesselZero = fzero(@(x) besselj(0,x), [2.4 2.5]);
    % save initial time of the simulation
    initialTime = time;
    % define dependencies of the rate coefficient
    dependentInfo = struct('onTime', false, 'onDensities', true, 'onGasTemperature', false, 'onElectronKinetics', false);
    % evaluate the IDs of the positive ions (singly ionized and gas phase) and check dependencies on gas temperature
    for i = 1:length(stateArray)
      state = stateArray(i);
      if strcmp(state.ionCharg, '+') && state.isVolumeSpecies
        positiveIonIDs(end+1) = state.ID;
        if ~dependentInfo.onGasTemperature && ~isempty(state.reducedDiffCoeffFunc)
          for param = state.reducedDiffCoeffParams
            if strcmp(param{1}, 'gasTemperature')
              dependentInfo.onGasTemperature = true;
              break;
            end
          end
        end
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
  end
  
  % --- calculations that are equal for every "classicalAmbipolar" reaction (at a given time) ---
  
  % evaluate weighted sums of the diffusion coefficient and reduced mobility of all positive ions (singly ionized)
  ionRedDiffCoeffSum = 0;
  ionRedMobCoeffSum = 0;
  for i = 1:length(positiveIonIDs)
    ionID = positiveIonIDs(i);
    ionDensity = densitiesAll(ionID);
    ionRedDiffCoeffSum = ionRedDiffCoeffSum + ionDensity*stateArray(ionID).evaluateReducedDiffCoeff(workCond);
    ionRedMobCoeffSum = ionRedMobCoeffSum + ionDensity*stateArray(ionID).evaluateReducedMobility(workCond);
  end
  % obtain electron data
  electronDensity = workCond.electronDensity;
  electronRedDiffCoeff = eTransProp.reducedDiffCoeff;
  electronRedMobCoeff = eTransProp.reducedMobility;
  % evaluate correction factor
  correctionFactor = (ionRedDiffCoeffSum-electronDensity*electronRedDiffCoeff)/...
    (ionRedMobCoeffSum+electronDensity*electronRedMobCoeff);

  
  % --- regular calculations ---
  
  % evaluate diffusion rate coefficient
  if workCond.electronDensity==0 && all(densitiesAll(positiveIonIDs)==0)
    rateCoeff = 0;
  else
    wallProbability = rateCoeffParams{1};
    redDiffCoeff = reactionArray(reactionID).reactantArray.evaluateReducedDiffCoeff(workCond);
    redMobility = reactionArray(reactionID).reactantArray.evaluateReducedMobility(workCond);
    rateCoeff = (redDiffCoeff-redMobility*correctionFactor)/(characteristicLengthSquared*totalGasDensity);
    rateCoeff = wallProbability*rateCoeff;
  end
  % set function dependencies
  dependent = dependentInfo;

end
