function [rateCoeff, dependent] = ionTransportQDB(~, densitiesAll, ~, reactionArray, reactionID, ...
  stateArray, workCond, ~, rateCoeffParams)
  % ionTransportQDB evaluates the transport rate coefficient of a particular positive ion
  % for a plasma with multiple positive and negative ions (assumed singly-charged)
  % - with similar temperatures, T_pos ~ T_neg  
  % - with very low mobilities, satisfying mu_pos ~ mu_neg << mu_e
  % using the transport model described in
  %  E. Stoffels, W. W. Stoffels, D. Vender, M. Haverlag, G. M. W. Kroesen, and F. J. de Hoog
  %  Contributions to Plasma Physics, 35 (1995) 331-357
  % and adopted by the QDB platform
  % For more info check the QDB documentation.

  persistent neutralSpeciesIDs;
  persistent positiveIonIDs;
  persistent negativeIonIDs;
  persistent charDiffLengthSquared;
  persistent sigma;
  persistent reducedMass;
  persistent dependentInfo;
  
  % local save of the ID of the reactant species
  reactantID = reactionArray(reactionID).reactantArray(1).ID;

  % --- performance sensitive calculations ---
  
  % calculations performed only once per simulationm when this function is called for the first time  
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
    % find the IDs of the (volume) positive and negative ions (singly ionized)
    positiveIonIDs = [];
    negativeIonIDs = [];
    for i = 1:length(stateArray)
      state = stateArray(i);
      if strcmp(state.ionCharg, '+') && state.isVolumeSpecies
        positiveIonIDs(end+1) = state.ID;
      elseif strcmp(state.ionCharg, '-') && state.isVolumeSpecies
        negativeIonIDs(end+1) = state.ID;
      end
    end
    % find the IDs of (volume) neutral species
    neutralSpeciesIDs = [];
    for i = 1:length(stateArray)
      if isempty(stateArray(i).ionCharg) && isempty(stateArray(i).childArray) && stateArray(i).isVolumeSpecies
        neutralSpeciesIDs(end+1) = i;
      end
    end
    % evaluate parameters needed for the calculation of the inverse of the binary diffusion coefficients
    reducedMass = zeros(length(stateArray));
    sigma = zeros(length(stateArray));
    % parameters for positive_ion-neutral collisions
    for i = positiveIonIDs
      for j = neutralSpeciesIDs
        % error checking
        if isempty(stateArray(i).gas.lennardJonesDistance)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
            '''lennardJonesDistance'' property of %s not found.\nPlease, fix the problem and run the code again'], ...
            reactionArray(reactionID).type, reactionArray(reactionID).description, stateArray(i).gas.name);
        elseif isempty(stateArray(j).gas.lennardJonesDistance)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\n' ...
            '''lennardJonesDistance'' property of %s not found.\nPlease, fix the problem and run the code again'], ...
            reactionArray(reactionID).type, reactionArray(reactionID).description, stateArray(j).gas.name);
        end
        % evaluation of the hard-spheres cross section for positive_ion-neutral collisions
        sigma(i,j) = (stateArray(i).gas.lennardJonesDistance+stateArray(j).gas.lennardJonesDistance)^2;
      end
    end
    % parameters for ion-ion collisions
    for i = positiveIonIDs
      for j = positiveIonIDs
        if i==j
          continue;
        end
        % error checking
        if isempty(stateArray(i).gas.mass)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\nMass of %s not found.\n' ...
            'Please, fix the problem and run the code again'], reactionArray(reactionID).type, ...
            reactionArray(reactionID).description, stateArray(i).gas.name);
        elseif isempty(stateArray(j).gas.mass)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\nMass of %s not found.\n' ...
            'Please, fix the problem and run the code again'], reactionArray(reactionID).type, ...
            reactionArray(reactionID).description, stateArray(j).gas.name);
        end
        % evaluation of the reduced mass of the colliding particles
        reducedMass(i,j) = stateArray(i).gas.mass*stateArray(j).gas.mass/(stateArray(i).gas.mass+stateArray(j).gas.mass);
      end
    end
    for i = positiveIonIDs
      for j = negativeIonIDs
        % error checking
        if isempty(stateArray(i).gas.mass)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\nMass of %s not found.\n' ...
            'Please, fix the problem and run the code again'], reactionArray(reactionID).type, ...
            reactionArray(reactionID).description, stateArray(i).gas.name);
        elseif isempty(stateArray(j).gas.mass)
          error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\nMass of %s not found.\n' ...
            'Please, fix the problem and run the code again'], reactionArray(reactionID).type, ...
            reactionArray(reactionID).description, stateArray(j).gas.name);
        end
        % evaluation of the reduced mass of the colliding particles
        reducedMass(i,j) = stateArray(i).gas.mass*stateArray(j).gas.mass/(stateArray(i).gas.mass+stateArray(j).gas.mass);
      end
    end
  end

  % --- regular calculations ---

  % check that the reactant species is a (volume) positive ion
  if ~strcmp(stateArray(reactantID).ionCharg, '+') || stateArray(reactantID).isSurfaceSpecies
    error(['Error found when evaluating ''%s'' rate coefficient for reaction:\n%s.\nReactant %s is not a (volume) ' ...
      'positive ion.\nPlease, fix the problem and run the code again'], reactionArray(reactionID).type, ...
      reactionArray(reactionID).description, stateArray(reactantID).name);
  end

  % Calculation of the ion temperature
  if workCond.gasPressure > 0.133 
    ionTemperature = (5800 - workCond.gasTemperature) * 0.133/workCond.gasPressure + workCond.gasTemperature;
  else
    ionTemperature = 5800;
  end

  % evaluate the thermal velocity for the reactant species
  thermalVelocity = sqrt(8*Constant.boltzmann*ionTemperature/(pi*stateArray(reactantID).gas.mass));

  % evaluate the inverse of the mean free path for the reactant species
  inverseLambda = 0;
  for j = neutralSpeciesIDs
    inverseLambda = inverseLambda + densitiesAll(j)*sigma(reactantID,j);
  end
  includeIonCorrectionDiffCoeff = rateCoeffParams{2}; % boolean rate coefficien parameter (true or false)
  if includeIonCorrectionDiffCoeff %check if the effective diffusion coefficient must include corrections due to ions
    for j = positiveIonIDs
      if j==reactantID
        continue;
      end
      b0 = Constant.electronCharge^2/(2*pi*Constant.vacuumPermittivity*reducedMass(reactantID,j)*thermalVelocity^2);
      lambdaDebye = sqrt(Constant.vacuumPermittivity*workCond.electronTemperature/ ...
        (Constant.electronCharge*workCond.electronDensity));
      sigma(reactantID,j) = pi*b0^2*log(2*lambdaDebye/b0);
      inverseLambda = inverseLambda + densitiesAll(j)*sigma(reactantID,j);
    end
    for j = negativeIonIDs
      b0 = Constant.electronCharge^2/(2*pi*Constant.vacuumPermittivity*reducedMass(reactantID,j)*thermalVelocity^2);
      lambdaDebye = sqrt(Constant.vacuumPermittivity*workCond.electronTemperature/ ...
        (Constant.electronCharge*workCond.electronDensity));
      sigma(reactantID,j) = pi*b0^2*log(2*lambdaDebye/b0);
      inverseLambda = inverseLambda + densitiesAll(j)*sigma(reactantID,j);
    end
  end

  % evaluate the diffusion coefficient for the reactant species
  effDiffCoeff = (pi/8) / inverseLambda*thermalVelocity;

  % evaluate the corrective factor due to the presence of negative ions
  alpha = 0;
  if isempty(negativeIonIDs)
    gamma = 0;
  else
    for i = negativeIonIDs
      alpha = alpha + densitiesAll(i)/workCond.electronDensity;
    end
    electronTemperatureInKelvin = workCond.electronTemperature / Constant.boltzmannInEV;
    gamma = electronTemperatureInKelvin/workCond.gasTemperature;
  end
  correctiveFactorNegativeIons = (1 + gamma*(1+2*alpha))/(1+alpha*gamma);

  % evaluate wall probability
  wallProbability = rateCoeffParams{1};
  % evaluate rate coefficient
  rateCoeff = wallProbability*effDiffCoeff*correctiveFactorNegativeIons/charDiffLengthSquared;

  % set function dependencies
  dependent = dependentInfo;


end

