function [rateCoeff, dependent] = classicalAmbipolarDiffNegIon(time, densitiesAll, totalGasDensity, reactionArray, ...
  reactionID, stateArray, workCond, eTransProp, ~)
% classicalAmbipolarDiffNegIon evaluates the diffusion rate coefficient of a particular positive ion, assuming the ambipolar
% regime together with the effect of various negative ions
% Generalized from: 
% V Guerra and J Loureiro Plasma Sources Sci. Technol. 8 (1999) 110-124
% Main approximations:
% - pressure sufficiently high, so as to verify the classical ambipolar regime
% - Cold ion approximation: T_char_ions << T_char_electrons
% - Ion mobilities much smaller than the electron mobility
% - Infinite cylinder
% Made by T C Dias and V Guerra (April 2022)
  
  persistent firstBesselZero;
  persistent positiveIonIDs;
  persistent negativeIonIDs;
  persistent electronCharEnergy;
  persistent characteristicLengthSquared;
  persistent previousTime;
  persistent rateCoeffs;
  persistent rateCoeffIsTimeDep;
  persistent ionReactions;
  
  % --- performance sensitive calculations ---
  
  % initialize variables the first time the classicalAmbipolar function is called
  if isempty(firstBesselZero)
    % evaluate the first zero of the zero order bessel function
    firstBesselZero = fzero(@(x) besselj(0,x), [2.4 2.5]);
    % evaluate the IDs of the positive ions (singly ionized) and check dependencies on gas temperature
    for i = 1:length(stateArray)
      state = stateArray(i);
      if strcmp(state.ionCharg, '+') && state.isVolumeSpecies
        positiveIonIDs(end+1) = state.ID;
        ionReactions = [ionReactions state.reactionsDestruction state.reactionsCreation];
      elseif strcmp(state.ionCharg, '-') && state.isVolumeSpecies
        negativeIonIDs(end+1) = state.ID;
        ionReactions = [ionReactions state.reactionsDestruction state.reactionsCreation];       
      end
    end 
    
    % eliminate repeated ion reactions
    ionReactions = unique(ionReactions);
    % eliminate diffusion reactions
    i = 1;
    while i <= length(ionReactions)
      if ionReactions(i).isTransport
        ionReactions(i) = [];
      else
        i = i +1;
      end
    end  

    % initialize the rate-coefficients array
    rateCoeffs = zeros(size(reactionArray));

    % initialize previousTime variable
    previousTime = -1;
  end

  if time == 0
    rateCoeffIsTimeDep = ones(size(reactionArray));
  end  
  
  % --- calculations that are equal for every "classicalAmbipolarDiffNegIon" reaction (at a given time) ---
  
  if previousTime ~= time
    % actualize previousTime variable
    previousTime = time;
    % obtain electron data
    electronDensity = workCond.electronDensity;
    electronRedDiffCoeff = eTransProp.reducedDiffCoeff;
    electronRedMobCoeff = eTransProp.reducedMobility;
    electronCharEnergy = electronRedDiffCoeff/electronRedMobCoeff;

    % ----- calculate the correction in the characteristic length due to  ---- % 
    % -----       the presence of the negative ion                        ---- %

    if ~isempty(negativeIonIDs)
      % calculate all the rate coefficients that will be needed (the ones
      % that involve an ion)
      for reaction = ionReactions
        if rateCoeffIsTimeDep(reaction.ID)
          [rateCoeffs(reaction.ID), dep] = reaction.rateCoeffFuncHandle(time, densitiesAll, totalGasDensity, reactionArray, reaction.ID, ...
            stateArray, workCond, eTransProp, reaction.rateCoeffParams);
          rateCoeffIsTimeDep(reaction.ID) = dep.onTime || dep.onDensities || dep.onGasTemperature;
        end  
      end

      % calculate the total density of negative ions
      totalNegIonDensity = sum(densitiesAll(negativeIonIDs));   
      
      % calculate the sum of the attachment (detachment) frequencies, weighted by the
      % inverse of the negative-ion mobilities
      totalAttachFrequencyWeighted = 0;
      totalDetachFrequencyWeighted = 0;
      for negIonID = negativeIonIDs
        negIon = stateArray(negIonID);
        negIonDensity = densitiesAll(negIonID);
        % attachment
        attachmentDensityRate = 0;
        for reaction = negIon.reactionsCreation
          partialAttachmentDensityRate = rateCoeffs(reaction.ID)*electronDensity^reaction.reactantElectrons;
          for i = 1:length(reaction.reactantArray)
            partialAttachmentDensityRate = partialAttachmentDensityRate*...
              densitiesAll(reaction.reactantArray(i).ID)^reaction.reactantStoiCoeff(i);
          end
          for i = 1:length(reaction.catalystArray)
            partialAttachmentDensityRate = partialAttachmentDensityRate*...
              densitiesAll(reaction.catalystArray(i).ID)^reaction.catalystStoiCoeff(i);
          end      
          attachmentDensityRate = attachmentDensityRate + partialAttachmentDensityRate;
        end
        totalAttachFrequencyWeighted = totalAttachFrequencyWeighted + attachmentDensityRate/electronDensity/negIon.evaluateReducedMobility(workCond);
        % detachment
        if negIonDensity == 0
          continue;
        end      
        detachmentDensityRate = 0;
        for reaction = negIon.reactionsDestruction
          partialDetachmentDensityRate = rateCoeffs(reaction.ID)*electronDensity^reaction.reactantElectrons;
          for i = 1:length(reaction.reactantArray)
            partialDetachmentDensityRate = partialDetachmentDensityRate*...
              densitiesAll(reaction.reactantArray(i).ID)^reaction.reactantStoiCoeff(i);
          end
          for i = 1:length(reaction.catalystArray)
            partialDetachmentDensityRate = partialDetachmentDensityRate*...
              densitiesAll(reaction.catalystArray(i).ID)^reaction.catalystStoiCoeff(i);
          end  
          detachmentDensityRate = detachmentDensityRate + partialDetachmentDensityRate;
        end
        totalDetachFrequencyWeighted = totalDetachFrequencyWeighted + detachmentDensityRate/negIonDensity/negIon.reducedMobility;               
      end    
  
      % calculate the sum of the net ionization frequencies, weighted by the
      % inverse of the positive-ion mobilities
      totalIonizFrequencyWeighted = 0;
      for posIonID = positiveIonIDs
        ionizDensityRate = 0; 
        posIon = stateArray(posIonID);
        for reaction = posIon.reactionsCreation
          if reaction.isTransport
            continue;
          end  
          partialIonizDensityRate = rateCoeffs(reaction.ID)*electronDensity^reaction.reactantElectrons;
          for i = 1:length(reaction.reactantArray)
            partialIonizDensityRate = partialIonizDensityRate*...
              densitiesAll(reaction.reactantArray(i).ID)^reaction.reactantStoiCoeff(i);
          end
          for i = 1:length(reaction.catalystArray)
            partialIonizDensityRate = partialIonizDensityRate*...
              densitiesAll(reaction.catalystArray(i).ID)^reaction.catalystStoiCoeff(i);
          end
          ionizDensityRate = ionizDensityRate + partialIonizDensityRate;
        end
        for reaction = posIon.reactionsDestruction
          if reaction.isTransport
            continue;
          end
          partialIonizDensityRate = -rateCoeffs(reaction.ID)*electronDensity^reaction.reactantElectrons;
          for i = 1:length(reaction.reactantArray)
            partialIonizDensityRate = partialIonizDensityRate*...
              densitiesAll(reaction.reactantArray(i).ID)^reaction.reactantStoiCoeff(i);
          end
          for i = 1:length(reaction.catalystArray)
            partialIonizDensityRate = partialIonizDensityRate*...
              densitiesAll(reaction.catalystArray(i).ID)^reaction.catalystStoiCoeff(i);
          end
          ionizDensityRate = ionizDensityRate + partialIonizDensityRate;        
        end
        totalIonizFrequencyWeighted = totalIonizFrequencyWeighted + ionizDensityRate/electronDensity/posIon.evaluateReducedMobility(workCond);
      end
      
      % parameter measuring the intensity of attachment
      P = totalAttachFrequencyWeighted/totalIonizFrequencyWeighted;

      % parameter measuring the intensity of detachment
      Q = totalDetachFrequencyWeighted/totalIonizFrequencyWeighted;

      % use an exponential fit to calculate the slope of lambda(P)
      % The fit was obtained so as to reproduce the results of figure 6 of Ferreira 1988
      a1 = 2.614; t1 = 0.5662;
      a2 = 1.34295; t2 = 9.34383;
      a3 = 0.114319; t3 = 139.119;
      slope = a1*exp(-Q/t1)+a2*exp(-Q/t2)+a3*exp(-Q/t3);

      lambda = slope*P + firstBesselZero^2;

      % at initial stages of the simulation, there can be a negative net
      % production of positive ions, which would lead to negative parameters P,Q
      % In these cases, we use the classical solution, with no influence of
      % negative ions
      if totalIonizFrequencyWeighted < 0 || totalAttachFrequencyWeighted < 0 || totalDetachFrequencyWeighted < 0
        characteristicLengthSquared = (workCond.chamberRadius/firstBesselZero)^2;
      else        
        characteristicLengthSquared = workCond.chamberRadius^2/lambda/(1-totalNegIonDensity/(totalNegIonDensity+electronDensity));
      end  
    else
      % evaluate the squared characteristic length
      if workCond.chamberLength == 0
        characteristicLengthSquared = (workCond.chamberRadius/firstBesselZero)^2;
      elseif workCond.chamberRadius == 0
        characteristicLengthSquared = (workCond.chamberLength/pi)^2;
      else
        characteristicLengthSquared = 1/((workCond.chamberRadius/firstBesselZero)^-2+(workCond.chamberLength/pi)^-2);
      end      
    end  
  end
  
  % --- regular calculations ---
  
  % evaluate diffusion rate coefficient
  redMobility = reactionArray(reactionID).reactantArray.evaluateReducedMobility(workCond);
  if isempty(redMobility)
    error(['Error found when evaluating the classicalAmbipolarDiffNegIon rate-coefficient function for\n%s\n'...
          'The ''reducedMobility'' of the state ''%s'' was not found.'],...
          reactionArray(reactionID).descriptionExtended, reactionArray(reactionID).reactantArray.name);
  end   
  rateCoeff = redMobility*electronCharEnergy/(characteristicLengthSquared*totalGasDensity);

  if workCond.electronDensity == 0 || sum(densitiesAll(positiveIonIDs)) == 0
    rateCoeff = 0;
  end  
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', true, 'onGasTemperature', true, 'onElectronKinetics', true);
end