function [rateCoeff, dependent] = nitrogenEVscaling(~, ~, ~, reactionArray, reactionID, ~, ~, ~, ~)
% nitrogenEVscaling evaluate the rate coefficients for eV transitions in nitrogen following the scaling law described in
% A. Bourdon, P. Vervisch, J. Thermophys. Heat Transf. 14 (2000) 489?495. doi:10.2514/2.6571.

  persistent equivalentReactionID;

  currentReaction = reactionArray(reactionID);                          % obtain current reaction
  vini = str2double(currentReaction.reactantArray.vibLevel);            % obtain initial vibrational level
  vfin = str2double(currentReaction.productArray.vibLevel);             % obtain final vibrational level
  n = vfin - vini;                                                      % evaluate vibrational quanta jump (positive or negative)
  equivReactionDscrp = sprintf('e + N2(X,v=0) <-> e + N2(X,v=%d)', abs(n));   % evaluate equivalent reaction description

  % find equivalent reaction
  if isempty(equivalentReactionID) || length(equivalentReactionID)<reactionID || equivalentReactionID(reactionID)==0
    for idx=1:length(reactionArray)
      if strcmp(reactionArray(idx).description, equivReactionDscrp)
        equivalentReactionID(reactionID) = idx;
        break;
      elseif idx == length(reactionArray)
        error('Could not find reaction: %s\nNeeded to evaluate ''nitrogenEVscaling'' for reaction %s\n', ...
          equivReactionDscrp, reactionArray(reactionID).description);
      end
    end
  end

  % scale equivalent rate coefficient
  ID = equivalentReactionID(reactionID);
  if n>0   % inelastic collision
    rateCoeff = reactionArray(ID).eedfEquivalent.ineRateCoeff/(1+0.15*vini);
  else     % superelastic collision
    rateCoeff = reactionArray(ID).eedfEquivalent.supRateCoeff/(1+0.15*vfin);
  end

  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', true);

end
