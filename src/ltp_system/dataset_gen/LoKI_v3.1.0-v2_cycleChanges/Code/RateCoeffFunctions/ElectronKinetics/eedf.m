function [rateCoeff, dependent] = eedf(~, ~, ~, reactionArray, reactionID, ~, ~, ~, ~)
% eedf obtains a reaction rate coefficient by integrating a cross section with an eedf (either prescribed or
% obtained as the solution of a boltzmann equation)
%

  rateCoeff = reactionArray(reactionID).eedfEquivalent.ineRateCoeff;
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', true);

end