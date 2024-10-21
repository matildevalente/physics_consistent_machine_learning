function [rateCoeff, dependent] = constantRateCoeff(~, ~, ~, ~, ~, ~, ~, ~, rateCoeffParams)
% constantRateCoeff evaluates a reaction rate coefficients using the following expression:

  rateCoeff = rateCoeffParams{1};
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', false);

end