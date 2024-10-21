function [rateCoeff, dependent] = modifiedArrheniusGasTemp(~, ~, ~, ~, ~, ~, workCond, ~, rateCoeffParams)
% modifiedArrheniusGasTemp evaluates a reaction rate coefficients using the following expression:

  Tg = workCond.gasTemperature;
  a = rateCoeffParams{1};
  b = rateCoeffParams{2};
  c = rateCoeffParams{3};

  rateCoeff = a * Tg^b * exp(c/Tg);
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', false);

end