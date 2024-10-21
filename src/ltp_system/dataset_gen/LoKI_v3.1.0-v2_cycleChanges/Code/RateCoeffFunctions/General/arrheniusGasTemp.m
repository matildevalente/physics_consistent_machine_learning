function [rateCoeff, dependent] = arrheniusGasTemp(~, ~, ~, ~, ~, ~, workCond, ~, rateCoeffParams)
% arrheniusGasTemp evaluates a reaction rate coefficients using the following expression:

  Tg = workCond.gasTemperature;
  a = rateCoeffParams{1};
  b = rateCoeffParams{2};

  rateCoeff = a * exp(b/Tg);
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', false);

end