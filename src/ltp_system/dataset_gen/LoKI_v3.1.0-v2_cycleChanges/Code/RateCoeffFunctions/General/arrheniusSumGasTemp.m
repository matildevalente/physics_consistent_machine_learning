function [rateCoeff, dependent] = arrheniusSumGasTemp(~, ~, ~, ~, ~, ~, workCond, ~, rateCoeffParams)
% arrheniusSumGasTemp evaluates a reaction rate coefficients using the following expression:

  Tg = workCond.gasTemperature;
  a0 = rateCoeffParams{1};
  a = rateCoeffParams{2};
  b = rateCoeffParams{3};

  rateCoeff = a0 + a * exp(b/Tg);
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', false);

end
