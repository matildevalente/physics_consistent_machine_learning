function [rateCoeff, dependent] = expGasTemp(~, ~, ~, ~, ~, ~, workCond, ~, rateCoeffParams)
% expGasTemp evaluates a reaction rate coefficients using the following expression:

  Tg = workCond.gasTemperature; % in K
  a = rateCoeffParams{1};
  b = rateCoeffParams{2};

  rateCoeff = a * exp(Tg/b);
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', false);

end