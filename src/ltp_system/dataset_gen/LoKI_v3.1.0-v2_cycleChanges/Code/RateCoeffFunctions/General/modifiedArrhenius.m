function [rateCoeff, dependent] = modifiedArrhenius(~, ~, ~, ~, ~, ~, ~, ~, rateCoeffParams)
% modifiedArrhenius evaluates a reaction rate coefficients using the following expression:

  T = rateCoeffParams(1);
  a = rateCoeffParams(2);
  b = rateCoeffParams(3);
  c = rateCoeffParams(4);

  rateCoeff = a * T^b * exp(c/T);
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', false);

end