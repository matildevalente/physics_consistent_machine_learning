function [rateCoeff, dependent] = powerGasCharEnergyTemp(~, ~, ~, ~, ~, ~, workCond, eTransProp, rateCoeffParams)
% power Gas char energy temp evaluates a reaction rate coefficients using the following expression:

  Tg = workCond.gasTemperature; % in K
  charEnergy = eTransProp.reducedDiffCoeff/eTransProp.reducedMobility/Constant.boltzmannInEV; % in K
  a = rateCoeffParams{1};
  b = rateCoeffParams{2};
  c = rateCoeffParams{3};

  rateCoeff = a * Tg^b * charEnergy^c;

  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', true);

end
