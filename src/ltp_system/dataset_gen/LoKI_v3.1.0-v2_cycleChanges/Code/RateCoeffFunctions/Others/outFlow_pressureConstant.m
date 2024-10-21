function [rateCoeff, dependent] = outFlow_pressureConstant(time, ~, totalGasDensity, reactionArray, ~, ~, workCond, ~, ~)
% outFlow evaluates the rate coefficient (in s-1) for the outlet flow of species 
%  assuming conservation of atoms in the gas/plasma mixture
%  (hence ensuring also mass conservation, although via a more restrictive assumption)

% calculated only for t=0
% this function is to be used together with the compagnion function inFlow
% to call the function use: e -> O2(X)  | outFlow | |
% !!! this is a preliminary implementation that uses the massless "e" species to define the mechanism !!!
% proposed by T Dias (May 2020)
% verified and adjusted by LL Alves (May and June 2020)

  currentPressure = totalGasDensity*Constant.boltzmann*workCond.gasTemperature;

  % calculate the inflow rate coefficient (in s-1)
  totalSimulationTime = 1000;
  rateCoeff = (currentPressure-workCond.gasPressure)/workCond.gasPressure * 1E6/totalSimulationTime;
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', true, 'onGasTemperature', true, 'onElectronKinetics', false);

end