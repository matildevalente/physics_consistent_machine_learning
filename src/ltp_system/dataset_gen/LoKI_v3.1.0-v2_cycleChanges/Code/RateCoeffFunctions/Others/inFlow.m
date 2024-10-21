function [rateCoeff, dependent] = inFlow(~, ~, ~, ~, ~, ~, workCond, ~, rateCoeffParams)
% inFlow evaluates the rate coefficient (in s-1) for the inlet flow of species 
%  assuming conservation of atoms in the gas/plasma mixture
%  (hence ensuring also mass conservation, although via a more restrictive assumption)

% calculated only for t=0
% this function if to be used together with the compagnion function outFlow
% to call the function use: e -> O2(X)  | inFlow | inletFreq, volumeFactor |
% inletFreq is the inlet frequency in SI units: 
%          inletFreq (s-1) = inFlowsccm * Constant.atmosphereInPa/Constant.boltzmann*1e-6/60/273.15
% volumeFactor = plasmaVolume/chamberVolume 
% !!! this is a preliminary implementation that uses the massless "e" species to define the mechanism !!!
% proposed by T Dias (May 2020)
% verified and adjusted by LL Alves (May and June 2020)

  persistent chamberVolume;
   if isempty(chamberVolume)
 	  if workCond.chamberRadius == 0 || workCond.chamberLength == 0
 		 error(['Error found when evaluating an inFlow rate-coefficient function.\n' ...
 			   'The length and the radius of the chamber must be different than zero.']);			  
 	  end
 	 chamberVolume = pi*(workCond.chamberRadius^2)*workCond.chamberLength;
  end
  
  inletFreq = rateCoeffParams{1};
  volumeFactor = rateCoeffParams{2};

  % calculate the inflow rate (in m-3 s-1)
  k_in = inletFreq/(volumeFactor*chamberVolume);
  
  % calculate the inflow rate coefficient (in s-1)
  % division by the electron density, since "e" is used as reactant
  rateCoeff = k_in/workCond.electronDensity;
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', false);
end

