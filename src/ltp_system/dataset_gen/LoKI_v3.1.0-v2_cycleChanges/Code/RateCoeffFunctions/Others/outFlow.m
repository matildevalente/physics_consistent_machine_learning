function [rateCoeff, dependent] = outFlow(time, ~, totalGasDensity, reactionArray, ~, ~, workCond, ~, ~)
% outFlow evaluates the rate coefficient (in s-1) for the outlet flow of species 
%  assuming conservation of atoms in the gas/plasma mixture
%  (hence ensuring also mass conservation, although via a more restrictive assumption)

% calculated only for t=0
% this function is to be used together with the compagnion function inFlow
% to call the function use: e -> O2(X)  | outFlow | |
% !!! this is a preliminary implementation that uses the massless "e" species to define the mechanism !!!
% proposed by T Dias (May 2020)
% verified and adjusted by LL Alves (May and June 2020)

  persistent totalK_in;
  persistent k_out;
  
  if isempty(totalK_in)
 	if workCond.chamberRadius == 0 || workCond.chamberLength == 0
 		error(['Error found when evaluating an outFlow rate-coefficient function.\n' ...
 			   'The length and the radius of the chamber must be different than zero.']);			  
 	end
 	chamberVolume = pi*(workCond.chamberRadius^2)*workCond.chamberLength;

    totalK_in = 0;
    for reaction = reactionArray
		if strcmp(reaction.type, 'inFlow')
            inletFreq = reaction.rateCoeffParams{1};
            volumeFactor = reaction.rateCoeffParams{2};
			totalK_in = totalK_in + inletFreq/(volumeFactor*chamberVolume);
		end
    end	
  end
  
  if time == 0
	  % calculated only for t=0 using the initial gas density
	  k_out = totalK_in/totalGasDensity;
  end
  
  % calculate the inflow rate coefficient (in s-1)
  rateCoeff = k_out;
  
  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', false, 'onGasTemperature', false, 'onElectronKinetics', false);

end