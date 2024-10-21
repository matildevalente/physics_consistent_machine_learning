function [rateCoeff, dependent] = nitrogenAtomicVT(~, ~, ~, reactionArray, reactionID, ~, workCond, ~, ~)
% nitrogenAtomicVT evaluates the rate coefficient for V-T transitions between nitrogen molecules and atoms
% described in:
% L.L. Alves, L. Marques, C.D. Pintassilgo, G. Wattieaux, E. Es-sebbar, J. Berndt, et al., Capacitively coupled 
% radio-frequency discharges in nitrogen at low pressures, Plasma Sources Sci. Technol. 21 (2012) 45008.
% http://dx.doi.org/10.1088/0963-0252/21/4/045008

  % persistent variables for performance reasons
  persistent dependentInfo;
  persistent levelsNeedToBeIdentified;
  persistent v;
  persistent w;
  
  % --- performance sensitive calculations ---
  
  % initialize variables the first time the classicalAmbipolar function is called
  if isempty(dependentInfo)
    dependentInfo = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', false);
    levelsNeedToBeIdentified = true(size(reactionArray));
    v = zeros(size(reactionArray));
    w = zeros(size(reactionArray));
  end
  
  % --- time independent calculations (also performance sensitive) ---
  
  % identify current V-T reaction by evaluating the vibrational levels ( N2(X,v) + N <-> N2(X,w) + N, v-w<6 ) 
  if levelsNeedToBeIdentified(reactionID)
    v(reactionID) = str2double(reactionArray(reactionID).reactantArray(1).vibLevel);
    w(reactionID) = str2double(reactionArray(reactionID).productArray(1).vibLevel);
    % once identified the levels of the reaction set the boolean variable to false
    levelsNeedToBeIdentified(reactionID) = false;
  end
  
  % --- regular calculations ---
  
  % local copies and definitions of different parameters used in the function
  Tg = workCond.gasTemperature;
  
  % evaluate rate coefficient
  rateCoeff = 0;
  if v(reactionID)-w(reactionID)<6 && v(reactionID)>6
    % Constants for reactive reactions
    A0r=2.2116e-12/(Tg^1.4261);
    A1r=32102.83575e0/(Tg^0.81473);
    A2r=252050.05679e0/(Tg^1.044);
    % next three lines contain old values for DC comparison
%     A0r=26.1e-16*(Tg/500)^1.5; 
%     A1r=251*sqrt(500/Tg);
%     A2r=780*sqrt(500/Tg);
    rateCoeff = rateCoeff + A0r*exp(-A1r/v(reactionID)+A2r/(v(reactionID)*v(reactionID))); % reactive contribution
    if v(reactionID)>8
      % non-reactive contribution
      A0nr=9.2403e-12/(Tg^1.63471);
      A1nr=18229.37525/(Tg^0.69843);
      A2nr=9892.96628/(Tg^0.43756);
      % next three lines contain old values for DC comparison
%       A0nr=3.16e-16*(Tg/500)^1.5;
%       A1nr=227.2*sqrt(500/Tg);
%       A2nr=625*sqrt(500/Tg);
      rateCoeff = rateCoeff + A0nr*exp(-A1nr/v(reactionID)+A2nr/(v(reactionID)*v(reactionID))); % non-reactive contribution
    end
  end
  
  % set function dependencies
  dependent = dependentInfo;
  
end