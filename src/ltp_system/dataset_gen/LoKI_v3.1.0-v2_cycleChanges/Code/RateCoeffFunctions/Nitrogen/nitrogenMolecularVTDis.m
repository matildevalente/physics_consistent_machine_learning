function [rateCoeff, dependent] = nitrogenMolecularVTDis(~, ~, ~, reactionArray, reactionID, ~, workCond, ~, ~)
% nitrogenMolecularVTDis evaluates the dissociation rate coefficient for V-T transitions between nitrogen molecules 
% described in:
% L.L. Alves, L. Marques, C.D. Pintassilgo, G. Wattieaux, E. Es-sebbar, J. Berndt, et al., Capacitively coupled 
% radio-frequency discharges in nitrogen at low pressures, Plasma Sources Sci. Technol. 21 (2012) 45008.
% http://dx.doi.org/10.1088/0963-0252/21/4/045008

  % persistent variables for performance reasons
  persistent dependentInfo;
  persistent kb;
  persistent hbar;
  persistent Lmin;
  persistent M;
  persistent omega;
  persistent chi;
  persistent NLevels;
  
  % --- performance sensitive calculations ---
  
  % initialize variables the first time the classicalAmbipolar function is called
  if isempty(dependentInfo)
    dependentInfo = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', false);
    kb = Constant.boltzmann;
    hbar = Constant.planckReduced;
    Lmin = 2e-11;                                                                   % Minimum distance between N2 molecules during V-V collision (m)
    M = reactionArray(reactionID).reactantArray(1).gas.mass;                        % Mass of N2 (Kg)
    omega = reactionArray(reactionID).reactantArray(1).gas.harmonicFrequency;       % Harmonic frequency of the oscillator N2 (rad/s)
    chi = reactionArray(reactionID).reactantArray(1).gas.anharmonicFrequency/omega; % Anharmonicity of the oscillator N2
    % identify current V-T reaction by evaluating the vibrational levels ( N2(X,Nlevels-1) + N2 <-> 2N + N2) 
    NLevels = str2double(reactionArray(reactionID).reactantArray(1).vibLevel)+1;
  end
  
  % --- regular calculations ---
  
  % local copies and definitions of different parameters used in the function
  Tg = workCond.gasTemperature;
  
  % intermediate calculations
  Y=0.5*pi*Lmin*omega*sqrt(M/(kb*Tg))*(1-2*chi*NLevels);
%   Y=809.708*(1-2*NLevels*chi)/sqrt(Tg); % LoKI1.2.0 expression
  if Y<=20
    F=0.5*(3-exp(-Y/1.5))*exp(-Y/1.5);
  else
    F=8*sqrt(pi/3)*Y^(7/3)*exp(-3*Y^(2/3));
%     F=8.1845*Y^(7/3)*exp(-3*Y^(2/3)); % LoKI1.2.0 expression
  end
  % Billing corrections
  kor=0.2772*Tg-80.32+35.5*((NLevels-1)/39)^0.8;  
  
  % evaluate rate coefficient
  rateCoeff =NLevels*(1-chi)/(1-chi*NLevels)*F*1.07e-18*Tg^(1.5)/kor*exp(-hbar*omega*(1-2*chi*NLevels)/(kb*Tg));
  
  % set function dependencies
  dependent = dependentInfo;
  
end
