function [rateCoeff, dependent] = nitrogenMolecularVV(~, ~, ~, reactionArray, reactionID, ~, workCond, ~, ~)
% nitrogenMolecularVV evaluates the rate coefficient for V-V transitions between nitrogen molecules described in
% J Loureiro and C M Ferreira, J. Phys. D: Appl. Phys. 19 (1986) 17-35
% (CHANGE!!!!)
% http://dx.doi.org/10.1088/0022-3727/19/1/007

  % persistent variables for performance reasons
  persistent dependentInfo;
  persistent kb;
  persistent Lmin;
  persistent M;
  persistent omega;
  persistent chi;
  persistent levelsNeedToBeIdentified;
  persistent v;
  persistent w;
  
  % --- performance sensitive calculations ---
  
  % initialize variables the first time the classicalAmbipolar function is called
  if isempty(dependentInfo)
    dependentInfo = struct('onTime', false, 'onDensities', false, 'onGasTemperature', true, 'onElectronKinetics', false);
    kb = Constant.boltzmann;
    Lmin = 2e-11;                                                                   % Minimum distance between N2 molecules during V-V collision (m)
    M = reactionArray(reactionID).reactantArray(1).gas.mass;                        % Mass of N2 (Kg)
    omega = reactionArray(reactionID).reactantArray(1).gas.harmonicFrequency;       % Harmonic frequency of the oscillator N2 (rad/s)
    chi = reactionArray(reactionID).reactantArray(1).gas.anharmonicFrequency/omega; % Anharmonicity of the oscillator N2
    levelsNeedToBeIdentified = true(size(reactionArray));
    v = zeros(size(reactionArray));
    w = zeros(size(reactionArray));
  end
  
  % --- time independent calculations (also performance sensitive) ---
  
  if levelsNeedToBeIdentified(reactionID)
    % identify current V-V reaction by evaluating the vibrational levels ( v + w-1 -> v-1 + w )
    if reactionArray(reactionID).reactantStoiCoeff(1) == 2
      v(reactionID) = str2double(reactionArray(reactionID).reactantArray(1).vibLevel);
      w(reactionID) = v(reactionID)+1;
    elseif reactionArray(reactionID).productStoiCoeff(1) == 2
      w(reactionID) = str2double(reactionArray(reactionID).productArray(1).vibLevel);
      v(reactionID) = w(reactionID)+1;
    else
      levels = zeros(2);
      levels(1,1) = str2double(reactionArray(reactionID).reactantArray(1).vibLevel);
      levels(2,1) = str2double(reactionArray(reactionID).reactantArray(2).vibLevel);
      levels(1,2) = str2double(reactionArray(reactionID).productArray(1).vibLevel);
      levels(2,2) = str2double(reactionArray(reactionID).productArray(2).vibLevel);
      if levels(1,2)-levels(1,1)==1 && levels(2,2)-levels(2,1)==-1
        w(reactionID) = levels(1,2);
        v(reactionID) = levels(2,1);
      elseif levels(2,2)-levels(1,1)==1 && levels(1,2)-levels(2,1)==-1
        w(reactionID) = levels(2,2);
        v(reactionID) = levels(2,1);
      elseif levels(1,2)-levels(1,1)==-1 && levels(2,2)-levels(2,1)==1
        v(reactionID) = levels(1,1);
        w(reactionID) = levels(2,2);
      elseif levels(2,2)-levels(1,1)==-1 && levels(1,2)-levels(2,1)==1
        v(reactionID) = levels(1,1);
        w(reactionID) = levels(1,2);
      else  % check for one quanta jump between vibrational levels
        error(['nitrogenMolecularVV can not evaluate the rate coefficient for the reaction:\n%s\n' ...
          'The reaction does not comply with the following structure:\n v + w-1 -> v-1 + w; for w<v+1\n'], ...
          reactionArray(reactionID).description);
      end
    end
    if w(reactionID)>v(reactionID)  % check the directionality of the reaction
      error(['nitrogenMolecularVV can not evaluate the rate coefficient for the reaction:\n%s\n' ...
        'The reaction does not comply with the following structure:\n v + w-1 -> v-1 + w; for w<v+1\n'], ...
        reactionArray(reactionID).description);
    end
    % once identified the levels of the reaction set the boolean variable to false
    levelsNeedToBeIdentified(reactionID) = false;
  end
  
  % --- regular calculations ---
  
  % local copy of gas temperature
  Tg = workCond.gasTemperature;
  % intermediate calculations
  Y=pi*Lmin*omega*chi*sqrt(M/(kb*Tg))*(v(reactionID)-w(reactionID));
%   Y=9.97*sqrt(1/Tg)*(v-w); % LoKI1.2.0 expression
  if Y<=20
    F=0.5*(3-exp(-Y/1.5))*exp(-Y/1.5);
  else
    F=8*sqrt(pi/3)*Y^(7/3)*exp(-3*Y^(2/3));
%     F=8.1845*Y^(7/3)*exp(-3*Y^(2/3)); % LoKI1.2.0 expression
  end
  % Billing corrections
  if v(reactionID) < 10 % LoKI1.2.0 uses <= here (error check paper in the header)
    kor=39.0625-1.5625*v(reactionID);
  else
    kor=25.2+24.1*((v(reactionID)-10)/30)^3;
  end
  
  % evaluate rate coefficient
  rateCoeff=(v(reactionID)/(1-chi*v(reactionID)))*(w(reactionID)/(1-chi*w(reactionID)))*(1-chi)^2*F*6.354e-23*Tg^(1.5)/kor;
%   rateCoeff=(v/(1-chi*v))*(w/(1-chi*w))*F*6.354e-23*Tg^(1.5)/kor; % LoKI1.2.0 expression (error, missing (1-chi)^2 factor, check paper in the header)
  
  % set function dependencies
  dependent = dependentInfo; 
  
end
