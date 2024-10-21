function [rateCoeff, dependent] = LiebermanDiff(time, densitiesAll, totalGasDensity, reactionArray, reactionID, ...
  stateArray, workCond, eTransProp, ~)
% LiebermanDiff evaluates the diffusion rate coefficient of a particular positive ion,
% for a plasma with multiple positive ions and a single negative ion with low/intermediate density,
% at low/high pressure
% using the transport model described in
% Adriana Annusova et al 2018 Plasma Sources Sci. Technol. 27 045006
% https://doi.org/10.1088/1361-6595/aab47d

% more references used throughout:
%Chabert 2016 http://dx.doi.org/10.1088/0963-0252/25/2/025010)
%Thorsteinsson 2010 (first paper) hhttp://dx.doi.org/10.1088/0963-0252/19/1/015001
%Thorsteinsson 2010 (second paper) http://dx.doi.org/10.1088/0963-0252/19/5/055008

%1e-18 factor coming with both redR and redL comes from R/lambda and L/lambda terms, 
%where lambda = 1/(totalGasDensity*sigma), sigma = 1e-18

  persistent firstBesselZero;
  persistent firstOrderBessel;
  persistent positiveIonIDs;
  persistent negativeIonID;
  persistent positiveIonDensitySum;
  persistent negativeIonDensity;
  persistent molecIonID;
  persistent atomIonID;
  persistent previousTime;
  persistent parArray;
  persistent IDk62;
  persistent IDk63;
  persistent hL;
  persistent L;
  persistent R;
  persistent kB;
  persistent alpha0;
  persistent tempRatio;
  persistent alphaS;
  persistent elecCharge;
  persistent elecTemperature;
  persistent redL;
  persistent redR;
  persistent gasTemperature;
  persistent kRec;

  % --- performance sensitive calculations ---

  % initialize variables the first time the LiebermanDiff function is called
  if isempty(firstBesselZero)
    % evaluate the first zero of the zero order bessel function and first order bessel function at that point
    firstBesselZero = fzero(@(x) besselj(0,x), [2.4 2.5]);
    firstOrderBessel = besselj(1,firstBesselZero);
    % save local copies of constants
    kB = Constant.boltzmann;
    elecCharge = Constant.electronCharge;
    % save local copies of chamber dimensions (with error checking)
    L = workCond.chamberLength;
    R = workCond.chamberRadius;
    % initialize previousTime variable
    previousTime = -1;
    %parameters for the numerical solution found for the electronegativity (alphaS in code) at the sheat edge 
    %as done in Thorsteinsson 2010 (first paper)
    parArray = [0.607, 5.555, -11.16, 1.634, 12*1e-3, -107*1e-3];
    % evaluate the IDs of the positive ions and negative ion (singly ionized and gas phase)
    for i = 1:length(stateArray)
      state = stateArray(i);
      if strcmp(state.ionCharg, '+') && state.isVolumeSpecies
        positiveIonIDs(end+1) = state.ID;
        if strcmp(state.name, 'O2(+,X)')
          molecIonID = state.ID;
        elseif strcmp(state.name, 'O(+,gnd)')
          atomIonID = state.ID;
        end
      elseif strcmp(state.ionCharg, '-')  && state.isVolumeSpecies
        negativeIonID(end+1) = state.ID;
      end
    end
    % looking for k62 and k63 (neutralization of positive atomic and molecular ions) as defined in Annusova 2018
    for ID=1:length(reactionArray)
      if isempty(IDk62) && strcmp(reactionArray(ID).description, 'O(+,gnd) + O(-,gnd) -> 2O(3P)')
        IDk62 = ID;
      elseif isempty(IDk63) && strcmp(reactionArray(ID).description, 'O2(+,X) + O(-,gnd) -> O2(X) + O(3P)')
        IDk63 = ID;
      end
      if ~isempty(IDk62) && ~isempty(IDk63)
        break
      end
    end
    % different error checks before proceding with calculations
    if L==0
      error('Cant do Lieberman Diffusion with L=0, at the moment.')
    elseif R==0
      error('Cant do Lieberman Diffusion with R=0, at the moment.')
    elseif isempty(negativeIonID)
      error('There is no negative ion to include in the diffusion model LiebermanDiff');
    elseif length(negativeIonID) > 1
      negativeIonNames = '';
      for i=1:length(negativeIonID)
        negativeIonNames = [negativeIonNames ', ' stateArray(negativeIonID(i)).name];
      end
      error('The diffusion model in LiebermanDiff is not valid for more than one negative ion (%s)', ...
        negativeIonNames(3:end));
    elseif isempty(IDk62)
      error('Reaction ''O(+,gnd) + O(-,gnd) -> 2O(3P)'' in LiebermanDiff was not found');
    elseif isempty(IDk63)
      error('Reaction ''O2(+,X) + O(-,gnd) -> O2(X) + O(3P)'' in LiebermanDiff was not found');
    end 
  end

  % --- calculations that are equal for every "LiebermanDiff" reaction (at a given time) ---
  
  if previousTime ~= time
    % actualize previousTime variable
    previousTime = time;
    
    % evaluate temperatures (assuming both positive and negative ions temperatures equal to gas temperature)
    elecTemperature = workCond.electronTemperature*elecCharge/kB; %in K  
    gasTemperature = workCond.gasTemperature; % in K
    tempRatio = elecTemperature/gasTemperature; % gamma
    % (ansatz and numerical sol for alphaS from Thorsteinsson 2010 (1st and 2nd paper))
    % only works for tempRatio > 10 (gamma > 10)!!
    if tempRatio <= 10
      error(['Ratio between electron temperature and gas temperature should be > 10 for high pressure regime ' ...
        'calculations in LiebermanDiff']);
    end
    
    % lambda = 1/(gasDens*1e-18); % mean free path (assuming same cross section for all processes? assuming only collisions
    % between ions and neutrals?) THIS COMES FROM A CONSTANT MEAN FREE PATH REGIME TALKED ABOUT IN CHABERT 2016

    redR = totalGasDensity*R;
    redL = totalGasDensity*L;
    
    % density of negative ion
    negativeIonDensity = densitiesAll(negativeIonID);
    if isnan(negativeIonDensity)
      error('neg ion dens is NaN')
    end
    electroNeg = negativeIonDensity/workCond.electronDensity;
    alpha0 = 1.5*electroNeg;

    % density of positive ions
    positiveIonDensitySum = 0;
    for i = 1:length(positiveIonIDs)
      positiveIonDensity = densitiesAll(positiveIonIDs(i));
      positiveIonDensitySum = positiveIonDensitySum + positiveIonDensity;
    end
    densAtomPositiveIon = densitiesAll(atomIonID); % density of positive atomic ion
    densMolPositiveIon = densitiesAll(molecIonID); % density of positive molecular ion

    % calculating axial edge-to-center positive ion density ratios (corrections for low pressure by Chabert 2016)   
    hL = 0.86*(3 + redL*1e-18/2 + sqrt(1 + electroNeg)*(redL*1e-18)^2/(5*tempRatio))^(-0.5)*((tempRatio - 1)/...
      (tempRatio*(1+electroNeg)^2) + 1/tempRatio)^(0.5);
    
    rho = abs(electroNeg + parArray(5)*(exp(parArray(6)*(tempRatio-50))-1));
    alphaS = electroNeg*((parArray(1)*erf(parArray(2)*rho + parArray(3))*exp(-parArray(4)/rho^(1.35)))/...
      (exp((tempRatio-1)/2*tempRatio - 0.49))); % expression from Thorsteinsson 2010 (first paper)
    
    % evaluate weighted recombination rate coefficient
    if (densAtomPositiveIon == 0) && (densMolPositiveIon == 0)
      kRec = 0;
    else
      k62 = reactionArray(IDk62).rateCoeffFuncHandle(time, densitiesAll, totalGasDensity, reactionArray, IDk62, ...
        stateArray, workCond, eTransProp,reactionArray(IDk62).rateCoeffParams);
      k63 = reactionArray(IDk63).rateCoeffFuncHandle(time, densitiesAll, totalGasDensity, reactionArray, IDk63, ...
        stateArray, workCond, eTransProp,reactionArray(IDk63).rateCoeffParams);
      kRec = (densAtomPositiveIon*k62 + densMolPositiveIon*k63)/(densAtomPositiveIon+densMolPositiveIon);
    end
  end

  % --- regular calculations ---
  
  redDi = reactionArray(reactionID).reactantArray.evaluateReducedDiffCoeff(workCond); 
  redDa = redDi*(1+tempRatio*(1+alphaS))/(1+tempRatio*alphaS);
  ionMass = reactionArray(reactionID).reactantArray(1).mass; 
  bohmVel = sqrt(kB*elecTemperature/ionMass); % m/s
  meanVel = sqrt(8*kB*gasTemperature/(pi*ionMass));
  
  if kRec == 0
    hC = 0;
  else 
    nAst = (15/56)*meanVel*totalGasDensity*1e-18/kRec;
    hC = (sqrt(tempRatio)*(1+sqrt(nAst)*positiveIonDensitySum/sqrt(negativeIonDensity^3)))^(-1);
  end
  
  %if it is to be done with the corrections like in Annusova 2018, comment the block of if L~=0
  %and uncomment the equivalent one some lines above
  %if L ~= 0
  %  %h0L = 0.86*(3 + redL*1e-18/2 + (0.86*redL*bohmVel/(pi*redDa))^2)^(-0.5);
  %  hL = ((h0L/(1+alpha0))^2 + hC^2)^(0.5);
  %else
  %    error('Cant do Lieberman Diffusion with L=0, at the moment.')
  %end 

  %calculating radial edge-to-center positive ion density ratios 
  h0R = 0.8*(4 + redR*1e-18 + (0.8*redR*bohmVel/(firstBesselZero*firstOrderBessel*redDa))^2)^(-0.5);
  hR = ((h0R/(1+alpha0))^2 + hC^2)^(0.5);

  rateCoeff = 2*bohmVel*(hL/L + hR/R);

  % set function dependencies
  dependent = struct('onTime', false, 'onDensities', true, 'onGasTemperature', true, 'onElectronKinetics', true);
end
