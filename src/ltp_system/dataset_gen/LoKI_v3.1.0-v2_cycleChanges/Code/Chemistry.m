% LoKI-B solves a time and space independent form of the two-term 
% electron Boltzmann equation (EBE), for non-magnetised non-equilibrium 
% low-temperature plasmas excited by DC/HF electric fields from 
% different gases or gas mixtures.
% Copyright (C) 2018 A. Tejero-del-Caz, V. Guerra, D. Goncalves, 
% M. Lino da Silva, L. Marques, N. Pinhao, C. D. Pintassilgo and
% L. L. Alves
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.

classdef Chemistry < handle
  %Chemistry Class that solves the heavy species rate balance equations under certain conditions to
  %time evoluation of their densities.
  %
  %   Chemistry uses an stiff ODE solver (ode15s Matlab solver) to solve the rate balance equations corresponding
  %   to the set of reactions specified in the ".chem" files provided by the user in the setup file. 
  

  properties

    gasArray = ChemGas.empty;             % handle array to the gas mixture (to perform operations in a per gas basis)
    stateArray = ChemState.empty;         % handle array to the states (species) taken into account
    reactionArray = Reaction.empty;       % handle array to the reactions taken into account
    workCond = WorkingConditions.empty;   % handle to the working conditions of the simulation
    electronKinetics = [];                % handle to the electron kinetics object of the simulation
    
    includeThermalModel = false;          % boolean to select whether the thermal model is active or not
    thermalModelBoundary = '';            % wall or external (location of the model boundary condition)
    thermalModelWallFraction = 0;         % fraction of power released at the wall that heats the plasma (see fw parameter at the documentation)
    intConvCoeff = [];                    % internal convection coefficient (of the cylindrical wall)
    extConvCoeff = [];                    % external convection coefficient (of the cylindrical wall)

    solveEedf = false;                    % boolean to select whether to solve the EEDF or not during post-discharge (WIP)
    
    gasesIDsToUpdateInElectronKinetics = [];
    statesIDsToUpdateInElectronKinetics = [];
    
    electronTransportProperties = struct('reducedDiffCoeff', [], 'reducedMobility', []);
    numberOfSpecies;
    initialDensities;
    gasIDs;
    childIDs;
    volumePhaseSpeciesIDs;
    surfacePhaseSpeciesIDs;
    
    numberOfReactions;
    rateCoeffFuncHandles;
    rateCoeffParams;
    reactantElectrons;
    reactantIDs;
    reactantStoiCoeffs;
    catalystIDs;
    catalystStoiCoeffs;
    productElectrons;
    productIDs;
    productStoiCoeffs;
    gasStabilisedReactionIDs;
    transportReactionIDs;
    volumeReactionIDs;
    
    odeSolver;
    odeOptions;
    odeSteadyStateTime = [];
    odePostDischargeTime = [];
    
    neutralityRelError = [];
    neutralityMaxIterations = [];
    neutralityIterationCurrent = [];
    neutralityRelErrorCurrent = [];    
    globalRelError = [];
    globalMaxIterations = [];
    globalIterationCurrent = [];
    globalRelErrorCurrent = [];
    elecDensityRelError = [];
    elecDensityMaxIterations = [];
    elecDensityIterationCurrent = [];
    elecDensityRelErrorCurrent = [];
    dischargeConditionForElecDensityCycle = [];

    solution;
    
  end
  
  events
    genericStatusMessage;
    newNeutralityCycleIteration;
    newGlobalCycleIteration;
    newElecDensityCycleIteration;    
    obtainedNewChemistrySolution;
  end

  methods (Access = public)

    function chemistry = Chemistry(setup)
      
      % store configuration of the thermal model
      chemistry.includeThermalModel = setup.info.chemistry.thermalModel.isOn;
      if chemistry.includeThermalModel
        chemistry.thermalModelBoundary = setup.info.chemistry.thermalModel.boundary;
        switch chemistry.thermalModelBoundary
          case 'wall'
            chemistry.intConvCoeff = setup.info.chemistry.thermalModel.intConvCoeff;
            chemistry.thermalModelWallFraction = setup.info.chemistry.thermalModel.wallFraction;
          case 'external'
            chemistry.intConvCoeff = setup.info.chemistry.thermalModel.intConvCoeff;
            chemistry.extConvCoeff = setup.info.chemistry.thermalModel.extConvCoeff;
        end
      end
      
      % store the gas array (in order to be able to perform operations in a per gas basis)
      chemistry.gasArray = setup.chemistryGasArray;
      % find and store IDs of gases with electron kinetics equivalents that needs to be updated
      gasesIDsToUpdateInElectronKinetics = [];
      for i = 1:length(chemistry.gasArray)
        gas = chemistry.gasArray(i);
        if ~isempty(gas.eedfEquivalent) && ~isempty(gas.eedfEquivalent.collisionArray)
          gasesIDsToUpdateInElectronKinetics(end+1) = i;
        end
      end
      chemistry.gasesIDsToUpdateInElectronKinetics = gasesIDsToUpdateInElectronKinetics;
      
      % store the state array (with all the species considered in the chemistry's gas mixture)
      chemistry.stateArray = setup.chemistryStateArray;
      
      % store the reaction array (with all the reactions considered in the chemistry)
      chemistry.reactionArray = setup.chemistryReactionArray;
      
      % store working conditions and add corresponding listeners (if needed)
      chemistry.workCond = setup.workCond;
      
      % store electron kinetics object in case it is enabled
      if setup.enableElectronKinetics
        chemistry.electronKinetics = setup.electronKinetics;
        % find and store IDs of states with electron kinetics equivalents that needs to be updated
        statesIDsToUpdateInElectronKinetics = [];
        for i = 1:length(chemistry.stateArray)
          state = chemistry.stateArray(i);
          if ~isempty(state.eedfEquivalent) && state.eedfEquivalent.isTarget
            statesIDsToUpdateInElectronKinetics(end+1) = i;
          end
        end
        chemistry.statesIDsToUpdateInElectronKinetics = statesIDsToUpdateInElectronKinetics;
      end
      
      % store electron transport properties (in case they are specified in the setup file)
      if isfield(setup.info.chemistry, 'electronProperties')
        eTransProp.reducedDiffCoeff = setup.info.chemistry.electronProperties.reducedDiffCoeff;
        eTransProp.reducedMobility = setup.info.chemistry.electronProperties.reducedMobility;
        chemistry.electronTransportProperties = eTransProp;
      end
      
      % convert species info into simple and fast numeric/cell arrays (ChemState objects -> arrays)
      numberOfSpecies = length(chemistry.stateArray);
      initialDensities = zeros(1, numberOfSpecies);
      childIDs = cell(1, numberOfSpecies);
      gasIDs = zeros(1, numberOfSpecies);
      volumePhaseSpeciesIDs = [];
      surfacePhaseSpeciesIDs = [];
      for i = 1:numberOfSpecies
        initialDensities(i) = chemistry.stateArray(i).density;
        gasIDs(i) = chemistry.stateArray(i).gas.ID;
        if ~isempty(chemistry.stateArray(i).childArray)
          childIDs{i} = [chemistry.stateArray(i).childArray.ID];
        end
        if chemistry.stateArray(i).isVolumeSpecies
          volumePhaseSpeciesIDs(end+1) = i;
        else
          surfacePhaseSpeciesIDs(end+1) = i;
        end
      end
      chemistry.numberOfSpecies = numberOfSpecies;
      chemistry.initialDensities = initialDensities;
      chemistry.gasIDs = gasIDs;
      chemistry.childIDs = childIDs;
      chemistry.volumePhaseSpeciesIDs = volumePhaseSpeciesIDs;
      chemistry.surfacePhaseSpeciesIDs = surfacePhaseSpeciesIDs;
      
      % convert reactions info into simple and fast numeric/cell arrays (Reaction objects -> arrays)
      numberOfReactions = length(chemistry.reactionArray);
      rateCoeffFuncHandles = cell(1, numberOfReactions);
      rateCoeffParams = cell(1, numberOfReactions);
      reactantElectrons = zeros(1, numberOfReactions);
      reactantIDs = cell(1, numberOfReactions);
      reactantStoiCoeffs = zeros(numberOfSpecies, numberOfReactions);
      catalystIDs = cell(1, numberOfReactions);
      catalystStoiCoeffs = zeros(numberOfSpecies, numberOfReactions);
      productElectrons = zeros(1, numberOfReactions);
      productIDs = cell(1, numberOfReactions);
      productStoiCoeffs = zeros(numberOfSpecies, numberOfReactions);
      gasStabilisedReactionIDs = [];
      transportReactionIDs = [];
      volumeReactionIDs = [];
      for i = 1:numberOfReactions
        rateCoeffFuncHandles{i} = chemistry.reactionArray(i).rateCoeffFuncHandle;
        rateCoeffParams{i} = chemistry.reactionArray(i).rateCoeffParams;
        reactantElectrons(i) = chemistry.reactionArray(i).reactantElectrons;
        reactantIDs{i} = [chemistry.reactionArray(i).reactantArray.ID];
        reactantStoiCoeffs(reactantIDs{i}, i) = chemistry.reactionArray(i).reactantStoiCoeff;
        catalystIDs{i} = [chemistry.reactionArray(i).catalystArray.ID];
        catalystStoiCoeffs(catalystIDs{i}, i) = chemistry.reactionArray(i).catalystStoiCoeff;
        productElectrons(i) = chemistry.reactionArray(i).productElectrons;
        productIDs{i} = [chemistry.reactionArray(i).productArray.ID];
        productStoiCoeffs([productIDs{i}], i) = chemistry.reactionArray(i).productStoiCoeff;
        if chemistry.reactionArray(i).isGasStabilised
          gasStabilisedReactionIDs(end+1) = i;
        end
        if chemistry.reactionArray(i).isTransport
          transportReactionIDs(end+1) = i;
        else
          volumeReactionIDs(end+1) = i;
        end
      end
      chemistry.numberOfReactions = numberOfReactions;
      chemistry.rateCoeffFuncHandles = rateCoeffFuncHandles;
      chemistry.rateCoeffParams = rateCoeffParams;
      chemistry.reactantElectrons = reactantElectrons;
      chemistry.reactantIDs = reactantIDs;
      chemistry.reactantStoiCoeffs = sparse(reactantStoiCoeffs);
      chemistry.catalystIDs = catalystIDs;
      chemistry.catalystStoiCoeffs = sparse(catalystStoiCoeffs);
      chemistry.productElectrons = productElectrons;
      chemistry.productIDs = productIDs;
      chemistry.productStoiCoeffs = sparse(productStoiCoeffs);
      chemistry.gasStabilisedReactionIDs = gasStabilisedReactionIDs;
      chemistry.transportReactionIDs = transportReactionIDs;
      chemistry.volumeReactionIDs = volumeReactionIDs;
      
      % store configuration of the ODE solver
      chemistry.odeSolver = str2func(setup.info.chemistry.timeIntegrationConf.odeSolver);
      chemistry.odeSteadyStateTime = setup.info.chemistry.timeIntegrationConf.steadyStateTime;
      chemistry.odePostDischargeTime = setup.info.chemistry.timeIntegrationConf.postDischargeTime;
      options = odeset();
      for parameter = fields(options)'
        if isfield(setup.info.chemistry.timeIntegrationConf, 'odeSetParameters') && ...
            isfield(setup.info.chemistry.timeIntegrationConf.odeSetParameters, parameter{1})
          options.(parameter{1}) = setup.info.chemistry.timeIntegrationConf.odeSetParameters.(parameter{1});
        else
          options.(parameter{1}) = [];
        end
      end
      options.NonNegative = 1:numberOfSpecies+1;    % ensure non negative values for the solution (densities & temperature)
%       options.OutputFcn = @odeProgressBar;          % activate only for debugging purposes 
      chemistry.odeOptions = options;
      
      % store configuration about the iterations schemes (quasineutrality, macroscopic and electron density)
      chemistry.neutralityRelError = setup.info.chemistry.iterationSchemes.neutralityRelError;
      chemistry.neutralityMaxIterations = setup.info.chemistry.iterationSchemes.neutralityMaxIterations;
      chemistry.elecDensityRelError = setup.info.chemistry.iterationSchemes.elecDensityRelError;
      chemistry.elecDensityMaxIterations = setup.info.chemistry.iterationSchemes.elecDensityMaxIterations;      
      if isempty(chemistry.electronKinetics)
        chemistry.globalRelError = [];
        chemistry.globalMaxIterations = 1;
      else
        chemistry.globalRelError = setup.info.chemistry.iterationSchemes.globalRelError;
        chemistry.globalMaxIterations = setup.info.chemistry.iterationSchemes.globalMaxIterations;
      end

      chemistry.dischargeConditionForElecDensityCycle = setup.info.chemistry.iterationSchemes.dischargeConditionForElecDensityCycle;
      
    end
    
    function solve(chemistry)

      eTransProp = chemistry.electronTransportProperties;
      if ~isempty(chemistry.electronKinetics)
        eTransProp.reducedDiffCoeff = chemistry.electronKinetics.swarmParam.redDiffCoeff;
        eTransProp.reducedMobility = chemistry.electronKinetics.swarmParam.redMobility;
      end
      
      % thresholds and other data for cycles
      maxElecDensityRelError = chemistry.elecDensityRelError;
      maxElecDensityIterations = chemistry.elecDensityMaxIterations;
      currentElecDensityIteration = 1;          
      maxGlobalRelError = chemistry.globalRelError;
      maxGlobalIterations = chemistry.globalMaxIterations;
      currentGlobalIteration = 1;  
      maxNeutralityRelError = chemistry.neutralityRelError;
      maxNeutralityIterations = chemistry.neutralityMaxIterations;
      currentNeutralityIteration = 1;    

      % local copy of different variables
      surfaceSiteDensity = chemistry.workCond.surfaceSiteDensity;
      volPhaseSpeciesIDs = chemistry.volumePhaseSpeciesIDs;
      surPhaseSpeciesIDs = chemistry.surfacePhaseSpeciesIDs;
      
      % save/set initial temperatures and initial value of the gas temperature used in the electron kinetics module
      initialGasTemperature = chemistry.workCond.gasTemperature;
      if chemistry.includeThermalModel
        if isempty(chemistry.workCond.nearWallTemperature)
          chemistry.workCond.nearWallTemperature = initialGasTemperature;
        end
        if strcmp(chemistry.thermalModelBoundary, 'external') && isempty(chemistry.workCond.wallTemperature)
          chemistry.workCond.wallTemperature = chemistry.workCond.extTemperature;
        end
        initialNearWallTemperature = chemistry.workCond.nearWallTemperature;
        initialWallTemperature = chemistry.workCond.wallTemperature;
      end

      % evaluate initial densities of species (volume phase and surface phase)
      initialGasPressure = chemistry.workCond.gasPressure;
      initialGasDensity = initialGasPressure/(Constant.boltzmann*initialGasTemperature);
      initialAbsDensities = zeros(1, chemistry.numberOfSpecies);
      initialAbsDensities(volPhaseSpeciesIDs) = initialGasDensity.*...
        chemistry.initialDensities(volPhaseSpeciesIDs);
      initialAbsDensities(surPhaseSpeciesIDs) = (chemistry.workCond.areaOverVolume*surfaceSiteDensity).*...
        chemistry.initialDensities(surPhaseSpeciesIDs);  

      electronKineticsGasTemperature = initialGasTemperature;
      
      % ----- ELECTRON DENSITY CYCLE ----
      currentElecDensityRelError = [];
      elecDensityRelErrorAll = [];
      elecDensityIterationAll = [];
      elecDensityAll = [];
      while currentElecDensityIteration<=maxElecDensityIterations

        % ----- MACROSCOPIC CYCLE ----
        currentGlobalRelError = [];
        while currentGlobalIteration<=maxGlobalIterations
          
          % ----- NEUTRALITY CYCLE -----
          currentNeutralityRelError = [];
          neutralityRelErrorAll = [];
          neutralityIterationAll = [];
          excitationParameterAll = [];
          while currentNeutralityIteration<=maxNeutralityIterations
            
            % set initial value for the temperatures
            chemistry.workCond.gasTemperature = initialGasTemperature;
            if chemistry.includeThermalModel
              chemistry.workCond.nearWallTemperature = initialNearWallTemperature;
              if strcmp(chemistry.thermalModelBoundary, 'external')
                chemistry.workCond.wallTemperature = initialWallTemperature;
              end
            end
            
            % evaluate initial values for the rate coefficients saving IDs of time dependent reactions
            directRateCoeffs = zeros(1, chemistry.numberOfReactions);
            inverseRateCoeffs = zeros(1, chemistry.numberOfReactions);
            timeDependentReactionIDs = [];
            for reactionID = 1:length(directRateCoeffs)
              [directRateCoeffs(reactionID), dependent] = chemistry.rateCoeffFuncHandles{reactionID}(0, ...
                initialAbsDensities', initialGasDensity, chemistry.reactionArray, reactionID, ...
                chemistry.stateArray, chemistry.workCond, eTransProp, chemistry.rateCoeffParams{reactionID});
              if dependent.onTime || dependent.onDensities || ...
                  (chemistry.includeThermalModel && dependent.onGasTemperature)
                timeDependentReactionIDs(end+1) = reactionID;
              end
              if chemistry.reactionArray(reactionID).isReverse
                inverseRateCoeffs(reactionID) = detailedBalance(chemistry.reactionArray(reactionID), ...
                  directRateCoeffs(reactionID), Constant.boltzmannInEV*chemistry.workCond.gasTemperature);
              end
            end

            % pre-calculate the additional heat for oxygen when the vibrations are not considered
            global O2XStateID sumO2XVibRateCoeffsXEnthalpies;
            for state = chemistry.stateArray
              if strcmp(state.name,'O2(X)') && isempty(state.childArray)
                O2XStateID = state.ID;
              end  
            end  
            if ~isempty(O2XStateID)
              pattern = 'e\+O2\(X,v=0\)<->e\+O2\(X,v=\d+\),Vibrational';
              sumO2XVibRateCoeffsXEnthalpies = 0;
              for collision = chemistry.stateArray(O2XStateID).eedfEquivalent.gas.collisionArrayExtra
                matches = regexp(collision.description, pattern, 'match');
                if ~isempty(matches)
                  sumO2XVibRateCoeffsXEnthalpies = sumO2XVibRateCoeffsXEnthalpies +  collision.ineRateCoeff*collision.threshold;
                end  
              end  
            end  
            
            % call the ODE solver
            start = tic;
            notify(chemistry, 'genericStatusMessage', ...
              StatusEventData('\t- Integrating particle rate-balance equations ...\n', 'status'));
            [time, timeSolution] = chemistry.odeSolver(@kinetics, [0 chemistry.odeSteadyStateTime], ...
              [initialAbsDensities initialGasTemperature], chemistry.odeOptions, chemistry.workCond, ...
              chemistry.gasIDs, chemistry.childIDs, directRateCoeffs, inverseRateCoeffs, ...
              chemistry.rateCoeffFuncHandles, chemistry.rateCoeffParams, chemistry.reactantElectrons, ...
              chemistry.reactantIDs, chemistry.reactantStoiCoeffs, chemistry.catalystIDs, ...
              chemistry.catalystStoiCoeffs, chemistry.productElectrons, chemistry.productIDs, ...
              chemistry.productStoiCoeffs, chemistry.gasStabilisedReactionIDs, timeDependentReactionIDs, ...
              chemistry.reactionArray, chemistry.gasArray, chemistry.stateArray, eTransProp, ...
              chemistry.electronKinetics, false, chemistry.includeThermalModel, chemistry, false);
            str = sprintf('\\t    Finished (%f seconds).\\n', toc(start));
            notify(chemistry, 'genericStatusMessage', StatusEventData(str, 'status'));
  
            % separate time solutions into its different components
            absDensitiesTime = timeSolution(:,1:end-1);
            gasTemperatureTime = timeSolution(:,end);
            
            % restore value of the gas temperature for electron kinetics calculations (if thermal model active)
            if chemistry.includeThermalModel
              chemistry.workCond.gasTemperature = electronKineticsGasTemperature;
            end
            
            if ~isempty(chemistry.electronKinetics)
              % evaluate the neutrality relative error
              currentNeutralityRelError = 0;
              for i = 1:chemistry.numberOfSpecies
                chargStr = chemistry.stateArray(i).ionCharg;
                if ~isempty(chargStr)
                  currentNeutralityRelError = currentNeutralityRelError + ...
                    length(regexp(chargStr,'+'))*absDensitiesTime(end,i);
                  currentNeutralityRelError = currentNeutralityRelError - ...
                    length(regexp(chargStr,'-'))*absDensitiesTime(end,i);
                end
              end
              currentNeutralityRelError = (chemistry.workCond.electronDensity - currentNeutralityRelError)/...
                chemistry.workCond.electronDensity;
              neutralityIterationAll(end+1) = currentNeutralityIteration;
              neutralityRelErrorAll(end+1) = currentNeutralityRelError;
              
              % broadcast results of this iteration
              chemistry.neutralityIterationCurrent = currentNeutralityIteration;
              chemistry.neutralityRelErrorCurrent = currentNeutralityRelError;
              notify(chemistry, 'newNeutralityCycleIteration');
              
              % prepare next iteration (new solution of the boltzmann equation)
              if abs(currentNeutralityRelError)>maxNeutralityRelError
                switch class(chemistry.electronKinetics)
                  case 'Boltzmann'
                    excitationParameterAll(end+1) = chemistry.workCond.reducedField;
                    chemistry.workCond.update('reducedField', ...
                      iterateOverParameter(neutralityIterationAll, neutralityRelErrorAll, excitationParameterAll, false, true));
                  case 'PrescribedEedf'
                    excitationParameterAll(end+1) = chemistry.workCond.electronTemperature;
                    chemistry.workCond.update('electronTemperature', ...
                      iterateOverParameter(neutralityIterationAll, neutralityRelErrorAll, excitationParameterAll, false, true));
                end
                chemistry.electronKinetics.solve();
                eTransProp.reducedDiffCoeff = chemistry.electronKinetics.swarmParam.redDiffCoeff;
                eTransProp.reducedMobility = chemistry.electronKinetics.swarmParam.redMobility;
                currentNeutralityIteration = currentNeutralityIteration+1;
                if currentNeutralityIteration>maxNeutralityIterations
                  error('Maximum number of iterations reached for the neutrality cycle without convergence.')
                end
              else
                break;
              end
            else
              break;
            end
            
          end
          
          if ~isempty(chemistry.electronKinetics) && currentGlobalIteration<maxGlobalIterations
            
            % update gas temperature dependencies of the electron kinetics (in case thermal model is active)
            if chemistry.includeThermalModel
              % update gas temperature used in the electron kinetics
              chemistry.workCond.gasTemperature = gasTemperatureTime(end);
              electronKineticsGasTemperature = gasTemperatureTime(end);
              % update possible populations depending on the gas temperature
              for gas = chemistry.electronKinetics.gasArray
                for state = gas.stateArray
                  if ~isempty(state.populationFunc) && isempty(state.chemEquivalent)
                    for parameter = state.populationParams
                      if strcmp(parameter{1}, 'gasTemperature')
                        state.evaluatePopulation(chemistry.workCond);
                        break;
                      end
                    end
                  end
                end
              end
              % update the densities of states accordingly
              for gas = chemistry.electronKinetics.gasArray
                for state = gas.stateArray
                  state.evaluateDensity();
                end
              end
            end
            
            % evaluate electron kinetics (eedf) densities (final time)
            eedfAbsDensities = absDensitiesTime(end,:);
            eedfTotalGasDensity = 0;
            for i = chemistry.statesIDsToUpdateInElectronKinetics
              if isempty(chemistry.childIDs{i})
                eedfTotalGasDensity = eedfTotalGasDensity + eedfAbsDensities(i);
              else
                eedfAbsDensities(i) = 0;
                for j = intersect(chemistry.childIDs{i}, chemistry.statesIDsToUpdateInElectronKinetics)
                  if ~isempty(chemistry.childIDs{j})
                    eedfAbsDensities(j) = 0;
                    for k = intersect(chemistry.childIDs{j}, chemistry.statesIDsToUpdateInElectronKinetics)
                      eedfAbsDensities(j) = eedfAbsDensities(j) + eedfAbsDensities(k);
                    end
                  end
                  eedfAbsDensities(i) = eedfAbsDensities(i) + eedfAbsDensities(j);
                end
              end
            end
            
            % update densities of states in the the electron kinetics gas mixture (normalized to gas density)
            for i = chemistry.statesIDsToUpdateInElectronKinetics
              chemistry.stateArray(i).eedfEquivalent.density = eedfAbsDensities(i)/eedfTotalGasDensity;
            end
            
            % renormalize electron kinetics gas mixture
            for i = chemistry.gasesIDsToUpdateInElectronKinetics
              chemistry.gasArray(i).eedfEquivalent.renormalizeWithDensities();
              % check for the distribution of states to be properly normalised
              chemistry.gasArray(i).eedfEquivalent.checkPopulationNorms();
            end
            
            % update the density dependencies of the electron kinetics (this also updates gas temperature dependencies)
            chemistry.electronKinetics.updateDensityDependencies();
            
            % solve the electron kinetics with new densities
            chemistry.electronKinetics.solve();
            
            % evaluate the macroscopic relative error
            currentGlobalRelError = (eTransProp.reducedDiffCoeff/chemistry.electronKinetics.swarmParam.redDiffCoeff + ...
              eTransProp.reducedMobility/chemistry.electronKinetics.swarmParam.redMobility)/2 - 1;
            
            % broadcast results of this iteration
            chemistry.globalIterationCurrent = currentGlobalIteration;
            chemistry.globalRelErrorCurrent = currentGlobalRelError;
            notify(chemistry, 'newGlobalCycleIteration');
            
            % prepare next iteration (new solution of the boltzmann equation)
            eTransProp.reducedDiffCoeff = chemistry.electronKinetics.swarmParam.redDiffCoeff;
            eTransProp.reducedMobility = chemistry.electronKinetics.swarmParam.redMobility;
            if abs(currentGlobalRelError)>maxGlobalRelError
              currentGlobalIteration = currentGlobalIteration+1;
              if currentGlobalIteration==maxGlobalIterations
                error('Maximum number of iterations reached for the global cycle without convergence.')
              end
            % else
            %   currentGlobalIteration = maxGlobalIterations;
            else
              break;
            end
          else
            break;
          end
     
        end

        ne = chemistry.workCond.electronDensity;
        I = Constant.electronCharge*ne*chemistry.electronKinetics.swarmParam.driftVelocity*pi*chemistry.workCond.chamberRadius^2;
        if strcmp(chemistry.dischargeConditionForElecDensityCycle,'dischargeCurrent')
          currentElecDensityRelError = (chemistry.workCond.dischargeCurrent - I)/chemistry.workCond.dischargeCurrent;
          linCorrectedElectronDensity = ne*chemistry.workCond.dischargeCurrent/I;
        elseif strcmp(chemistry.dischargeConditionForElecDensityCycle,'dischargePowerPerLength')
          powerPerLength = I*chemistry.workCond.reducedFieldSI*chemistry.workCond.gasDensity;
          currentElecDensityRelError = (chemistry.workCond.dischargePowerPerLength - powerPerLength)/chemistry.workCond.dischargePowerPerLength;
          linCorrectedElectronDensity = ne*chemistry.workCond.dischargePowerPerLength/powerPerLength;
        end  
        elecDensityIterationAll(end+1) = currentElecDensityIteration;
        elecDensityRelErrorAll(end+1) = currentElecDensityRelError;

        % broadcast results of this iteration
        chemistry.elecDensityIterationCurrent = currentElecDensityIteration;
        chemistry.elecDensityRelErrorCurrent = currentElecDensityRelError;
        notify(chemistry, 'newElecDensityCycleIteration');        
        
        % prepare next iteration with a new electron density
        if abs(currentElecDensityRelError)>maxElecDensityRelError
          elecDensityAll(end+1) = chemistry.workCond.electronDensity;
          if length(elecDensityAll) == 1
            chemistry.workCond.update('electronDensity', linCorrectedElectronDensity);
          else  
            chemistry.workCond.update('electronDensity', iterateOverParameter(elecDensityIterationAll, elecDensityRelErrorAll, elecDensityAll, true, false));
          end  
          currentElecDensityIteration = currentElecDensityIteration + 1;
          if currentElecDensityIteration > maxElecDensityIterations
            error('Maximum number of iterations reached for the electron density cycle without convergence.');
          end
        else
          break;
        end  
      end  

      % ---- END OF GLOBAL CYCLE
      
      % evaluate density of parent states in the final solution (time evolution) and final total gas density (final time)
      totalGasDensity = 0;
      for i = 1:chemistry.numberOfSpecies
        if isempty(chemistry.childIDs{i}) && any(i == volPhaseSpeciesIDs)
          totalGasDensity = totalGasDensity+absDensitiesTime(end,i);
        elseif ~isempty(chemistry.childIDs{i})
          absDensitiesTime(:,i) = 0;
          for j = chemistry.childIDs{i}
            if ~isempty(chemistry.childIDs{j})
              absDensitiesTime(:,j) = 0;
              for k = chemistry.childIDs{j}
                absDensitiesTime(:,j) = absDensitiesTime(:,j) + absDensitiesTime(:,k);
              end
            end
            absDensitiesTime(:,i) = absDensitiesTime(:,i) + absDensitiesTime(:,j);
          end
        end
      end

      % evaluate temporal evolution of intermediate temperatures (if thermal model is activated)
      nearWallTemperatureTime = [];
      wallTemperatureTime = [];
      if chemistry.includeThermalModel
        switch chemistry.thermalModelBoundary
          case 'wall'
            nearWallTemperatureTime = zeros(1,length(gasTemperatureTime));
            nearWallTemperatureTime(1) = initialNearWallTemperature;
            for i = 1:length(gasTemperatureTime)-1
              chemistry.workCond.gasTemperature = gasTemperatureTime(i);
              chemistry.workCond.nearWallTemperature = nearWallTemperatureTime(i);
              [~, thermalModel] = kinetics(time(i), [absDensitiesTime(i,:) gasTemperatureTime(i)]', chemistry.workCond, ...
                chemistry.gasIDs, chemistry.childIDs, directRateCoeffs, inverseRateCoeffs, ...
                chemistry.rateCoeffFuncHandles, chemistry.rateCoeffParams, chemistry.reactantElectrons, ...
                chemistry.reactantIDs, chemistry.reactantStoiCoeffs, chemistry.catalystIDs, ...
                chemistry.catalystStoiCoeffs, chemistry.productElectrons, chemistry.productIDs, ...
                chemistry.productStoiCoeffs, chemistry.gasStabilisedReactionIDs, timeDependentReactionIDs, ...
                chemistry.reactionArray, chemistry.gasArray, chemistry.stateArray, eTransProp, ...
                chemistry.electronKinetics, false, chemistry.includeThermalModel, chemistry, false);
              nearWallTemperatureTime(i+1) = thermalModel.nearWallTemperature;
            end
          case 'external'
            nearWallTemperatureTime = zeros(1,length(gasTemperatureTime));
            nearWallTemperatureTime(1) = initialNearWallTemperature;
            wallTemperatureTime = zeros(1,length(gasTemperatureTime));
            wallTemperatureTime(1) = initialWallTemperature;
            for i = 1:length(gasTemperatureTime)-1
              chemistry.workCond.gasTemperature = gasTemperatureTime(i);
              chemistry.workCond.nearWallTemperature = nearWallTemperatureTime(i);
              chemistry.workCond.wallTemperature = wallTemperatureTime(i);
              [~, thermalModel] = kinetics(time(i), [absDensitiesTime(i,:) gasTemperatureTime(i)]', chemistry.workCond, ...
                chemistry.gasIDs, chemistry.childIDs, directRateCoeffs, inverseRateCoeffs, ...
                chemistry.rateCoeffFuncHandles, chemistry.rateCoeffParams, chemistry.reactantElectrons, ...
                chemistry.reactantIDs, chemistry.reactantStoiCoeffs, chemistry.catalystIDs, ...
                chemistry.catalystStoiCoeffs, chemistry.productElectrons, chemistry.productIDs, ...
                chemistry.productStoiCoeffs, chemistry.gasStabilisedReactionIDs, timeDependentReactionIDs, ...
                chemistry.reactionArray, chemistry.gasArray, chemistry.stateArray, eTransProp, ...
                chemistry.electronKinetics, false, chemistry.includeThermalModel, chemistry, false);
              nearWallTemperatureTime(i+1) = thermalModel.nearWallTemperature;
              wallTemperatureTime(i+1) = thermalModel.wallTemperature;
            end
        end
      else
        thermalModel = struct.empty;
      end

      
      % evaluate final (last time point) rates
      
      % evaluate time dependent rate coefficients (at final time)
      chemistry.workCond.gasTemperature = gasTemperatureTime(end);
      if chemistry.includeThermalModel
        switch chemistry.thermalModelBoundary
          case 'wall'
            chemistry.workCond.nearWallTemperature = nearWallTemperatureTime(end);
          case 'external'
            chemistry.workCond.nearWallTemperature = nearWallTemperatureTime(end);
            chemistry.workCond.wallTemperature = wallTemperatureTime(end);
        end
      end
      for i = 1:length(timeDependentReactionIDs)
        reactionID = timeDependentReactionIDs(i);
        directRateCoeffs(reactionID) = chemistry.rateCoeffFuncHandles{reactionID}(time(end), ...
          absDensitiesTime(end,:)', totalGasDensity, chemistry.reactionArray, reactionID, ...
          chemistry.stateArray, chemistry.workCond, eTransProp, chemistry.rateCoeffParams{reactionID});
        if chemistry.reactionArray(reactionID).isReverse
          inverseRateCoeffs(reactionID) = detailedBalance(chemistry.reactionArray(reactionID), ...
            directRateCoeffs(reactionID), Constant.boltzmannInEV*chemistry.workCond.gasTemperature);
        end
      end
      % evaluate reaction rate
      reactionRates = directRateCoeffs.*prod(repmat(absDensitiesTime(end,:)',[1 chemistry.numberOfReactions]).^ ...
        (chemistry.reactantStoiCoeffs+chemistry.catalystStoiCoeffs),1).* ...
        chemistry.workCond.electronDensity.^chemistry.reactantElectrons- ...
        inverseRateCoeffs.*prod(repmat(absDensitiesTime(end,:)',[1 chemistry.numberOfReactions]).^ ...
        (chemistry.productStoiCoeffs+chemistry.catalystStoiCoeffs),1).* ...
        chemistry.workCond.electronDensity.^chemistry.productElectrons;
      
      % store steady state solution of the heavy species kinetics in the chemistry properties
      steadyStateDensity = absDensitiesTime(end,:);
      steadyStateDensity(surPhaseSpeciesIDs) = steadyStateDensity(surPhaseSpeciesIDs)./chemistry.workCond.areaOverVolume;
      chemistry.solution.steadyStateDensity = steadyStateDensity;
      chemistry.solution.reactionsInfo = struct.empty;
      for id = [chemistry.reactionArray.ID]
        chemistry.solution.reactionsInfo(end+1).reactID = chemistry.reactionArray(id).ID;
        if chemistry.reactionArray(id).isReverse
          chemistry.solution.reactionsInfo(end).rateCoeff = [directRateCoeffs(id) inverseRateCoeffs(id)];
        else
          chemistry.solution.reactionsInfo(end).rateCoeff = directRateCoeffs(id);
        end
        chemistry.solution.reactionsInfo(end).netRate = reactionRates(id);
        chemistry.solution.reactionsInfo(end).energy = chemistry.reactionArray(id).enthalpy;
        chemistry.solution.reactionsInfo(end).description = erase(chemistry.reactionArray(id).descriptionExtended, ' ');
      end
      chemistry.solution.thermalModel = thermalModel;
      
      %%%%%%%%%%%%%%%%%%%%%%%%% START OF POST-DISCHARGE CODE (WIP) %%%%%%%%%%%%%%%%%%%%%%%%%
      if chemistry.odePostDischargeTime > 0
        % set electron temperature equal to gas temperature
        % (this is overwritten later if the evolution of the eedf is solved or the thermal module is activated)
        chemistry.workCond.electronTemperature = gasTemperatureTime(end)*Constant.boltzmann/Constant.electronCharge;
        % evaluate initial values for the rate coefficients saving IDs of time dependent reactions (post-discharge)
        directRateCoeffs = zeros(1, chemistry.numberOfReactions);
        inverseRateCoeffs = zeros(1, chemistry.numberOfReactions);
        timeDependentReactionIDs = [];
        for reactionID = 1:length(directRateCoeffs)
          % set to zero all "eedf" rate coefficients 
          % (this is overwritten later if the evolution of the eedf is solved)
          if strcmp(chemistry.reactionArray(reactionID).type, 'eedf')
            chemistry.reactionArray(reactionID).eedfEquivalent.ineRateCoeff = 0.0;
            chemistry.reactionArray(reactionID).eedfEquivalent.supRateCoeff = 0.0;
            directRateCoeffs(reactionID) = 0.0;
            inverseRateCoeffs(reactionID) = 0.0;
          end
          [directRateCoeffs(reactionID), dependent] = chemistry.rateCoeffFuncHandles{reactionID}(time(end), ...
          absDensitiesTime(end,:)', totalGasDensity, chemistry.reactionArray, reactionID, ...
          chemistry.stateArray, chemistry.workCond, eTransProp, chemistry.rateCoeffParams{reactionID});
          if dependent.onTime || dependent.onDensities || ...
              (chemistry.includeThermalModel && dependent.onGasTemperature) || dependent.onElectronKinetics
            timeDependentReactionIDs(end+1) = reactionID;
          end
          if chemistry.reactionArray(reactionID).isReverse
            inverseRateCoeffs(reactionID) = detailedBalance(chemistry.reactionArray(reactionID), ...
              directRateCoeffs(reactionID), Constant.boltzmannInEV*chemistry.workCond.gasTemperature);
          end
        end
        chemistry.workCond.update('reducedField', 0);
        % flush persistent memory of kinetics function
        kinetics(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, true);
%         chemistry.odeOptions.OutputFcn = @odeProgressBar;          % activate only for debugging purposes
        if chemistry.solveEedf
          chemistry.odeOptions.NonNegative = ...
            1:chemistry.numberOfSpecies+1+chemistry.electronKinetics.energyGrid.cellNumber;
          [timePostDschrg, timeSolutionPostDschrg] = chemistry.odeSolver(@kinetics, ...
            [time(end) time(end)+chemistry.odePostDischargeTime], ...
            [absDensitiesTime(end,:) gasTemperatureTime(end) chemistry.electronKinetics.eedf], ...
          chemistry.odeOptions, chemistry.workCond, chemistry.gasIDs, chemistry.childIDs, directRateCoeffs, ...
          inverseRateCoeffs, chemistry.rateCoeffFuncHandles, chemistry.rateCoeffParams, chemistry.reactantElectrons, ...
          chemistry.reactantIDs, chemistry.reactantStoiCoeffs, chemistry.catalystIDs, chemistry.catalystStoiCoeffs, ...
          chemistry.productElectrons, chemistry.productIDs, chemistry.productStoiCoeffs, ...
            chemistry.gasStabilisedReactionIDs, timeDependentReactionIDs, chemistry.reactionArray, ...
            chemistry.gasArray, chemistry.stateArray, eTransProp, chemistry.electronKinetics, true, ...
            chemistry.includeThermalModel, chemistry, false);
          % separate eedf solution
          eedfTimePostDschrg = timeSolutionPostDschrg(:,chemistry.numberOfSpecies+2:end);
        else
          [timePostDschrg, timeSolutionPostDschrg] = chemistry.odeSolver(@kinetics, ...
            [time(end) time(end)+chemistry.odePostDischargeTime], ...
            [absDensitiesTime(end,:) gasTemperatureTime(end)], ...
            chemistry.odeOptions, chemistry.workCond, chemistry.gasIDs, chemistry.childIDs, directRateCoeffs, ...
            inverseRateCoeffs, chemistry.rateCoeffFuncHandles, chemistry.rateCoeffParams, chemistry.reactantElectrons, ...
            chemistry.reactantIDs, chemistry.reactantStoiCoeffs, chemistry.catalystIDs, chemistry.catalystStoiCoeffs, ...
            chemistry.productElectrons, chemistry.productIDs, chemistry.productStoiCoeffs, ...
            chemistry.gasStabilisedReactionIDs, timeDependentReactionIDs, chemistry.reactionArray, ...
            chemistry.gasArray, chemistry.stateArray, eTransProp, chemistry.electronKinetics, true, ...
            chemistry.includeThermalModel, chemistry, false);
        end
        % separate time solutions into its different components
        absDensitiesTimePostDschrg = timeSolutionPostDschrg(:,1:chemistry.numberOfSpecies);
        gasTemperatureTimePostDschrg = timeSolutionPostDschrg(:,chemistry.numberOfSpecies+1);
        % evaluate density of parent states in the porst-discharge solution (time evolution) and final total gas density
        totalGasDensity = 0;
        for i = 1:chemistry.numberOfSpecies
          if isempty(chemistry.childIDs{i}) && any(i == volPhaseSpeciesIDs)
            totalGasDensity = totalGasDensity+absDensitiesTimePostDschrg(end,i);
          elseif ~isempty(chemistry.childIDs{i})
            absDensitiesTimePostDschrg(:,i) = 0;
            for j = chemistry.childIDs{i}
              if ~isempty(chemistry.childIDs{j})
                absDensitiesTimePostDschrg(:,j) = 0;
                for k = chemistry.childIDs{j}
                  absDensitiesTimePostDschrg(:,j) = absDensitiesTimePostDschrg(:,j) + absDensitiesTimePostDschrg(:,k);
                end
              end
              absDensitiesTimePostDschrg(:,i) = absDensitiesTimePostDschrg(:,i) + absDensitiesTimePostDschrg(:,j);
            end
          end
        end
        absDensitiesTime = cat(1, absDensitiesTime, absDensitiesTimePostDschrg);
        gasTemperatureTime = [gasTemperatureTime; gasTemperatureTimePostDschrg];
        nearWallTemperatureTimePostDschrg = [];
        wallTemperatureTimePostDschrg = [];
        if chemistry.includeThermalModel
          switch chemistry.thermalModelBoundary
            case 'wall'
              nearWallTemperatureTimePostDschrg = zeros(1,length(gasTemperatureTimePostDschrg));
              nearWallTemperatureTimePostDschrg(1) = nearWallTemperatureTime(end);
              for i = 1:length(gasTemperatureTimePostDschrg)-1
                chemistry.workCond.gasTemperature = gasTemperatureTimePostDschrg(i);
                chemistry.workCond.nearWallTemperature = nearWallTemperatureTimePostDschrg(i);
                [~, thermalModel] = kinetics(timePostDschrg(i), ...
                  [absDensitiesTimePostDschrg(i,:) gasTemperatureTimePostDschrg(i)]', chemistry.workCond, ...
                  chemistry.gasIDs, chemistry.childIDs, directRateCoeffs, inverseRateCoeffs, ...
                  chemistry.rateCoeffFuncHandles, chemistry.rateCoeffParams, chemistry.reactantElectrons, ...
                  chemistry.reactantIDs, chemistry.reactantStoiCoeffs, chemistry.catalystIDs, ...
                  chemistry.catalystStoiCoeffs, chemistry.productElectrons, chemistry.productIDs, ...
                  chemistry.productStoiCoeffs, chemistry.gasStabilisedReactionIDs, timeDependentReactionIDs, ...
                  chemistry.reactionArray, chemistry.gasArray, chemistry.stateArray, eTransProp, ...
                  chemistry.electronKinetics, false, chemistry.includeThermalModel, chemistry, false);
                nearWallTemperatureTimePostDschrg(i+1) = thermalModel.nearWallTemperature;
              end
            case 'external'
              nearWallTemperatureTimePostDschrg = zeros(1,length(gasTemperatureTimePostDschrg));
              nearWallTemperatureTimePostDschrg(1) = nearWallTemperatureTime(end);
              wallTemperatureTimePostDschrg = zeros(1,length(gasTemperatureTimePostDschrg));
              wallTemperatureTimePostDschrg(1) = wallTemperatureTime(end);
              for i = 1:length(gasTemperatureTimePostDschrg)-1
                chemistry.workCond.gasTemperature = gasTemperatureTimePostDschrg(i);
                chemistry.workCond.nearWallTemperature = nearWallTemperatureTimePostDschrg(i);
                chemistry.workCond.wallTemperature = wallTemperatureTimePostDschrg(i);
                [~, thermalModel] = kinetics(timePostDschrg(i), ...
                  [absDensitiesTimePostDschrg(i,:) gasTemperatureTimePostDschrg(i)]', chemistry.workCond, ...
                  chemistry.gasIDs, chemistry.childIDs, directRateCoeffs, inverseRateCoeffs, ...
                  chemistry.rateCoeffFuncHandles, chemistry.rateCoeffParams, chemistry.reactantElectrons, ...
                  chemistry.reactantIDs, chemistry.reactantStoiCoeffs, chemistry.catalystIDs, ...
                  chemistry.catalystStoiCoeffs, chemistry.productElectrons, chemistry.productIDs, ...
                  chemistry.productStoiCoeffs, chemistry.gasStabilisedReactionIDs, timeDependentReactionIDs, ...
                  chemistry.reactionArray, chemistry.gasArray, chemistry.stateArray, eTransProp, ...
                  chemistry.electronKinetics, false, chemistry.includeThermalModel, chemistry, false);
                nearWallTemperatureTimePostDschrg(i+1) = thermalModel.nearWallTemperature;
                wallTemperatureTimePostDschrg(i+1) = thermalModel.wallTemperature;
              end
          end
        end
        nearWallTemperatureTime = [nearWallTemperatureTime nearWallTemperatureTimePostDschrg];
        wallTemperatureTime = [wallTemperatureTime wallTemperatureTimePostDschrg];
        time = cat(1,time,timePostDschrg);
      end
      %%%%%%%%%%%%%%%%%%%%%%%%% END OF POST-DISCHARGE CODE (WIP) %%%%%%%%%%%%%%%%%%%%%%%%%
      
      % store time dependent solution of the heavy species kinetics in the chemistry properties
      absDensitiesTime(:,surPhaseSpeciesIDs) = absDensitiesTime(:,surPhaseSpeciesIDs)./chemistry.workCond.areaOverVolume;
      chemistry.solution.time = time;
      chemistry.solution.densitiesTime = absDensitiesTime;
      chemistry.solution.gasTemperatureTime = gasTemperatureTime;
      chemistry.solution.nearWallTemperatureTime = nearWallTemperatureTime;
      chemistry.solution.wallTemperatureTime = wallTemperatureTime;
      
      % broadcast obtention of a solution for the chemistry equation
      notify(chemistry, 'obtainedNewChemistrySolution');
      
    end
    
  end

end

function [derivatives, thermalModel] = ...
  kinetics(time, variables, workCond, gasIDs, childIDs, directRateCoeffs, inverseRateCoeffs, ...
  rateCoeffFuncHandles, rateCoeffParams, reactantElectrons, reactantIDs, reactantStoiCoeffs, catalystIDs, ...
  catalystStoiCoeffs, productElectrons, productIDs, productStoiCoeffs, gasStabilisedReactionIDs, ...
  timeDependentReactionIDs, reactionArray, gasArray, stateArray, eTranspProp, electronKinetics, solveElectrons, ...
  includeThermalModel, chemistry, clearPersistentVars)
  
  % define persistent variables (do not change during the simulation)
  persistent electronDensitiesReactant;
  persistent electronDensitiesProduct;
  persistent totalReactantStoiCoeffs;
  persistent totalProductStoiCoeffs;
  persistent productMinusReactantStoiCoeffs;
  persistent isTimeDependentReactionReverse;
  persistent reactionEnthalpies;
  persistent numberOfGases;
  persistent numberOfSpecies;
  persistent volumePhaseSpeciesIDs;
  persistent surfacePhaseSpeciesIDs;
  persistent numberOfReactions;
  persistent massArray;                       % array with masses of all gases (needed for the thermal model)
  persistent charLengthThermalModelSquared;   % characteristic length for the thermal model (see documentation)
  persistent lstdsteq1;  
  persistent lstdstgt1;
  persistent lstisteq1;
  persistent lstistgt1;
  persistent drateaux;
  persistent irateaux;
  
  % flush persistent memory for a new simulation (needed before post-discharge integration)
  if clearPersistentVars
    vars = whos;
    vars = vars([vars.persistent]);
    varName = {vars.name};
    clear(varName{:});
    derivatives = [];
    return
  end

  % initialize values of persistent variables
  if isempty(electronDensitiesReactant)
    totalReactantStoiCoeffs = (reactantStoiCoeffs+catalystStoiCoeffs)';
    totalProductStoiCoeffs = (productStoiCoeffs+catalystStoiCoeffs)';
    productMinusReactantStoiCoeffs = productStoiCoeffs - reactantStoiCoeffs;
    numberOfGases = max(gasIDs);
    numberOfSpecies = chemistry.numberOfSpecies;
    volumePhaseSpeciesIDs = chemistry.volumePhaseSpeciesIDs;
    surfacePhaseSpeciesIDs = chemistry.surfacePhaseSpeciesIDs;
    numberOfReactions = length(directRateCoeffs);
    lstdsteq1=find(totalReactantStoiCoeffs==1);   %List of positions with stoichiometric coeff == 1 in direct reactions
    lstdstgt1=find(totalReactantStoiCoeffs>1);    %List of positions with stoichiometric coeff > 1 in direct reactions
    lstisteq1=find(totalProductStoiCoeffs==1);
    lstistgt1=find(totalProductStoiCoeffs>1);
    drateaux=ones(size(totalReactantStoiCoeffs));  %Initialize density matrix with 1 for direct rates calculation 
    irateaux=ones(size(totalProductStoiCoeffs));
    isTimeDependentReactionReverse = false(size(timeDependentReactionIDs));
    for i = 1:length(timeDependentReactionIDs)
      reactionID = timeDependentReactionIDs(i);
      if reactionArray(reactionID).isReverse
        isTimeDependentReactionReverse(i) = true;
      end
    end
    if includeThermalModel
      massArray = [gasArray.mass];
      % saving reaction enthalpies in a simple and fast array of doubles
      reactionEnthalpies = zeros(size(reactionArray));
      for i = 1:numberOfReactions
        reactionEnthalpies(i) = reactionArray(i).enthalpy;
      end
      % evaluate characteristic length for the thermal model (radial profile for gas temperature)
      charLengthThermalModelSquared = workCond.chamberRadius^2/8;
    end
  end

  electronDensitiesReactant = workCond.electronDensity.^reactantElectrons';
  electronDensitiesProduct = workCond.electronDensity.^productElectrons';  

  % separate variables
  densities = variables(1:numberOfSpecies);
  gasTemperature = variables(numberOfSpecies+1);
  if solveElectrons
    eedf = variables(numberOfSpecies+2:end);
  end
  
  % evaluate density of parent states, individual gases and total gas density
  totalGasDensity = 0;
  individualGasDensity = zeros(1, numberOfGases);
  for i = 1:numberOfSpecies
    if isempty(childIDs{i}) && any(i == volumePhaseSpeciesIDs)
      totalGasDensity = totalGasDensity+densities(i);
      individualGasDensity(gasIDs(i)) = individualGasDensity(gasIDs(i))+densities(i);
    elseif ~isempty(childIDs{i})
      densities(i) = 0;
      IDs1 = childIDs{i};
      for j = 1:length(IDs1)
        if ~isempty(childIDs{IDs1(j)})
          densities(IDs1(j)) = 0;
          IDs2 = childIDs{IDs1(j)};
          for k = 1:length(IDs2)
            densities(IDs1(j)) = densities(IDs1(j)) + densities(IDs2(k));
          end
        end
        densities(i) = densities(i) + densities(IDs1(j));
      end
    end
  end
  
  % evaluate thermal balance equation parameters (if thermal model is activated) --- 1st part of the calculations
  if includeThermalModel
    % update working conditions object with current gas temperature
    workCond.gasTemperature = gasTemperature;
    % reevaluate thermal properties of gases that might depend on the gas temperature
    heatCapacityArray = zeros(1,numberOfGases);
    thermalConductivityArray = zeros(1,numberOfGases);
    for i = 1:numberOfGases
      if gasArray(i).isVolumeSpecies
        heatCapacityArray(i) = gasArray(i).evaluateHeatCapacity(workCond);
        thermalConductivityArray(i) = gasArray(i).evaluateThermalConductivity(workCond);
      end
    end
    % evaluate total system heat capacity 
    heatCapacity = dot(individualGasDensity,heatCapacityArray/Constant.avogadro);
    % evaluate total system heat conductivity (can not be vectorized because of exceptions in the loops)
    euckenFactorArray = 0.115+0.354*heatCapacityArray/Constant.idealGas*Constant.electronCharge;
    thermalConductivity = 0;
    for i = 1:numberOfGases
      if individualGasDensity(i) == 0
        continue
      end
      aux = 1;
      for j = 1:numberOfGases
        if j == i || individualGasDensity(j) == 0
          continue
        end
        aux = aux + 1.065/(2*sqrt(2*(1+massArray(i)/massArray(j))))*(1+sqrt(thermalConductivityArray(i)* ...
          euckenFactorArray(j)/(thermalConductivityArray(j)*euckenFactorArray(i))*sqrt(massArray(i)/massArray(j))))^2* ...
          individualGasDensity(j)/individualGasDensity(i);
      end
      thermalConductivity = thermalConductivity + thermalConductivityArray(i) / aux;
    end
    % update near wall temperature in case of 'wall' boundary condition
    if strcmp(chemistry.thermalModelBoundary, 'wall')
      aux = 4*thermalConductivity/workCond.chamberRadius;
      hint = chemistry.intConvCoeff;
      nearWallTemperature = (aux*gasTemperature/hint+workCond.wallTemperature) / (aux/hint+1);
      workCond.nearWallTemperature = nearWallTemperature;
      wallTemperature = workCond.wallTemperature;
    end
  end
  
  % solve evolution of electron density by imposing quasi-neutrality
  if solveElectrons
    % evaluate current electron density (imposing quasi-neutrality)
    electronDensity = 0;
    for i = 1:numberOfSpecies
      if strcmp(stateArray(i).ionCharg, '+')
        electronDensity = electronDensity + densities(i);
      elseif strcmp(stateArray(i).ionCharg, '-')
        electronDensity = electronDensity - densities(i);
      end
    end
    workCond.electronDensity = electronDensity;
    electronDensitiesReactant = electronDensity.^reactantElectrons';
    electronDensitiesProduct = electronDensity.^productElectrons';
    % evaluate derivatives of the components of the EEDF (in case it is activated)
    if chemistry.solveEedf
      % update gas temperature dependencies of the electron kinetics (in case thermal model is active)
      if includeThermalModel
        % update possible populations depending on the gas temperature
        for gas = electronKinetics.gasArray
          for state = gas.stateArray
            if ~isempty(state.populationFunc) && isempty(state.chemEquivalent)
              for parameter = state.populationParams
                if strcmp(parameter{1}, 'gasTemperature')
                  state.evaluatePopulation(workCond);
                  break;
                end
              end
            end
          end
        end
        % update the densities of states accordingly
        for gas = electronKinetics.gasArray
          for state = gas.stateArray
            state.evaluateDensity();
          end
        end
      end
      % evaluate electron kinetics (eedf) densities
      eedfAbsDensities = densities;
      eedfTotalGasDensity = 0;
      for i = chemistry.statesIDsToUpdateInElectronKinetics
        if isempty(chemistry.childIDs{i})
          eedfTotalGasDensity = eedfTotalGasDensity + eedfAbsDensities(i);
        else
          eedfAbsDensities(i) = 0;
          for j = intersect(chemistry.childIDs{i}, chemistry.statesIDsToUpdateInElectronKinetics)
            if ~isempty(chemistry.childIDs{j})
              eedfAbsDensities(j) = 0;
              for k = intersect(chemistry.childIDs{j}, chemistry.statesIDsToUpdateInElectronKinetics)
                eedfAbsDensities(j) = eedfAbsDensities(j) + eedfAbsDensities(k);
              end
            end
            eedfAbsDensities(i) = eedfAbsDensities(i) + eedfAbsDensities(j);
          end
        end
      end
      % update densities of states in the the electron kinetics gas mixture (normalized to gas density)
      for i = chemistry.statesIDsToUpdateInElectronKinetics
        chemistry.stateArray(i).eedfEquivalent.density = eedfAbsDensities(i)/eedfTotalGasDensity;
      end
      % renormalize electron kinetics gas mixture
      for i = chemistry.gasesIDsToUpdateInElectronKinetics
        chemistry.gasArray(i).eedfEquivalent.renormalizeWithDensities();
        % check for the distribution of states to be properly normalised
        chemistry.gasArray(i).eedfEquivalent.checkPopulationNorms();
      end
      % update the density dependencies of the electron kinetics (this also updates gas temperature dependencies)
      electronKinetics.updateDensityDependencies();
      % evaluate eedf temporal derivatives with new densities
      %       chemistry.electronKinetics.solve();
      eedfDerivatives = eedfTimeDerivative(time, [eedf; electronDensity], electronKinetics, false, false);
      eedfDerivatives = eedfDerivatives(1:end-1);
      % evaluate new electron impact rate coefficients and transport parameters
      electronKinetics.eedf = eedf';
      electronKinetics.evaluateMacroscopicParameters;
      eTranspProp.reducedDiffCoeff = electronKinetics.swarmParam.redDiffCoeff;
      eTranspProp.reducedMobility = electronKinetics.swarmParam.redMobility;
    elseif includeThermalModel
      workCond.electronTemperature = gasTemperature*Constant.boltzmann/Constant.electronCharge;
    end
  end
  
  % evaluate time dependent rate coefficients
  KbTgInEV = Constant.boltzmannInEV*workCond.gasTemperature;
  for i = 1:length(timeDependentReactionIDs)
    reactionID = timeDependentReactionIDs(i);
    directRateCoeffs(reactionID) = rateCoeffFuncHandles{reactionID}(time, densities, totalGasDensity, ...
      reactionArray, reactionID, stateArray, workCond, eTranspProp, rateCoeffParams{reactionID});
    if isTimeDependentReactionReverse(i)
      inverseRateCoeffs(reactionID) = detailedBalance(reactionArray(reactionID), directRateCoeffs(reactionID), KbTgInEV);
    end
  end

  % evaluate rates of the reactions
  densitiesMatrix = repmat(densities',[numberOfReactions 1]);
  drateaux(lstdsteq1)=densitiesMatrix(lstdsteq1);
  drateaux(lstdstgt1)=densitiesMatrix(lstdstgt1).^totalReactantStoiCoeffs(lstdstgt1);
  irateaux(lstisteq1)=densitiesMatrix(lstisteq1);
  irateaux(lstistgt1)=densitiesMatrix(lstistgt1).^totalProductStoiCoeffs(lstistgt1);
  reactionRates = directRateCoeffs'.*(electronDensitiesReactant.*prod(drateaux,2)) - ...
    inverseRateCoeffs'.*(electronDensitiesProduct.*prod(irateaux,2));
  for i = gasStabilisedReactionIDs
    reactionRates(i) = reactionRates(i)*totalGasDensity;
  end
  
  % evaluate temporal derivatives for the densities
  densityDerivatives = productMinusReactantStoiCoeffs*reactionRates;
  
  % evaluate temporal derivative of the gas temperature (if thermal model is activated) --- 2nd part of the calculations
  if includeThermalModel
    volumeReactionIDs = chemistry.volumeReactionIDs;
    transportReactionIDs = chemistry.transportReactionIDs;
    fw = chemistry.thermalModelWallFraction;
    elasticCollisions = -workCond.electronDensity*totalGasDensity*electronKinetics.power.elasticNet;
    volumeSource = dot(reactionRates(volumeReactionIDs), reactionEnthalpies(volumeReactionIDs));
    global O2XStateID sumO2XVibRateCoeffsXEnthalpies;
    if ~isempty(O2XStateID)
      volumeSource = volumeSource + workCond.electronDensity*densities(O2XStateID)*sumO2XVibRateCoeffsXEnthalpies;
    end  
    wallSource = dot(reactionRates(transportReactionIDs), reactionEnthalpies(transportReactionIDs));
    % update near wall temperature and wall temperature in case of 'external' boundary condition
    if strcmp(chemistry.thermalModelBoundary, 'external')
        aux = 4*thermalConductivity/workCond.chamberRadius;
        hint = chemistry.intConvCoeff;
        hext = chemistry.extConvCoeff;
        wallTemperature = (aux*gasTemperature/hext + (aux/hint+1)*workCond.extTemperature + ...
          (aux/hint/hext+1/hext)*wallSource*workCond.chamberRadius*0.5) / (aux/hint+aux/hext+1);
        nearWallTemperature = ((aux/hint+aux/hext)*gasTemperature + workCond.extTemperature + ...
          wallSource*workCond.chamberRadius*0.5/hext) / (aux/hint+aux/hext+1);
        workCond.wallTemperature = wallTemperature;
        workCond.nearWallTemperature = nearWallTemperature;
    end
    conduction = -thermalConductivity*(gasTemperature-nearWallTemperature)/charLengthThermalModelSquared;
    temperatureDerivative = (conduction + elasticCollisions + volumeSource + fw*wallSource)/heatCapacity;
    thermalModel = struct('nearWallTemperature', nearWallTemperature, 'wallTemperature', wallTemperature, ...
      'conduction', conduction, 'elasticCollisions', elasticCollisions, 'volumeSource', volumeSource, ...
      'wallSource', wallSource);
  else
    temperatureDerivative = 0;
    thermalModel = struct.empty;
  end
  
  % combine derivatives of the diferent variables
  derivatives = [densityDerivatives; temperatureDerivative];
  if solveElectrons && chemistry.solveEedf
    derivatives = [derivatives; eedfDerivatives];
  end
  
end

function inverseRateCoeff = detailedBalance(reaction, directRateCoeff, KbTgInEV)
  % detailedBalance evaluate the rate coefficient of the inverse reaction by taking into account the detailed
  % balance. For the moment, the detailed balance is only implemented for reactions of type "eedf", i.e. rate
  % coefficients calculated through the integration of a cross section with an EEDF, and for two body reactions
  % between heavy species (not electrons).
  
  persistent statWeightRatio;
  persistent exponent;
  
  % in case the inverse rate coefficient is already evaluated by the rate coefficient function return that value
  if ~isempty(reaction.backRateCoeff)
    inverseRateCoeff = reaction.backRateCoeff;
    return;
  end

  % in case the inverse rate coefficient has not been calculated check if it falls into one of the "standard" categories
  switch reaction.type
    case 'eedf'
      inverseRateCoeff = reaction.eedfEquivalent.supRateCoeff;
    otherwise
      reactionID = reaction.ID;
      if length(statWeightRatio)<reactionID || statWeightRatio(reactionID) == 0
        if reaction.reactantElectrons == 0 && reaction.productElectrons == 0 && ...
            sum(reaction.reactantStoiCoeff)+sum(reaction.catalystStoiCoeff) == 2 && ...
            sum(reaction.productStoiCoeff)+sum(reaction.catalystStoiCoeff) == 2
          statWeightRatio(reactionID) = 1;
          exponent(reactionID) = 0;
          for i = 1:length(reaction.reactantArray)
            state = reaction.reactantArray(i);
            if isempty(state.statisticalWeight)
              error('Unable to find %s statatistical weight to evaluate the detail balance of reaction: \n%s', ...
                state.name, reaction.description);
            elseif isempty(state.energy)
              error('Unable to find %s energy to evaluate the detail balance of reaction: \n%s', ...
                state.name, reaction.description);
            end
            statWeightRatio(reactionID) = statWeightRatio(reactionID) * ...
              state.statisticalWeight^reaction.reactantStoiCoeff(i);
            exponent(reactionID) = exponent(reactionID) - ...
              state.energy*reaction.reactantStoiCoeff(i);
          end
          for i = 1:length(reaction.productArray)
            state = reaction.productArray(i);
            if isempty(state.statisticalWeight)
              error('Unable to find %s statatistical weight to evaluate the detail balance of reaction: \n%s', ...
                state.name, reaction.description);
            elseif isempty(state.energy)
              error('Unable to find %s energy to evaluate the detail balance of reaction: \n%s', ...
                state.name, reaction.description);
            end
            statWeightRatio(reactionID) = statWeightRatio(reactionID) / ...
              state.statisticalWeight^reaction.productStoiCoeff(i);
            exponent(reactionID) = exponent(reactionID) + ...
              state.energy*reaction.productStoiCoeff(i);
          end        
        else
          error(['Error found when evaluating the inverse rate coefficient for reaction:\n%s\n' ...
            'Detailed balance is not implemented for this reaction.'], reaction.description);
        end
      end
      inverseRateCoeff = directRateCoeff*statWeightRatio(reactionID)*exp(exponent(reactionID)/KbTgInEV);
  end
  
end

function newParameter = iterateOverParameter(iterationIDs, relErrors, parameters, linearExtrapolationToBeDone, bissectionToBeDone)
% iterateOverParameter evaluates the new value of a certain parameter that it is iterated over based on data from
% previous iterations (the IDs of the iterations, the errors obtained and the values of the parameter for those
% iterations)

  if length(parameters)>5
    iterationIDs = iterationIDs(end-4:end);
    relErrors = relErrors(end-4:end);
    parameters = parameters(end-4:end);
  end

  allSameSign = ~( any(relErrors>0) && any(relErrors<0) );
  
  if  length(iterationIDs)==1 || (allSameSign && ~linearExtrapolationToBeDone)
    limitedRelError = max([min([relErrors(end) 1]) -0.5]);
    newParameter = parameters(end)*(1+limitedRelError);   
  else
    % order arrays of parameters and relative errors in ascending order of parameter
    [auxParameters, indecesParameter] = sort(parameters, 'ascend');
    auxRelErrors = relErrors(indecesParameter);

    if allSameSign
      newParameter = interp1(auxRelErrors,auxParameters,0,'linear','extrap');
      if  newParameter > 10*auxParameters(end)
        newParameter = 10*auxParameters(end);
      elseif  newParameter < 0.1*auxParameters(1) 
        newParameter = 0.1*auxParameters(1);      
      end 
      return;
    end  
    
    % find smallest negative and positive relative errors
    posErr = inf;
    negErr = -inf;
    for i = 1:length(auxRelErrors)
      if auxRelErrors(i) > 0 && auxRelErrors(i) < posErr
        posErr = auxRelErrors(i);
        positiveErrorID = i;
      elseif auxRelErrors(i) < 0 && auxRelErrors(i) > negErr
        negErr = auxRelErrors(i);
        negativeErrorID = i;
      end
    end
    
    % bisection method when the errors are too big
    if (posErr > 1 || negErr < -1) && bissectionToBeDone
      newParameter = (auxParameters(positiveErrorID)+auxParameters(negativeErrorID))/2;
      % avoid any parameter already used
      if any(auxParameters==newParameter)
        newParameter = newParameter*(rand*1.5+0.5);
      end
      return;
    end
    
    % select range of parameters and relative errors to be used in the interpolation function
    switch positiveErrorID
      case negativeErrorID-1
        minID = positiveErrorID;
        error = posErr;
        for i = positiveErrorID-1:-1:1
          if auxRelErrors(i)>error
            minID = i;
            error = auxRelErrors(i);
          else
            break;
          end
        end
        maxID = negativeErrorID;
        error = negErr;
        for i = negativeErrorID+1:length(auxRelErrors)
          if auxRelErrors(i)<error
            maxID = i;
            error = auxRelErrors(i);
          else
            break;
          end
        end
      case negativeErrorID+1
        maxID = positiveErrorID;
        error = posErr;
        for i = positiveErrorID+1:length(auxRelErrors)
          if auxRelErrors(i)>error
            maxID = i;
            error = auxRelErrors(i);
          else
            break;
          end
        end
        minID = negativeErrorID;
        error = negErr;
        for i = negativeErrorID-1:-1:1
          if auxRelErrors(i)<error
            minID = i;
            error = auxRelErrors(i);
          else
            break;
          end
        end
      otherwise
        minID = min([positiveErrorID negativeErrorID]);
        maxID = max([positiveErrorID negativeErrorID]);
        while maxID-minID > 1
          if sign(auxRelErrors(minID)) == sign(auxRelErrors(minID+1))
            minID = minID+1;
          end
          if sign(auxRelErrors(maxID)) == sign(auxRelErrors(maxID-1))
            maxID = maxID-1;
          end
        end
    end
    
    % interpolate new value for the parameter to ensure zero relative error
    if maxID-minID > 2
      newParameter = interp1(auxRelErrors(minID:maxID), auxParameters(minID:maxID), 0, 'spline');
      if newParameter > max([auxParameters(positiveErrorID) auxParameters(negativeErrorID)]) || ...
          newParameter < min([auxParameters(positiveErrorID) auxParameters(negativeErrorID)])
        newParameter = interp1(auxRelErrors(minID:maxID), auxParameters(minID:maxID), 0, 'linear');
      end
    else
      newParameter = interp1(auxRelErrors(minID:maxID), auxParameters(minID:maxID), 0, 'linear');
    end
    
    % avoid any parameter already used
    if any(auxParameters==newParameter)
      newParameter = newParameter*(rand*1.5-0.5);
    end
  end
  
end

function status = odeProgressBar(t,variables,flag, varargin)

  persistent progressFigure1;
  persistent progressFigure2;
  persistent progressGraph1;
  persistent progressGraph2;
  persistent integrationTimeStr1;
  persistent integrationTimeStr2;
  persistent initialClock;
  persistent initialTime;
  persistent finalTime;
  persistent allt;
  persistent allne;
  
  switch(flag)
    case 'init'
      initialTime = t(1)-1e3;
      finalTime = t(2)-1e3;
      progressFigure1 = figure('Name', 'Chemistry solver debugging window', 'NumberTitle', 'off', 'MenuBar', 'none');
      progressFigure2 = figure('Name', 'Chemistry solver debugging window', 'NumberTitle', 'off', 'MenuBar', 'none');
      integrationTimeStr1 = uicontrol('Parent', progressFigure1, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.1 0.9 0.5 0.05], 'HorizontalAlignment', 'left', 'String', 'Computational time: 0 s');
      integrationTimeStr2 = uicontrol('Parent', progressFigure2, 'Style', 'text', 'Units', 'normalized', ...
        'Position', [0.1 0.9 0.5 0.05], 'HorizontalAlignment', 'left', 'String', 'Computational time: 0 s');
      progressGraph1 = axes('Parent', progressFigure1, 'Units', 'normalized', 'OuterPosition', [0 0 1 0.9], ...
        'Box', 'on', 'XScale', 'log');
      progressGraph2 = axes('Parent', progressFigure2, 'Units', 'normalized', 'OuterPosition', [0 0 1 0.9], ...
        'Box', 'on');
      initialClock = clock;
      allt = initialTime;
      allne = variables(8)+variables(9)+variables(23)+variables(24)+variables(25);
    case 'done'
      close(progressFigure1)
      vars = whos;
      vars = vars([vars.persistent]);
      varName = {vars.name};
      clear(varName{:});
    otherwise
      allt(end+1) = t-1e3;
      allne(end+1) = variables(8)+variables(9)+variables(23)+variables(24)+variables(25);
      loglog(progressGraph1, allt, allne)
      legend({'n_e'});
      semilogy(progressGraph2, 1:1000, variables(end-999:end))
      legend({'eedf'});
      integrationTimeStr1.String = sprintf('Computational time: %.1f s\nPhysical time: %e s', ...
        etime(clock, initialClock), t-1e3);
      integrationTimeStr2.String = sprintf('Computational time: %.1f s\nPhysical time: %e s', ...
        etime(clock, initialClock), t-1e3);
  end
  status = 0;
  drawnow;

end
