function coefficient = temperatureDependentEinsteinDiffCoeff(state, argumentArray, workCond)
  % einsteinTemperatureDependentDiffCoeff(have to be writen)
  
  temperature = argumentArray{1};
  if ~isnumeric(temperature)
    switch temperature
      case 'gasTemperature'
        temperature = workCond.gasTemperature;
      case 'electronTemperature'
        temperature = workCond.electronTemperature*Constant.electronCharge/Constant.boltzmann;
      otherwise
        error(['Error found when evaluating coefficient of state %s.\nTemperature ''%s'' not defined in the ' ...
          'working conditions.\nPlease, fix the problem and run the code again.'], state.name, temperature);
    end
  end
  
  % retrieve reduced mobility (and check if available)
  redMob = state.evaluateReducedMobility(workCond);
  if isempty(redMob)
    error(['Error found when evaluating the reduced diffusion coefficient for state %s.\n' ...
      'Reduced mobility not available (it should be defined before calling this function).\n'...
      'Please, fix the problem and run the code again.'], state.name);
  end
  % apply Einstein relation to obtain the diffusion coefficient
  coefficient = redMob/Constant.electronCharge*Constant.boltzmann*temperature;
end
