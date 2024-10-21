function coefficient = generalizedTemperatureDependentCoeff(state, argumentArray, workCond)
  % generalizedTemperatureDependentCoeff(have to be writen)
  
  normalizationConstant = argumentArray{1};
  temperature = argumentArray{2};
  power = argumentArray{3};
  if ~isnumeric(temperature)
    switch temperature
      case 'gasTemperature'
        temperature = workCond.gasTemperature;
      case 'electronTemperature'
        temperature = workCond.electronTemperature;
      otherwise
        error(['Error found when evaluating coefficient of state %s.\nTemperature ''%s'' not defined in the ' ...
          'working conditions.\nPlease, fix the problem and run the code again.'], state.name, temperature);
    end
  end
  
  coefficient = normalizationConstant*temperature^power;
  
end
