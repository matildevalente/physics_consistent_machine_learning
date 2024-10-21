function thermalConductivity = oxygenThermalConductivity(gas, ~, workCond)
  % oxygenThermalConductivity (have to be writen)
  
  % thermal conductivity for oxygen (SI units)
  thermalConductivity = (1.056+0.087*workCond.gasTemperature-8.912e-6*workCond.gasTemperature^2)*1e-3;
  % change energy units to eV
  thermalConductivity = thermalConductivity/Constant.electronCharge;
  % store value on gas object properties
  gas.thermalConductivity = thermalConductivity;
  
  
end
