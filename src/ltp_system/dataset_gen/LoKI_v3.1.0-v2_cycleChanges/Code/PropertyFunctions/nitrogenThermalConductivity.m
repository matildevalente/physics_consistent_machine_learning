function thermalConductivity = nitrogenThermalConductivity(gas, ~, workCond)
  % nitrogenThermalConductivity (have to be writen)
  
  % thermal conductivity for oxygen (SI units)
  thermalConductivity = (1.717+0.084*workCond.gasTemperature-1.948e-5*workCond.gasTemperature^2)*1e-3;
  % change energy units to eV
  thermalConductivity = thermalConductivity/Constant.electronCharge;
  % store value on gas object properties
  gas.thermalConductivity = thermalConductivity;
  
  
end
