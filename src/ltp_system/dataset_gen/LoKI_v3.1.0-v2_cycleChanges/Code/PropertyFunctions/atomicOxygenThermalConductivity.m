function thermalConductivity = atomicOxygenThermalConductivity(gas, ~, workCond)
  % atomicOxygenThermalConductivity (have to be writen)
  % taken from https://doi.org/10.1016/0032-0633(62)90064-8
  
  % thermal conductivity for atomic oxygen (SI units) % accurate in the interval [100,2000] K
  thermalConductivity = (67.1*workCond.gasTemperature^0.71)*1e-5;
  % change energy units to eV
  thermalConductivity = thermalConductivity/Constant.electronCharge;
  % store value on gas object properties
  gas.thermalConductivity = thermalConductivity;
  
  
end
