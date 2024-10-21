import os
import re
import pdb
import sys
import time
import random
import numpy as np
import pandas as pd


#-----------------------------------------------------------------------------------------------------
class Parameters():
    def __init__(self, npoints,  uniform_sample):
        self.n_points = npoints

        # Set of parameters
        self.k_set = None
        self.pressure_set = uniform_sample[:,0]
        self.current_set  = uniform_sample[:,1]
        self.radius_set = uniform_sample[:,2]
        self.electDensity_set = uniform_sample[:,3]
    
    def set_npoints(self, npoints):
        self.n_points = npoints

#-----------------------------------------------------------------------------------------------------
    
class Simulations():
    def __init__(self, npoints,  uniform_sample, config):
        chem_file = config['dataset_generation']['chem_file']
        self.chem_file_dir = config['dataset_generation']['chem_file_dir']
        
        self.chem_file = chem_file
        self.outptFolder = chem_file[:-5]

        self.loki_path = config['dataset_generation']['loki_path']
        self.nsimulations = npoints
        self.parameters = Parameters(npoints, uniform_sample)
        
        self._generateChemFiles= False
        self.setup_file = config['dataset_generation']['setup_file']
        self.setup_file_dir = config['dataset_generation']['setup_file_dir']
        self.dataset_dir = config['dataset_generation']['dataset_dir']

        
        # create input folder if does not exist
        dir = self.loki_path+ '/Code/Input/'+self.outptFolder
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Private methods
    def _genChemFiles(self):
        # Read in the example file
        with open(self.chem_file, 'r') as file :  
            lines = []
            for line in file:
                lines.append(line.strip())
            chemFiledata = lines

        # Replace for all self.parameters.k_set
        for j in range (0,self.nsimulations, 1): # for each datapoint 
            list_out = []
            for i in range(len(self.parameters.k_set[j])): # for each k (for each line in file)
                line = chemFiledata[i]
                # the regular expression matches a decimal number in scientific notation
                line = re.sub(r'\d+.\d+E[+,-]?\d+', "{:.12E}".format(self.parameters.k_set[j][i]), line)
                list_out.append(line)

            # Write the out chem file 
            outfile = open(self.loki_path+ '/Code/Input/'+self.outptFolder+'/'+self.outptFolder+'_' +str(j)+'.chem', 'w')
            outfile.write("\n".join(list_out))
            outfile.close()
            
    def _genSetupFiles(self):
        # Read in the example file
        with open(self.setup_file_dir, 'r') as file :
            setup_data = file.read() # (for the setup files we dont need to separate the string in lines)
        
        # Then replace for all self.parameters 
        for j in range (0, self.nsimulations, 1): 
            if self._generateChemFiles: 
                new_chemFile_name = self.outptFolder+ "\\\\"+ self.outptFolder+'_' +str(j)+ '.chem'
                setup_data = re.sub(r"chemFiles:\s*\n\s*- (.+?)\n", f"chemFiles:\n      - {new_chemFile_name}\n", setup_data) #replace chem file name
            setup_data = re.sub(r'folder:+\s+(\S+)', 'folder: ' + self.outptFolder +'_'+str(j), setup_data) #replace output folder name

            if self.parameters.pressure_set is not None:
                # replace the pressure value in the setup file, that follows the string "gasPressure: "
                setup_data = re.sub(r'gasPressure: \d+.\d+', 'gasPressure: ' + "{:.4f}".format(self.parameters.pressure_set[j]), setup_data)

            if self.parameters.radius_set is not None:
                # replace the radius value in the setup file, that follows the string "chamberRadius: "
                setup_data = re.sub(r'chamberRadius:\s*\d+\.?\d*e?-?\d*\s*%', f'chamberRadius: {self.parameters.radius_set[j]:.8f} %', setup_data)

            if self.parameters.electDensity_set is not None:
                # replace the radius value in the setup file, that follows the string "electronDensity:  "
                setup_data = re.sub(r'electronDensity:\s*\d+\.?\d*e?-?\d*\s*%', f'electronDensity: {self.parameters.electDensity_set[j]:.8f} %', setup_data)
            
            if self.parameters.current_set is not None:
                # replace the discharge current value in the setup file, that follows the string "dischargeCurrent: "
                setup_data = re.sub(r'dischargeCurrent:\s*[\d\.\*e+-]+', f'dischargeCurrent: {self.parameters.current_set[j]:.8f}', setup_data)


            # Write out the setUp files
            outfile = open(self.loki_path+ '/Code/Input/'+self.outptFolder+'/'+self.setup_file[:-3]+'_' +str(j)+'.in', 'w')
            outfile.write(setup_data)
            outfile.close()

    def _read_outputs(self):

        def readFileDensities(file_densities):
            with open(file_densities, 'r') as file :
                densities = []
                for line in file:
                    if line.startswith(' '):
                        densities.append(line.split()[1])
            return densities
        
        def readFileTemperatures(file_temperatures):
            with open(file_temperatures, 'r') as f:
                temperatures = []
                data = f.read()
            gas_temp = re.search(r'Gas Temperature = (\S+) \(K\)', data).group(1)
            near_wall_temp = re.search(r'Near wall temperature = (\S+) \(K\)', data).group(1)
            temperatures = [gas_temp, near_wall_temp]
            return temperatures

        def readFileSwarmParameters(file_swarm_parameters):
            with open(file_swarm_parameters, 'r') as f:
                swarm_parameters = []
                data = f.read()
            red_elec_field = re.search(r'Reduced electric field = (\S+) \(Td\)', data).group(1)
            drift_vel = re.search(r'Drift velocity = (\S+) \(ms\^-1\)', data).group(1)
            mean_energy = re.search(r'Mean energy = (\S+) \(eV\)', data).group(1)
            swarm_parameters = [red_elec_field, drift_vel, mean_energy]
            return swarm_parameters
        
        def readFileSetupParameters(file_setup_parameters):
            with open(file_setup_parameters, 'r') as f:
                setup_parameters = []
                data = f.read()
            electron_density = re.search(r'electronDensity: (\S+)', data).group(1)
            setup_parameters = [electron_density]
            return setup_parameters
        
        def readFileElectronDensity(file_densities):
            # Read the content from the file
            with open(file_densities, 'r') as file:
                data = file.read()

            # Use re.search to find the match
            electron_density = [re.search(r"Electrons\s+(\S+)", data).group(1)]
                
            return electron_density

        densities, temperatures, swarm_parameters, setup_parameters, ne_parameters = [], [], [], [], []
        outputs = np.empty(self.nsimulations, dtype=object)

        # Read data from all output folders
        for i in range(self.nsimulations):
            folder_path = self.loki_path + '/Code/Output/' + self.outptFolder + '_' + str(i) + '/'

            try:
                file_densities = folder_path + 'chemFinalDensities.txt'
                file_temperatures = folder_path + 'finalTemperatures.txt'
                file_swarm_parameters = folder_path + 'swarmParameters.txt'
                file_setup_parameters = folder_path + 'setup.txt'

                densities.append(readFileDensities(file_densities))
                temperatures.append(readFileTemperatures(file_temperatures))
                swarm_parameters.append(readFileSwarmParameters(file_swarm_parameters))
                setup_parameters.append(readFileSetupParameters(file_setup_parameters))
                ne_parameters.append(readFileElectronDensity(file_densities))

                # Append results
                outputs[i] = densities[i] + temperatures[i] + swarm_parameters[i] + ne_parameters[i]

            except FileNotFoundError:
                # Handle the case where the directory or file is not found
                print(f"Directory or file not found for simulation {i}. Appending NaN values.")
                
                # Append NaN values to densities, temperatures, swarm_parameters, and setup_parameters
                densities.append(np.nan)
                temperatures.append(np.nan)
                swarm_parameters.append(np.nan)
                setup_parameters.append(np.nan)
                ne_parameters.append(np.nan)

                # Set outputs[i] to NaN
                outputs[i] = densities[i] + temperatures[i] + swarm_parameters[i] + ne_parameters[i]

        # Convert lists to NumPy arrays
        densities = np.array(densities, dtype=object)
        temperatures = np.array(temperatures, dtype=object)
        swarm_parameters = np.array(swarm_parameters, dtype=object)
        setup_parameters = np.array(setup_parameters, dtype=object)
        ne_parameters = np.array(ne_parameters, dtype=object)
        outputs = np.array(outputs, dtype=object)
        
        return np.array(densities), np.array(temperatures), np.array(swarm_parameters), np.array(setup_parameters),np.array(outputs)


    # Public methods
    def runSimulations(self):
        #------------------Generate the Chemistry + SetUp files--------------------------#
        if self._generateChemFiles:
            self._genChemFiles()
        else:
            print('\nChemistry files are not generated. The following file is used for all simulations: ' + self.chem_file)
            # read the example file and write it to the input folder
            with open(self.chem_file_dir, 'r') as file:
                chemFiledata = file.read()
            outfile = open(self.loki_path +'/Code/Input/'+self.outptFolder+'/'+self.chem_file[:-5] +'.chem', 'w')
            outfile.write(chemFiledata)
            outfile.close()

        self._genSetupFiles()
        
        #--------------------------------------Run the matlab script---------------------#
        outfile = open(self.loki_path + "/Code/loop_config.txt", 'w')
        outfile.write(str(self.nsimulations)) # save nsimul for matlab script
        outfile.write("\n"+ self.outptFolder+'/'+self.setup_file[:-3]+'_') # save output folder name for matlab script
        """if(self.flag_dataset_type == 1):
            outfile.write("\n" + str(abs(i_fixed))) # save nsimul for matlab script
            outfile.write("\n" + str(R_fixed)) # save nsimul for matlab script"""
        outfile.close()

        print("Running stoped here. Please run the MatLab code to generate the outputs. \nIf you have run the MatLab script, press 'c' to continue. ")
        pdb.set_trace()

    def writeDataFile(self, filename):

        _, _, _, _, outputs = self._read_outputs()

        dir = self.dataset_dir

        if not os.path.exists(dir):
            os.makedirs(dir)

        # Write the data file
        with open(dir + filename, 'w') as file:
            for i in range(self.nsimulations):
                # check if the simulation went ok or has nan values
                if isinstance(outputs[i], list):
                    # write the k values
                    if self.parameters.k_set is not None:
                        k_line = self.parameters.k_set[i]
                        file.write(' '.join( "{:.12E}".format(item) for item in k_line))
                        file.write(' ')
                    
                    # write the pressure values
                    if self.parameters.pressure_set is not None:
                        file.write("{:.12E}".format(self.parameters.pressure_set[i]))
                        file.write(' ')

                    # write the current values
                    if self.parameters.current_set is not None:
                        file.write("{:.12E}".format(self.parameters.current_set[i]))
                        file.write(' ')
                    
                    # write the radius values
                    if self.parameters.radius_set is not None:
                        file.write("{:.12E}".format(self.parameters.radius_set[i]))
                        file.write(' ')

                    # write to dataset
                    file.write(' '.join( "{:.12E}".format(float(item)) for item in outputs[i])+'\n')


    # Setters and getters
    def set_nsimulations(self, n):
        self.nsimulations = n
        self.parameters.set_npoints(n)

    def set_ChemFile_OFF(self):
        self._generateChemFiles = False

    def set_ChemFile_ON(self):
        self._generateChemFiles = True
