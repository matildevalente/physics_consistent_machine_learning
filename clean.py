import os
import shutil

def flush_model_artifacts(system):
    """
    Handles model flushing (cleaning checkpoints, figures, and artifacts) and training setup.
    Returns the retrain flag.
    
    Returns:
        tuple: (retrain_model (bool))
    """

    # Define directory mappings for each system
    system_directories = {
        'ltp': [
            'output/ltp_system/checkpoints/',
            'src/ltp_system/figures/EDA/',
            'src/ltp_system/figures/Figures_4/',
            'src/ltp_system/figures/Figures_6a/',
            'src/ltp_system/figures/Figures_6b/',
            'src/ltp_system/figures/Figures_6d/',
            'src/ltp_system/figures/loss_curves/',
        ],
        'spring': [
            'output/spring_mass_system/checkpoints/',
            'src/spring_mass_system/figures/loss_curves/',
            'src/spring_mass_system/figures/several_initial_conditions/',
            'src/spring_mass_system/figures/single_initial_condition/'
        ]
    }
    
    print("──────────────────────────────────────────────────────────────────────────────")
    
    confirmation = input(f"\nAre you sure you want to clean all the plots and tables from the system directories? (y/n): ")
    
    if confirmation.lower() in ['y', 'yes']:
        print("[INFO] Performing system cleanup...")
        directories = system_directories.get(system)
        if not directories:
            print(f"Error: Unknown system choice '{system}'")
            return None
        
        for directory in directories:
            try:
                if os.path.exists(directory):
                    # Remove all contents of the directory
                    for filename in os.listdir(directory):
                        file_path = os.path.join(directory, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                            print(f"Cleaned: {file_path}")
                        except Exception as e:
                            print(f"Failed to delete {file_path}. Reason: {e}")
                else:
                    print(f"Directory not found: {directory}")
            except Exception as e:
                print(f"Error processing directory {directory}: {e}")
        
        print(f"\n[INFO] Cleaning complete for {system.upper()} system!")
        print("[INFO] Initiating model retraining sequence...")
        print("──────────────────────────────────────────────────────────────────────────────\n")
        retrain = True
        return retrain
    else:
        
        system = None
        retrain = False
        print("[INFO] Directory cleaning cancelled")
        print("[INFO] Using existing model weights and pre-computed results ...")
        print("──────────────────────────────────────────────────────────────────────────────\n")
        return retrain
    

def flush_data():
    """
    Handles dataset flushing in the spring mass system.
    Returns the regenerate_data flag.
    
    Returns:
        tuple: (regenerate_data (bool))
    """

    # Define dataset directory
    directory = 'data/spring_mass_system/'
    
    print("──────────────────────────────────────────────────────────────────────────────")
    
    confirmation = input(f"\nAre you sure you want to delete the dataset? (y/n): ")
    
    if confirmation.lower() in ['y', 'yes']:
        print("[INFO] Deleting existing dataset ...")
        
        try:
            if os.path.exists(directory):
                # Remove all contents of the directory
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        print(f"Cleaned: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
            else:
                print(f"Directory not found: {directory}")
        except Exception as e:
            print(f"Error processing directory {directory}: {e}")
        
        print(f"\n[INFO] Cleaning complete!")

        # True for regenerating data
        return True
    else:
        
        print("[INFO] Dataset cleaning cancelled")
        print("[INFO] Using existing data to train the model and compute results ...")
        print("──────────────────────────────────────────────────────────────────────────────\n")
        return False  


def flush_lambda_study(system):
    """
    Handles flushing of the lambda analysis results in the spring mass system.
    Returns the regenerate_data flag.
    
    Returns:
        tuple: (regenerate_data (bool))
    """

    # Define dataset directory
    if(system == 'ltp'):
        directory = 'src/ltp_system/figures/additional_results/'

    elif(system == 'spring'):
        directory = 'src/spring_mass_system/figures/additional_results/'
    
    print("──────────────────────────────────────────────────────────────────────────────")
    
    confirmation = input(f"\nDo you want to delete the lambda_physics study results? (y/n): ")
    
    if confirmation.lower() in ['y', 'yes']:
        print("[INFO] Deleting existing lambda_physics results ...")
        
        try:
            if os.path.exists(directory):
                # Remove all contents of the directory
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        print(f"Cleaned: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
            else:
                print(f"Directory not found: {directory}")
        except Exception as e:
            print(f"Error processing directory {directory}: {e}")
        
        print(f"\n[INFO] Cleaning complete!")

        # True for regenerating data
        return True
    else:
        
        print("[INFO] Using lambda_physics existing results ...")
        print("──────────────────────────────────────────────────────────────────────────────\n")
        return False  
