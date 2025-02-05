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
            'src/ltp_system/figures/'
        ],
        'spring': [
            'output/spring_mass_system/checkpoints/',
            'src/spring_mass_system/figures/'
        ]
    }
    
    print("──────────────────────────────────────────────────────────────────────────────")
    
    confirmation = input(f"\nAre you sure you want to clean all the plots and tables from the LTP system directories? (y/n): ")
    
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