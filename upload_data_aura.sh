#!/bin/bash

# --- PREREQUISITES ---
# 1. You must connect first the eduVPN (https://it.muni.cz/sluzby/eduvpn)
# 2. Also you must use this command onces to upload the dataset to the server (REMEMBER TO CHANGE THE UCO!):
    # scp -r data evaluator evaluate.sh run.sh x579419@aura.fi.muni.cz:/home/x579419/NeuralNetworks/

# Change the UCO
UCO="x579419"

# Copy the source files
scp -r src ${UCO}@aura.fi.muni.cz:/home/${UCO}/NeuralNetworks/


# Necessary steps AFTER executing this file (!)

# 1. Connect to the server
    # ssh ${UCO}@aura.fi.muni.cz

# 2. (Maybe not needed) Once connected, you have to give permissions to the "run.sh" file
    # cd NeuralNetworks/
    # cmod +x run.sh

# 3. Execute the project
    # ./run.sh

# 4. Evaluate the accuracy of the execution
    # ./evaluate.sh

# IMPORTANT NOTE: You may need to delete the files from AURA server if you have deleted on the repository and but didn't on Aura server.