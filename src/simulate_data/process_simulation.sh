# Process the simulation data with Python
# This will store the location data in the data/ot_data as well as store plots of the data in the data/plots folder
# cd /home/mike/Masters_DS/dsl_2023_all/dsl_2023/ground_truth_generation

# LOCATION_DATA=data/simulation_data/loc_data_L0_A1_R1_D0_M0_6000_2.csv
# META_DATA=data/simulation_data/meta_data_L0_A1_R1_D0_M0_6000_2.csv

# #python src_simulation/process_simulation_data.py $LOCATION_DATA $META_DATA

# LOCATION_DATA=data/simulation_data/loc_data_L1_A1_R1_D0_M0_6000_2.csv
# META_DATA=data/simulation_data/meta_data_L1_A1_R1_D0_M0_6000_2.csv

# python src_simulation/process_simulation_data.py $LOCATION_DATA $META_DATA

# LOCATION_DATA=data/simulation_data/loc_data_L1_A1_R1_D0_M1_6000_2.csv
# META_DATA=data/simulation_data/meta_data_L1_A1_R1_D0_M1_6000_2.csv

# python src_simulation/process_simulation_data.py $LOCATION_DATA $META_DATA


LOCATION_DATA=data/simulation_data/loc_data_L0_A1_R1_D0_M0_20000_1.csv
META_DATA=data/simulation_data/meta_data_L0_A1_R1_D0_M0_20000_1.csv

python src_simulation/process_simulation_data.py $LOCATION_DATA $META_DATA

LOCATION_DATA=data/simulation_data/loc_data_L1_A1_R1_D0_M0_20000_1.csv
META_DATA=data/simulation_data/meta_data_L1_A1_R1_D0_M0_20000_1.csv

python src_simulation/process_simulation_data.py $LOCATION_DATA $META_DATA

LOCATION_DATA=data/simulation_data/loc_data_L1_A1_R1_D0_M1_20000_1.csv
META_DATA=data/simulation_data/meta_data_L1_A1_R1_D0_M1_20000_1.csv

python src_simulation/process_simulation_data.py $LOCATION_DATA $META_DATA
