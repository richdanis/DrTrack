# Build the simulation with CMake: Make sure to run run_simulation.sh in the ground_truth_generation or cd to the ground_truth_generation folder.
# cd /home/mike/Masters_DS/dsl_2023_all/dsl_2023/ground_truth_generation
cd src_simulation
mkdir -p build
cd build
cmake ..
make

# Edit the parameters in main of the droplet_simulation.cpp file as wished
# Run the simulation. The data will be stored in data/simulation_data
echo Simulation Starting
./droplet_simulation_small
./droplet_simulation_medium
./droplet_simulation_large
echo Simulation Complete
cd ..
