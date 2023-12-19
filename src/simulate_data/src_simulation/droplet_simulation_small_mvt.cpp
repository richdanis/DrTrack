#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>
#include <cmath>
#include <tuple>
#include <chrono>
#include <omp.h>

class DropletSimulator {
public:
    // Data Types
    using T = float;
    using Tuple = std::tuple<T, T, T, T>;
    using Container = std::vector<Tuple>;

    // Constructor
    DropletSimulator();

    // Destructor
    ~DropletSimulator() {
        // Close output files
        std::cout.rdbuf(this->oldbuf_location_data);
        std::cout.rdbuf(this->oldbuf_meta_data);
        this->outfile_location_data.close();
        this->outfile_meta_data.close();
    }

    // Member functions
    void setup();
    void simulate();
    void store_meta_data();

    // Length of simulation
    int number_of_frames;
    int number_of_recordings;

    // Droplet parameters
    int num_droplets;
    int num_droplets_start;
    T droplet_radius;
    bool random_movement;
    bool intrinsic_movement;
    T max_random_velocity;
    T max_intrinsic_velocity;
    bool disappearing_droplets;
    T disappear_probability;

    // Screen parameters
    int screen_width;
    int screen_height;

    // Attraction point (positive focal point) parameters
    bool attraction_points;
    bool attraction_movement;
    int num_attractions;
    T attraction_radius;
    T attraction_strength;
    T attraction_strength_droplets; // Not being used as a variable at the moment
    T attraction_speed;

    // repulsion point (negative focal point) parameters
    bool repulsion_points;
    bool repulsion_movement;
    int num_repulsions;
    T repulsion_radius;
    T repulsion_strength;
    T repulsion_strength_droplets; // Not being used as a variable at the moment
    T repulsion_speed;

    // Giant droplet parameters
    bool larger_droplet;
    T larger_droplet_radius;
    T larger_droplet_speed_x;
    T larger_droplet_speed_y;
    T larger_droplet_x;
    T larger_droplet_y;
    T larger_droplet_dx;
    T larger_droplet_dy;
    T larger_droplet_start_x;
    T larger_droplet_start_y;

    // Output files
    std::string location_data_file;
    std::string meta_data_file;

private:
    // Data structures for circles, attraction points and repulsion points
    Container circles;
    Container attractions;
    Container repulsions;

    // Methods
    void _remove_droplets();
    void _generate_droplets();
    void _generate_attraction_points();
    void _generate_repulsion_points();
    void _move_giant_droplet();
    void _move_attraction_points();
    void _move_repulsion_points();
    void _move_circles(int start, int end);
    void _detect_collisions(int start, int end);

    // Random number generator and distributions
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis_prob;
    std::uniform_real_distribution<> dis_vel;
    std::uniform_real_distribution<> dis_loc_x;
    std::uniform_real_distribution<> dis_loc_y;
    std::uniform_real_distribution<> dis_vel_att;
    std::uniform_real_distribution<> dis_vel_rej;
    std::uniform_real_distribution<> dis_loc_att_x;
    std::uniform_real_distribution<> dis_loc_att_y;
    std::uniform_real_distribution<> dis_loc_rej_x;
    std::uniform_real_distribution<> dis_loc_rej_y;

    // Output files
    std::ofstream outfile_location_data;
    std::ofstream outfile_meta_data;
    std::streambuf *oldbuf_location_data;
    std::streambuf *oldbuf_meta_data;

    // Timer
    double total_runtime;
    double runtime_per_frame;
};

DropletSimulator::DropletSimulator()
    :   // Length of simulation
        number_of_frames(100),
        number_of_recordings(10),

        // Droplet parameters
        num_droplets(400),
        droplet_radius(11),
        random_movement(true),
        intrinsic_movement(true),
        max_random_velocity(0.1),
        max_intrinsic_velocity(0.1),
        disappearing_droplets(false),
        disappear_probability(0.01),

        // Screen parameters
        screen_width(1920),
        screen_height(1080),

        // Attraction point (positive focal point) parameters
        attraction_points(true),
        attraction_movement(true),
        num_attractions(15),
        attraction_radius(80),
        attraction_strength(0.2),
        attraction_strength_droplets(0.03),
        attraction_speed(1),

        // repulsion point (negative focal point) parameters
        repulsion_points(false),
        repulsion_movement(false),
        num_repulsions(10),
        repulsion_radius(80),
        repulsion_strength(0.2),
        repulsion_strength_droplets(0.03),
        repulsion_speed(1),

        // Giant droplet parameters
        larger_droplet(true),
        larger_droplet_radius(80),
        larger_droplet_speed_x(0.2),
        larger_droplet_speed_y(0.01),
        larger_droplet_start_x(screen_width / 5.),
        larger_droplet_start_y(screen_height / 2.),

        // Output files
        location_data_file("../data/location_data.csv"),
        meta_data_file("../data/meta_data.csv") {}

void DropletSimulator::setup() {
    // Set up random number generator and distributions
    this->gen = std::mt19937(this->rd());
    this->dis_prob = std::uniform_real_distribution<>(0, 1);
    this->dis_vel = std::uniform_real_distribution<>(-this->max_intrinsic_velocity, this->max_intrinsic_velocity);
    this->dis_loc_x = std::uniform_real_distribution<>(this->droplet_radius, this->screen_width - this->droplet_radius);
    this->dis_loc_y = std::uniform_real_distribution<>(this->droplet_radius, this->screen_height - this->droplet_radius);
    this->dis_vel_att = std::uniform_real_distribution<>(-this->attraction_speed, this->attraction_speed);
    this->dis_vel_rej = std::uniform_real_distribution<>(-this->repulsion_speed, this->repulsion_speed);
    this->dis_loc_att_x = std::uniform_real_distribution<>(this->attraction_radius, this->screen_width - this->attraction_radius);
    this->dis_loc_att_y = std::uniform_real_distribution<>(this->attraction_radius, this->screen_height - this->attraction_radius);
    this->dis_loc_rej_x = std::uniform_real_distribution<>(this->repulsion_radius, this->screen_width - this->repulsion_radius);
    this->dis_loc_rej_y = std::uniform_real_distribution<>(this->repulsion_radius, this->screen_height - this->repulsion_radius);

    // Open output files
    // Output files
    this->location_data_file = "../../data/simulation_data/loc_data_L" + std::to_string(this->larger_droplet) + "_A" + std::to_string(this->attraction_points) + "_R" + std::to_string(this->repulsion_points) + "_D" + std::to_string(this->disappearing_droplets) + "_M" + std::to_string(this->random_movement)+ "_" + std::to_string(this->num_droplets) + "_" + std::to_string(int(this->droplet_radius)) + ".csv";
    this->meta_data_file = "../../data/simulation_data/meta_data_L" + std::to_string(this->larger_droplet) + "_A" + std::to_string(this->attraction_points) + "_R" + std::to_string(this->repulsion_points) + "_D" + std::to_string(this->disappearing_droplets) + "_M" + std::to_string(this->random_movement)+ "_" + std::to_string(this->num_droplets) + "_" + std::to_string(int(this->droplet_radius)) + ".csv";
    
    // Create directory if not exists
    std::filesystem::path dir("../../data/simulation_data");
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    // Link to output files
    this->outfile_location_data.open(this->location_data_file);
    this->outfile_meta_data.open(this->meta_data_file);
    this->oldbuf_location_data = std::cout.rdbuf(outfile_location_data.rdbuf());
    this->oldbuf_meta_data = std::cout.rdbuf(outfile_meta_data.rdbuf());

    // Set up the giant droplet
    if (this->larger_droplet) {
        this->larger_droplet_x = this->larger_droplet_start_x;
        this->larger_droplet_y = this->larger_droplet_start_y;
        this->larger_droplet_dx = this->larger_droplet_speed_x;
        this->larger_droplet_dy = this->larger_droplet_speed_y;
    }

    // Set up the circles
    this->_generate_droplets();
    this->num_droplets_start = this->num_droplets;

    // Set up the attraction points
    if (this->attraction_points) {
        this->_generate_attraction_points();
    }

    // Set up the repulsion points
    if (this->repulsion_points) {
        this->_generate_repulsion_points();
    }
}

void DropletSimulator::store_meta_data() {
    // Store meta data
    outfile_meta_data << "number_of_frames,number_of_recordings,num_droplets,screen_width,screen_height,droplet_radius,max_random_velocity,max_intrinsic_velocity,disappearing_droplets,disappear_probability,attraction_points,attraction_movement,num_attractions,attraction_radius,attraction_strength,attraction_strength_droplets,attraction_speed,repulsion_points,repulsion_movement,num_repulsions,repulsion_radius,repulsion_strength,repulsion_strength_droplets,larger_droplet,larger_droplet_radius,larger_droplet_speed_x,larger_droplet_speed_y,larger_droplet_start_x,larger_droplet_start_y,location_data_file,meta_data_file,total_runtime,runtime_per_frame" << std::endl;
    outfile_meta_data << this->number_of_frames << "," << this->number_of_recordings << "," << this->num_droplets << "," << this->screen_width << "," << this->screen_height << "," << this->droplet_radius << "," << this->max_random_velocity << "," << this->max_intrinsic_velocity << "," << this->disappearing_droplets << "," << this->disappear_probability << "," << this->attraction_points << "," << this->attraction_movement << "," << this->num_attractions << "," << this->attraction_radius << "," << this->attraction_strength << "," << this->attraction_strength_droplets << "," << this->attraction_speed << "," << this->repulsion_points << "," << this->repulsion_movement << "," << this->num_repulsions << "," << this->repulsion_radius << "," << this->repulsion_strength << "," << this->repulsion_strength_droplets << "," << this->larger_droplet << "," << this->larger_droplet_radius << "," << this->larger_droplet_speed_x << "," << this->larger_droplet_speed_y << "," << this->larger_droplet_start_x << "," << this->larger_droplet_start_y << "," << this->location_data_file << "," << this->meta_data_file << "," << this->total_runtime << "," << this->runtime_per_frame << std::endl;
}

void DropletSimulator::simulate() {
    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Main game loop
    int counter = 0;

    // Print header row with droplet ids    
    outfile_location_data << "Id_Loc";
    for (int i = 0; i < this->num_droplets_start; i++) {
        outfile_location_data << "," << "D" << i << "_x," << "D" << i << "_y";
    }

    // Print header row with larger droplet id
    if (this->larger_droplet) {
        outfile_location_data << "," << "L_x" << "," << "L_y";
    }

    // Print header row with attraction point ids
    if (this->attraction_points) {
        for (int i = 0; i < this->num_attractions; i++) {
            outfile_location_data << "," << "A" << i << "_x," << "A" << i << "_y";
        }
    }

    // Print header row with repulsion point ids
    if (this->repulsion_points) {
        for (int i = 0; i < this->num_repulsions; i++) {
            outfile_location_data << "," << "R" << i << "_x," << "R" << i << "_y";
        }
    }
    outfile_location_data << std::endl;


    // Run as long as the counter is running, the number of frames is reached and there are still droplets
    while (counter < this->number_of_frames && this->num_droplets > 0) {
        // Print current frame
        // Update dataframe with position data
        if (counter % (this->number_of_frames / this->number_of_recordings) == 0 || counter == 1) {
            // Print circle positions
            outfile_location_data << counter;

            for (int i = 0; i < this->num_droplets_start; i++) {
                outfile_location_data << "," << std::get<0>(this->circles[i]) << "," << std::get<1>(this->circles[i]);
            }

            // Print giant droplet position if it exists
            if (this->larger_droplet) {
                outfile_location_data << "," << this->larger_droplet_x << "," << this->larger_droplet_y;
            }

            // Print attraction points
            if (this->attraction_points) {
                for (int i = 0; i < this->num_attractions; i++) {
                    outfile_location_data << "," << std::get<0>(this->attractions[i]) << "," << std::get<1>(this->attractions[i]);
                }
            }

            // Print repulsion points
            if (this->repulsion_points) {
                for (int i = 0; i < this->num_repulsions; i++) {
                    outfile_location_data << "," << std::get<0>(this->repulsions[i]) << "," << std::get<1>(this->repulsions[i]);
                }
            }

            outfile_location_data << std::endl;
        }

        // Move the giant droplet
        if (this->larger_droplet) {
            this->_move_giant_droplet();
        }

        // Remove droplets
        if (this->disappearing_droplets) {
            this->_remove_droplets();
        }

        // Move the circles
        this->_move_circles(0, this->num_droplets);

        // Move the attraction points
        if (this->attraction_points && this->attraction_movement) {
            this->_move_attraction_points();
        }

        // Move the repulsion points
        if (this->repulsion_points && this->repulsion_movement) {
            this->_move_repulsion_points();
        }
        
        // Update counter
        counter += 1;
    }

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    this->total_runtime = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    this->runtime_per_frame = this->total_runtime / this->number_of_frames;
}

void DropletSimulator::_remove_droplets() {
    // Remove droplets with certain probability
    if (this->dis_prob(this->rd) < this->disappear_probability) {
        this->circles[this->num_droplets-1] = std::make_tuple(-1,-1,0,0);
        this->num_droplets -= 1;
    }
}

void DropletSimulator::_generate_droplets() {
    for (int i = 0; i < this->num_droplets; i++) {
        double x = this->dis_loc_x(this->gen);
        double y = this->dis_loc_y(this->gen);
    
        if (this->larger_droplet){ 
            T dist = sqrt(pow(x - this->larger_droplet_x, 2) + pow(y - this->larger_droplet_y, 2));
            // if (dist < this->larger_droplet_radius + this->droplet_radius){
            if (dist < this->larger_droplet_radius + this->droplet_radius){ //|| (y < this->screen_height / 3. && i % 3 == 0) || (y < 2*this->screen_height / 3. && i % 5 == 0)){
                i--;
                continue;
            }
        }

        T dx = 0;
        T dy = 0;

        if (this->intrinsic_movement) {
            dx = this->dis_vel(this->gen);
            dy = this->dis_vel(this->gen);
        }

        this->circles.push_back(std::make_tuple(x, y, dx, dy));
    }
}

void DropletSimulator::_generate_attraction_points() {
    for (int i = 0; i < this->num_attractions; i++) {
        T x = this->dis_loc_att_x(this->gen);
        T y = this->dis_loc_att_y(this->gen);

        T dx = 0;
        T dy = 0;

        if (this->attraction_movement) {
            dx = this->dis_vel_att(this->gen);
            dy = this->dis_vel_att(this->gen);
        }
       
        this->attractions.push_back(std::make_tuple(x, y, dx, dy));
    }
}

void DropletSimulator::_generate_repulsion_points() {
    for (int i = 0; i < this->num_repulsions; i++) {
        T x = this->dis_loc_rej_x(this->gen);
        T y = this->dis_loc_rej_y(this->gen);

        T dx = 0;
        T dy = 0;

        if (this->repulsion_movement) {
            dx = this->dis_vel_rej(this->gen);
            dy = this->dis_vel_rej(this->gen);
        }
        
        this->repulsions.push_back(std::make_tuple(x, y, dx, dy));
    }
}

void DropletSimulator::_move_giant_droplet() {
    // Check whether large droplet is already out of range. If not, let it move.
    if(this->larger_droplet_x < this->screen_width && this->larger_droplet_x > 0 && this->larger_droplet_y < this->screen_height && this->larger_droplet_y > 0){
        this->larger_droplet_x += this->larger_droplet_dx;
        this->larger_droplet_y += this->larger_droplet_dy;
    }   
}

void DropletSimulator::_move_attraction_points() {
    for (int i = 0; i < this->num_attractions; i++) {
        T x, y, dx, dy;
        std::tie(x, y, dx, dy) = this->attractions[i];

        if (x < this->attraction_radius || x > this->screen_width - this->attraction_radius) {
            dx = -dx;
        }
        if (y < this->attraction_radius || y > this->screen_height - this->attraction_radius) {
            dy = -dy;
        }

        x += dx;
        y += dy;
        this->attractions[i] = std::make_tuple(x, y, dx, dy);
    }
}

void DropletSimulator::_move_repulsion_points() {
    for (int i = 0; i < this->num_repulsions; i++) {
        T x, y, dx, dy;
        std::tie(x, y, dx, dy) = this->repulsions[i];

        if (x < this->repulsion_radius || x > this->screen_width - this->repulsion_radius) {
            dx = -dx;
        }
        if (y < this->repulsion_radius || y > this->screen_height - this->repulsion_radius) {
            dy = -dy;
        }

        x += dx;
        y += dy;
        this->repulsions[i] = std::make_tuple(x, y, dx, dy);
    }
}

void DropletSimulator::_move_circles(int start, int end) {
    for (int i = start; i < end; i++) {
        T x, y, dx, dy;
        std::tie(x, y, dx, dy) = this->circles[i];

        // Apply current speed
        x += dx;
        y += dy;

        // Apply random movement
        if (this->random_movement) {
            x += this->dis_vel(this->gen);
            y += this->dis_vel(this->gen);
        }

        // Apply attraction forces
        if (this->attraction_points) {
            T ax, ay, adx, ady;
            for (int j = 0; j < this->num_attractions; j++) {
                std::tie(ax, ay, adx, ady) = this->attractions[j];
                T dist = sqrt(pow(x - ax, 2) + pow(y - ay, 2));
                if (dist < this->attraction_radius) {
                    T angle = atan2(ay - y, ax - x);
                    x += this->attraction_strength * cos(angle);
                    y += this->attraction_strength * sin(angle);
                }
            }
        }

        // Apply repulsion forces
        if (this->repulsion_points) {
            T rx, ry, rdx, rdy;
            for (int j = 0; j < this->num_repulsions; j++) {
                std::tie(rx, ry, rdx, rdy) = this->repulsions[j];
                T dist = sqrt(pow(x - rx, 2) + pow(y - ry, 2));
                if (dist < this->repulsion_radius) {
                    T angle = atan2(ry - y, rx - x);
                    x -= this->repulsion_strength * cos(angle);
                    y -= this->repulsion_strength * sin(angle);
                }
            }
        }


        // Avoid the giant droplet
        if (this->larger_droplet) {
            T dist = sqrt(pow(x - this->larger_droplet_x, 2) + pow(y - this->larger_droplet_y, 2));
            if (dist < this->larger_droplet_radius + this->droplet_radius) {
                // T diff_x = x - this->larger_droplet_x;
                // T diff_y = y - this->larger_droplet_y;

                // // Avoid division by zero
                // if (dist < 0e-8) {
                //     dist = 0.0001;
                // }

                double angle = atan2(y - this->larger_droplet_y, x - this->larger_droplet_x);
                x = this->larger_droplet_x + cos(angle) * (this->larger_droplet_radius + this->droplet_radius);
                y = this->larger_droplet_y + sin(angle) * (this->larger_droplet_radius + this->droplet_radius);
                // T ndiff_x = diff_x / dist;
                // T ndiff_y = diff_y / dist;
                // x = this->larger_droplet_x + ndiff_x * (this->larger_droplet_radius + this->droplet_radius);
                // y = this->larger_droplet_y + ndiff_y * (this->larger_droplet_radius + this->droplet_radius);
            }
        }

        // Check for collisions with other circles
        #pragma omp parallel for
        for (int j = 0; j < this->num_droplets; j++) {
            T x2, y2, dx2, dy2;
            std::tie(x2, y2, dx2, dy2) = this->circles[j];

            // Do not bother checking for collisions with itself or if the circles are too far apart or if the circle is already removed
            if (i == j || std::abs(x-x2) > this->droplet_radius * 2 || std::abs(y-y2) > this->droplet_radius * 2 || ((x2==-1) && (y2==-1))) {
                continue;
            }

            T dist = sqrt(pow(x - x2, 2) + pow(y - y2, 2));

            if (dist < this->droplet_radius * 2) {
                // Adjust velocities to prevent overlap
                // T diff_x = x2 - x;
                // T diff_y = y2 - y;

                // // Avoid division by zero
                // if (dist < 0e-8) {
                //     dist = 0.0001;
                // }

                // T ndiff_x = diff_x / dist;
                // T ndiff_y = diff_y / dist;
                // x2 = x + ndiff_x * this->droplet_radius * 2;
                // y2 = y + ndiff_y * this->droplet_radius * 2;

                double angle = atan2(y2 - y, x2 - x);
                x2 = x + cos(angle) * (2 * this->droplet_radius);
                y2 = y + sin(angle) * (2 * this->droplet_radius);

                // Check for collisions with walls
                if (x2 < this->droplet_radius) {
                    x2 = this->droplet_radius;
                    // Change direction of speed if collision with wall
                    dx2 = -dx2;
                } else if (x2 > this->screen_width - this->droplet_radius) {
                    x2 = this->screen_width - this->droplet_radius;
                    dx2 = -dx2;
                }

                if (y2 < this->droplet_radius) {
                    y2 = this->droplet_radius;
                    dy2 = -dy2;
                } else if (y2 > this->screen_height - this->droplet_radius) {
                    y2 = this->screen_height - this->droplet_radius;
                    dy2 = -dy2;
                }
                this->circles[j] = std::make_tuple(x2, y2, dx2, dy2);
            }

        }

        // Check for collisions with walls
        if (x < this->droplet_radius) {
            x = this->droplet_radius;
            // Change direction of speed if collision with wall
            dx = -dx;
        } else if (x > this->screen_width - this->droplet_radius) {
            x = this->screen_width - this->droplet_radius;
            dx = -dx;
        }

        if (y < this->droplet_radius) {
            y = this->droplet_radius;
            dy = -dy;
        } else if (y > this->screen_height - this->droplet_radius) {
            y = this->screen_height - this->droplet_radius;
            dy = -dy;
        }

        // Move circles
        this->circles[i] = std::make_tuple(x, y, dx, dy);
    }
}


int main() {
    // Set up data generator
    DropletSimulator* droplet_simulator = new DropletSimulator();

    /*
    SIMULATION 1
    */
    // Set parameters
    // Length of simulation
    droplet_simulator->number_of_frames = 1000;
    droplet_simulator->number_of_recordings = 12;

    // Droplet parameters
    droplet_simulator->num_droplets = 20000;
    droplet_simulator->droplet_radius = 1.55;
    droplet_simulator->random_movement = false;
    droplet_simulator->intrinsic_movement = false;
    droplet_simulator->max_random_velocity = 0.05;
    droplet_simulator->max_intrinsic_velocity = 0.05;

    // Disappearing droplets
    droplet_simulator->disappearing_droplets = false;
    droplet_simulator->disappear_probability = 0.3;

    // Screen parameters
    droplet_simulator->screen_width = 500;
    droplet_simulator->screen_height = 500;

    // Attraction point (positive focal point) parameters
    droplet_simulator->attraction_points = true;
    droplet_simulator->attraction_movement = true;
    droplet_simulator->num_attractions = 4;
    droplet_simulator->attraction_radius = 30;
    droplet_simulator->attraction_strength = 0.01;
    droplet_simulator->attraction_strength_droplets = 0.01;
    droplet_simulator->attraction_speed = 1;

    // repulsion point (negative focal point) parameters
    droplet_simulator->repulsion_points = true;
    droplet_simulator->repulsion_movement = true;
    droplet_simulator->num_repulsions = 4;
    droplet_simulator->repulsion_radius = 30;
    droplet_simulator->repulsion_strength = 0.01;
    droplet_simulator->repulsion_strength_droplets = 0.015; // Not being used as a variable at the moment
    droplet_simulator->repulsion_speed = 1;

    // Giant droplet parameters
    droplet_simulator->larger_droplet = false;
    droplet_simulator->larger_droplet_radius = 50;
    droplet_simulator->larger_droplet_speed_x = 0.05;
    droplet_simulator->larger_droplet_speed_y = -0.25;
    droplet_simulator->larger_droplet_start_x = 1.2*droplet_simulator->screen_width / 5.;
    droplet_simulator->larger_droplet_start_y = 5.8*droplet_simulator->screen_height / 6.;

    
    // Run simulation
    droplet_simulator->setup();
    droplet_simulator->simulate();
    droplet_simulator->store_meta_data();

    delete droplet_simulator;
    return 0;
}