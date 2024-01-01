# Track Droplets

This script, `track_droplets.py`, is used to track droplets in a given image or video file. It utilizes computer vision techniques to detect and track droplets over time.

## Prerequisites

Before using this script, make sure you have the following prerequisites installed:

- Python 3.x
- OpenCV library
- NumPy library

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/track-droplets.git
    ```

2. Install the required dependencies:

    ```bash
    pip install opencv-python numpy
    ```

## Usage

To use the `track_droplets.py` script, follow these steps:

1. Open a terminal and navigate to the project directory:

    ```bash
    cd /path/to/track-droplets
    ```

2. Run the script with the desired configuration options:

    ```bash
    python track_droplets.py --input /path/to/input_file --output /path/to/output_file --threshold 100 --min_area 50
    ```

    - `--input`: Path to the input image or video file.
    - `--output`: Path to save the output file with tracked droplets.
    - `--threshold`: Threshold value for droplet detection (default: 100).
    - `--min_area`: Minimum area of a droplet to be considered (default: 50).

3. Wait for the script to process the input file and track the droplets. The output file will be saved at the specified location.

## Example

Here's an example command to track droplets in a video file:
