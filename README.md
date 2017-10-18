# FORK adaptation for RealSense

This project works with Intel's line of RealSense depth sensors to create a cross-platform system for determining occupancy in a room. Currently, the only platforms supported are macOS and Linux. This code is currently designed to work with the RealSense ZR300. 

## Getting Started


### Prerequisites
* USB 3.0 port
* [Intel RealSense ZR300](https://click.intel.com/intelr-realsensetm-development-kit-featuring-the-zr300.html)
* [OpenCV](http://opencv.org/) (v2.4.13+)
* [Intel RealSense Cross Platform API](https://github.com/IntelRealSense/librealsense)

**Note: **An extension cable cannot be used with the depth sensor.

###Calibration
The header file, "FORK_RS.hpp", contains all the options and parameters.

The following parameters **must** be accurate for the program to function correctly: max\_people\_height, min\_people\_height, max\_center\_diff, min\_radius\_threshold, max\_radius\_threshold, min\_area\_threshold. 

This code is currently calibrated for: **9ft**

### Running


Compile and run the code:

 `make`

`./fork_imp`

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

