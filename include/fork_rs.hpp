#include "fork_logger.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <time.h>
#include <pthread.h>
#include <queue>
#include <math.h>
#include <csignal>
#include <librealsense/rs.hpp>
#include <signal.h>

#define PI 3.14159265359
#define PAST_HORIZON_SIZE 15
#define MAX_PEOPLE 25 // we are not tracking more than 25 people
#define MAX_HEADS 25 //no more than 25 heads will be seen in a frame
#define INF 2147483647
#define VIDEO_FPS 8
#define IMAGE_DIVIDE 4500.0
#define MAX_DEPTH 2743  //if a depth is more than this, this is an outlier. Kinect v2 has maximum distance of 4.5 meter = 4.5*39.37*2.54*10 = 4499.99 mm
#define HEIGHT_ARRAY_SIZE 50
#define INF_HEIGHT 9999999
#define FPS_MAX_LEN 20
#define TOTAL_BINS 300
#define MAX_SQR 70000  //we are precomputing the square root of up to 70000
#define NUM_THREADS 10
#define MAX_HEIGHT_LEVELS 20

#define FT_TO_MM(x) (x * 304.8) // converts feet to millimeters
#define MM_TO_FT(x) (x / 304.8 )// converts millimeters to feet
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////General Parameters/////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
bool match_by_velocity = false;
bool count_opposite_way = true;
bool display_occupancy = true;
bool enable_threading = false;
bool print_debug_info = false;
bool display_polys = false;
bool decrease_fps = false;
bool log_data = false;
bool store_frame = false;

int fps_divisor = 6; // camera_fps / divisor;

bool done_capturing = false;
bool done_processing = false;
bool realsense_shutdown = false;
bool use_realsense = false;
bool enable_out_of_order_height_check = false;  //if this is true we try with the "top_k_heights" first, then try the lowest height, and then check in between
int top_k_heights = 4;    //when trying with different heights, these heights are considered first. Then we try the last height and try in between.

int max_height_limit = FT_TO_MM(9); //anything higher than 9 feet is an outlier
int min_height_limit = FT_TO_MM(1.8); // anything less than 1.8 ft is an outler

double max_people_height = 6.5; // 5.5 feet
double min_people_height = 3.5; //min_people_height will not be considered..we will start from min_people_height + height decrement
double height_decrement = 0.5; //0.5 feet
double max_center_diff = 150; //if the center of two circles is less than 200 pixels for different height thresholds, then they are considered the same circle and the circle with the topmost height is considered
int total_diff_heights = ceil((double)(max_people_height - min_people_height)/height_decrement);

int freq_distribution_threshold = 0; //1100 //if the frequency of depths for a bin is less than this, we will not do height pruning for this height

bool discard_height_data_above = false; //if true, for height 5 feet, it will discard all the depth below 5 feet and above 5.8 feet
float height_prune_above_threshold = 0.8*12*2.54*10; //0.8 feet

bool use_canny = true; // if true, it will do canny edge detection
bool suppress_images = true;
bool write_video = false;
bool dual_threshold_for_contours = false;



///////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////FILE I/O/////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

// rowe_box_start = 110
int startFrame = 0; // 1100 // 0 for multi
int endFrame = 770;  //9000 753 w/o cap // 852 multi people //770 for rowe w/ box

int folder_location = 6;
// 6 - Groundtruth data set
// 111 - Rakesh, Tall case
// 91 - Bhumika, Short case
// 81 - Different test cases

//#define IMG_LOC "/Volumes/Seagate Backup Plus Drive/Projects/Kinect/Images/%d/depth/depth_%d.png"
//#define IMG_LOC "/Users/Gladiator/Documents/Research/Occupancy/Kinect/libfreenect2/examples/images/from_seagate/%d/depth/depth_%d.png"
//#define IMG_LOC "/Volumes/Seagate Backup Plus Drive/Projects/Kinect/Images/%d/depth/depth_%d.png"
#define IMG_LOC  "/Users/PapaYaw/Documents/Bosch_Research/images/16_bit/%d.png"

//string groundTruthFile = "/Users/Gladiator/Documents/workspace/kinect/FORK_NOBGS/src/headGT.csv";
std::string groundTruthFile = "/Users/Apple/Desktop/Bosch/headGT.csv";

//string occGroundTruthFile = "/Users/Gladiator/Documents/workspace/kinect/FORK_NOBGS/src/occupancy_contour2.csv";
std::string occGroundTruthFile = "/Users/Apple/Desktop/Bosch/occupancy.csv";



///////////////////////////////////////////////////////////////////////////////////////
///////////////////////Background Determination///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

int same_shoulder_thresh = 25; // ~.5 inch different between depth of shoulder contours allowed
bool perform_background_subtraction = true; // if true, it will do background subtraction
// 7 frames rowe_box
// 20 for multi peple
int number_of_frames_for_background_determination = 10; //it uses the first 200 frames to determine the background...if someone is there, he should be out of there within this 200 frames
int background_determination_threshold = 1500; //if the sum of histogram difference is less than 1500, then it is a background
bool background_image_determined = false; // after the background image is determined, it will be set to true
cv::Mat background_image;   //it contains the background image after it is determined

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////Human Detection General///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
bool check_smaller_contours = false; // Enables the checking of smaller contours in a region where the contour size found was bigger than the maximum head radius threshold.
bool throat_check = false; // will enable or disable  throat checking
bool head_check = false; // will enable or disable head checking
bool shoulder_check = true; // will enable or disable shoulder checking
int no_of_checks = 1; // If the number of checks i.e. head, throat and shoulder pass this value then we will add a head in the algorithm. Make the count equal to the number of checks to make each check decisive.

int head_counter = 0; // Number of head checks passed in the frame sequence. If a head passes the head check then this counter is incremented.
int shoulder_counter = 0;// Number of shoulder checks passed in the frame sequence. If a head passes the shoulder check then this counter is incremented.
int throat_counter = 0;// Number of throat checks passed in the frame sequence. If a head passes the throat check then this counter is incremented.


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////Head Detection///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

int error_thresh = 40; // 40 for RT. If the depth per pixel error in a contour is higher than this, it is not considered as a head.
int rad_z = 110; //The depth of the head is assumed to be around 25 cm, so this value is set to half of that = 12.5 cm = 125 mm. A sphere will have a 3D center where the (x,y) points will be given by the center of the contour and the z value is not yet determined and is assumed to be 12.5 cm below from the 'top-most point' in the head.
double circle_fraction = .35; //40. Area threshold. If the contour pixels are atleast "circle_fraction" of the circle, then the sphere fit error will be determined and if it passes both conditions then it will be considered as a head.

int min_radius_threshold = 40; //heads having smaller than this radius are ignored
int max_radius_threshold = 600; //heads having bigger than this radius are ignored
int min_area_threshold = 503; //heads having smaller area than this are ignored

int roi_radius = 8; //8 find local minima in a 16x16 square ROI


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////Shoulder Detection///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
int head_shoulder_thresh = 127; // if the difference in the depth of two contours is greater than this, remove the higher one
int head_height = 150; // It is a bit more than half-head (because we need to consider chin). It is used along with shoulder_height to determine maximum shoulder depth
int shoulder_height = 250; // It is the shoulder drop in depth from the mid-head
int shoulder_bin_size = 25; //Each bin 35 possible depth values. If we reduce the bin size then the array will accumalate depths in a smaller range.
int ellipse_fit_measure = 350; //Determines the pixel-wise closeness of our detected contour to an ellipse. Error must be as low as possible for a perfect fit to an ellipse, thus if it is less than this value we will consider it as a shoulder. If we increase the value, partial ellipses will be considered as shoulders.
int min_bin_vote = 500; //If the number of pixels in the shoulder ROI are greater than this value then it is a shoulder (with further checking).
float shoulder_radius_multiplier = 2.5; //The roi square length is 2.5*radius


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////Throat Detection///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
int throat_drop_limit = 6 * 2.54 * 10.0f; // (in mm).  All depths that are greater than this from the 'top most' point in the head will be = 0.
float throat_probability = 70; //70 if the probability of pixels in the ROI is greater than this, then the contour has a throat drop
int radius_extension = 5; // from the original radius of the head, we increment this much to detect a throat drop


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////ROI/////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

// Accuracy computation for ROI within the rectangle defined by the following two points
cv::Point top_left(55,50);
cv::Point bottom_right(450,390); // (x1,y1) - left top vertex, (x2,y2) - right bottom vertex
bool compute_precision_within_ROI = true; // Enables computation of precision and recall within the defined ROI


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////Tracking////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

int successive_head_difference = 100;//8 + 8; //it was 50*50..for a single person, this is the maximum distance between his heads in two consecutive frames...if the distance is larger than this, then we will count two people
int remove_threshold = 4; //if a head is missing for this number of frame, the person is gone
int add_threshold = 1; //if a head is present for this number of frame, the person is added to the scene
int threshold_for_find_contours = 240; //findContour() function requires binary image

int prev_state_count = 1; //if a person is left side of the door for 2 frames, then right side of the door for 2 frames -> he entered
int next_state_count = 1; //it was 3 before

///////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////Door Detection////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

int door_x1 = 0; //190
int door_y1 = 310; //310

int door_x2 = 520; //370 minimal
int door_y2 = 310; //310

cv::Point door1(door_x1, door_y1);
cv::Point door2(door_x2, door_y2);

///////////////////////////////////////////////////////////////////////////////////////
/////////////Precision and Recall computation for head, shoulder detection/////////////
//////////////////////////////////////////////////////////////////////////////////////


bool writeCSVfile = false; // "True" if we want to write the detected heads in a CSV
bool findAccuracy = false; // "True" if accuracy needs to be calculated

std::ifstream grndTruth;
std::ofstream allError;
std::ofstream noHead;
std::ofstream head;
std::string temp_name;


int truePositive[4] = {0}; // It is a head and labeled as a head
int trueNegative[4] = {0}; // It is not a head and it is labeled as "no head"
int falsePositive[4] = {0}; // It is not a head but labeled as a head
int falseNegative[4] = {0}; // It is a head but labeled as "no head"
int missedHeads[4] = {0};
int totalHeadCount = 0; // Total number of heads detected between the frames
int totalTrueHeads = 0; // Total number of heads in the ground truth between the frames

///////////////////////////////////////////////////////////////////////////////////////
/////////////Occupancy Estimation Evaluation using Ground Truth ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////

//parameters for occupancy estimation based on a ground truth file
bool generate_ground_truth_of_occupancy_estimation = false;   //note that ground truth depends on door: we use (x1,y1) - (x2,y2) = (0, 310) - (520, 310)
bool estimate_accuracy_of_occupancy_estimation = false;
#define TOTAL_FRAMES 9014

int in[TOTAL_FRAMES] = {0}; //number of people got inside for each frame
int out[TOTAL_FRAMES] = {0}; //number of people got outside for each frame

int true_entrance_detected = 0;
int false_entrance_detected = 0;
int true_leave_detected = 0;
int false_leave_detected = 0;
