/*
 @date March 15, 2015, @author Sirajum Munir
 @date May 14, 2015, @author Ripudaman Singh Arora
 @date March 13, 2017, @author Jonathan Appiagyei
 */

// TODO: add instruction to set door line

#include "fork_rs.hpp"

using namespace cv;
using namespace std;
using namespace fork_logger;

// for unprocessed frames
pthread_mutex_t frame_lock;
pthread_cond_t frame_cond;

// for displayed processed frames
pthread_mutex_t im_lock;
pthread_cond_t im_cond;

// used when showing rgb contours
pthread_mutex_t poly_lock;
pthread_cond_t poly_cond;

pthread_t threads[NUM_THREADS];
int thread_ids[NUM_THREADS];

// queue for raw frames
queue<tuple<Mat, string>> *people_image_queue = new queue<tuple<Mat, string>>;
// raw frames filtered by max and min heights
queue<Mat> *filtered_image_queue = new queue<Mat>;
// frames used to display occupancy
queue<Mat> *im_queue = new queue<Mat>;
// frames with rgb contours
queue<Mat> *poly_queue = new queue<Mat>;
queue<vector<struct person_info>> *past_people_queue = new queue<vector<struct person_info>>;
queue<int> past_people_count;

rs::device *dev; // depth sensor
ForkLogger *logger;
int occupancy;
int frame_height;
int frame_width;
int image_no;

bool stop_frame_capture;
bool stop_logger = 0; 

struct person_info{
  Point center[PAST_HORIZON_SIZE];
  int radius[PAST_HORIZON_SIZE];
  int height[PAST_HORIZON_SIZE];
  bool seen[PAST_HORIZON_SIZE];
  int isLeft[PAST_HORIZON_SIZE];  //3 states: 0 (not seen), -1 (right), 1 (left)
  int seen_count;
  int current_time_index;
  int dx;		//average distance from the previous center x
  int dy;     //average distance from the previous center y
  int total_samples; //number of samples for dx, dy computation
  bool in_the_scene;
};

// For bipartite matching
struct distance_info{
  float distance;
  short people_id; //people we have seen before
  short head_id; //head we are seeing in this frame
};

vector<person_info> people_info;

int people_inside_count;
int people_image_no;
int done;

void *process_frames(void *);

static void kill_threads() {
  cout << "exiting" << endl;
  
  stop_frame_capture = 1;
  stop_logger = 1;
  dev->stop();
  
  for (auto thread : threads)
    pthread_join(thread, nullptr);
  
  exit(0);
}

void signal_handler(int sig) {
  kill_threads();
}

bool compareByDistance(const distance_info &a, const distance_info &b)
{
  return a.distance < b.distance;
}

void drawDoor(Mat image)
{
  Point pt1(door_x1,door_y1);
  Point pt2(door_x2,door_y2);
  cv::line(image, pt1, pt2, Scalar(0, 0, 255),2, CV_AA);
  //    rectangle(image, top_left, bottom_right, Scalar(0, 0, 255),2,8);
}

void drawImage(Mat image, string window_name)
{
  Mat image2;
  image.convertTo(image2, CV_32F);
  image2 = image2/IMAGE_DIVIDE;
  
  drawDoor(image2);
  
  namedWindow( window_name.c_str(), WINDOW_AUTOSIZE ); // Create a window for display.
  imshow( window_name.c_str(), image2 );                // Show our image inside it.
}

void drawImageNoDivide(Mat image, string window_name)
{
  drawDoor(image);
  namedWindow( window_name.c_str(), WINDOW_AUTOSIZE ); // Create a window for display.
  imshow( window_name.c_str(), image );
}


Mat readImage(char* image_loc, string window_name, bool display_image)
{
  Mat image;
  image = imread(image_loc, CV_LOAD_IMAGE_ANYDEPTH); // Read the file
  
  if (image.empty())                      // Check for invalid input
  {
    cout <<  "Could not open or find the image" << std::endl ;
    return image;
  }
  
  if (display_image)
    drawImage(image, window_name);
  
  return image;
}

Mat convert_to_grayscale_8bit(Mat src)
{
  Mat temp_src,src_gray;
  
  /// Convert the image to grayscale
  src_gray.create(src.size(), CV_32F);
  
  double min, max;
  minMaxLoc(src, &min, &max);
  
  temp_src = src*255.0f/max; //max
  temp_src.convertTo(src_gray, CV_8U);
  
  return src_gray;
}

Mat convert_to_grayscale_32bit(Mat src)
{
  Mat temp_src,src_gray;
  
  /// Convert the image to grayscale
  src_gray.create(src.size(), CV_32F);
  
  double min, max;
  cv::minMaxLoc(src, &min, &max);
  //cout << "max: " << max << endl;
  temp_src = src*255.0f/IMAGE_DIVIDE; //max
  temp_src.convertTo(src_gray, CV_32F);
  
  return src_gray;
}

Mat canny_edge_detector(Mat src, string window_name, bool display, int lower_threshold, int upper_threshold,  int kernel_size, bool return_mask)
{
  src = convert_to_grayscale_8bit(src);
  Mat dest, detected_edges;
  detected_edges = src;
  
  /// Canny detector
  detected_edges.convertTo(detected_edges, CV_8U);
  Canny(detected_edges, detected_edges, lower_threshold, upper_threshold, kernel_size );
  
  if(return_mask)
    return detected_edges;
  
  /// Using Canny's output as a mask, we display our result
  dest.create(src.size(), src.type());
  dest = Scalar::all(0);
  
  src.copyTo(dest, detected_edges);
  return dest;
}

Mat filter_based_on_max_depth(Mat fixed_image)
{
  for (int i = 0; i < fixed_image.rows; i++)
  {
    for(int j = 0; j < fixed_image.cols; j++)
    {
      if( fixed_image.at<ushort>(i, j) < (max_height_limit - FT_TO_MM(max_people_height)))
        fixed_image.at<ushort>(i, j) = (short) max_height_limit;
      else if (fixed_image.at<ushort>(i, j) > max_height_limit - FT_TO_MM(min_people_height))
        fixed_image.at<ushort>(i, j) = (short)max_height_limit;
    }
  }
  return fixed_image;
}


Mat filter_based_on_max_depth2(Mat image)
{
  Mat fixed_image;
  image.convertTo(fixed_image, CV_16U);
  
  double min, max;
  minMaxLoc(image, &min, &max);
  //cout << "max: " << max << endl;
  
  //cout << "Filter based on max depth type: " << image.type() << " " << CV_16U << endl;
  assert(image.type() == CV_16U);
  
  int height_limit = (int) max_height_limit - 2*12*2.54*10.0f; //2 feet less
  
  //cout << "height limits..." << max_height_limit << " " << (short) max_height_limit << endl;
  for (int i = 0; i < fixed_image.rows; i++)
  {
    for(int j = 0; j < fixed_image.cols; j++)
    {
      if( fixed_image.at<ushort>(i,j) > height_limit || fixed_image.at<ushort>(i,j) < 10)
        fixed_image.at<ushort>(i,j) = (short) max_height_limit;
    }
  }
  return fixed_image;
}

inline int isLeft(Point p0, Point p1, Point p2)
{
  int val = ((p1.x - p0.x)*(p2.y - p0.y) - (p2.x - p0.x)*(p1.y - p0.y));
  if (val >= 0)
    return 1;
  else
    return -1;
}

void update_people_info(vector<Point2f> center, vector<float> radius, vector<float> depth)
{
  int total_people = int(people_info.size()); //number of people based on previous frames
  int total_heads = int(center.size());   //number of heads seen in this frame
  
  //for the ith person, which head is closest?
  
  int center_distance[MAX_PEOPLE][MAX_HEADS];
  int height_distance[MAX_PEOPLE][MAX_HEADS];
  int radius_distance[MAX_PEOPLE][MAX_HEADS];
  float overall_distance[MAX_PEOPLE][MAX_HEADS];
  
  
  int max_center_distance = 0; //for normalization
  int max_height_distance = 0;
  int max_radius_distance = 0;
  
  int center_distance_weight = 100;
  int height_distance_weight = 100;
  int radius_distance_weight = 100;
  
  for( int i = 0; i < total_people; i++) // number of people previously observed
  {
    for(int j = 0; j < total_heads; j++) //number of heads seen in this frame
    {
      
      int current_time_index = people_info[i].current_time_index;
      if (current_time_index == -1)
      {
        center_distance[i][j] = INF;
        height_distance[i][j] = INF;
        radius_distance[i][j] = INF;
        continue;
      }
      
      Point center2 = people_info[i].center[current_time_index];
      
      int x = (int) (center[j].x - center2.x);
      int y = (int) (center[j].y - center2.y);
      int center_difference = sqrt(x*x + y*y) ;
      
      if (center_difference > successive_head_difference) //5000 //100*100 //2*100*100
      {
        center_distance[i][j] = INF;
        height_distance[i][j] = INF;
        radius_distance[i][j] = INF;
        continue;
      }
      
      //            int height_difference = abs(people_info[i].height[current_time_index] - depth[j]);
      
      /*
       if (center_difference > 2*30*30 && height_difference > 30) //if the heads are distant and good height difference
       {
       center_distance[i][j] = INF;
       height_distance[i][j] = INF;
       radius_distance[i][j] = INF;
       continue;
       }
       */
      
      center_distance[i][j] = center_difference;
      if (center_difference > max_center_distance)
        max_center_distance = center_difference;
      
      
      //            height_distance[i][j] = height_difference;
      //            if (height_difference > max_height_distance)
      //                max_height_distance = height_difference;
      
      int radius_difference = abs(people_info[i].radius[current_time_index] - radius[j]);
      radius_distance[i][j] = radius_difference;
      if (radius_difference > max_radius_distance)
        max_radius_distance = radius_difference;
    }
  }
  
  //compute the distances between people and heads, and push it in a vector
  
  vector<struct distance_info> distance_data;
  
  for( int i = 0; i < total_people; i++) // number of people previously observed
  {
    for( int j = 0; j < total_heads; j++) //number of heads seen in this frame
    {
      if (center_distance[i][j] == INF)
      {
        overall_distance[i][j] = INF;
        continue;
      }
      //float center_distance_val = center_distance[i][j]; //center_distance_weight*(center_distance[i][j]/(float)max_center_distance);
      float center_distance_val = center_distance_weight*(center_distance[i][j]/(float)max_center_distance);
      float height_distance_val = height_distance_weight*(height_distance[i][j]/(float)max_height_distance);
      float radius_distance_val = radius_distance_weight*(radius_distance[i][j]/(float)max_radius_distance);
      
      //do more pruning if needed...may be based on distance of head between subsequent frames
      
      struct distance_info current_distance;
      //            current_distance.distance = center_distance_val + height_distance_val + radius_distance_val;
      current_distance.distance = center_distance_val + height_distance_val;
      current_distance.people_id = i;
      current_distance.head_id = j;
      
      distance_data.push_back(current_distance);
    }
  }
  
  if(print_debug_info)
    cout << "Total data points to sort: " << distance_data.size() << endl;
  //sort the vector based on minimum distance
  sort(distance_data.begin(), distance_data.end(), compareByDistance);
  
  //after sorting..
  if(print_debug_info)
  {
    for(int i = 0; i < distance_data.size(); i++)
      cout << "Distance after sorting: " << distance_data[i].distance << " " << distance_data[i].people_id << " " << distance_data[i].head_id << endl;
  }
  //now do the matching...
  
  int total_people_matched = 0;
  int total_heads_matched = 0;
  
  bool people_chosen[MAX_PEOPLE] = {false}; //for people_info
  bool head_chosen[MAX_HEADS] = {false}; //for center (heads in the current frame), whether the head of this frame is already chosen for a person
  
  for(int i = 0; i < distance_data.size(); i++)
  {
    short people_id = distance_data[i].people_id;
    short head_id = distance_data[i].head_id;
    
    if (people_chosen[people_id] || head_chosen[head_id])
      continue;
    
    //match this people with this head
    people_chosen[people_id] = true;
    head_chosen[head_id] = true;
    
    int prev_time_index = people_info[people_id].current_time_index;
    people_info[people_id].current_time_index = (people_info[people_id].current_time_index + 1)%PAST_HORIZON_SIZE;
    int current_time_index = people_info[people_id].current_time_index;
    people_info[people_id].center[current_time_index] = center[head_id];
    people_info[people_id].radius[current_time_index] = radius[head_id];
    //        people_info[people_id].height[current_time_index] = depth[head_id];
    
    people_info[people_id].total_samples++;
    int n = people_info[people_id].total_samples;
    int new_dx = center[head_id].x - people_info[people_id].center[prev_time_index].x;
    int new_dy = center[head_id].y - people_info[people_id].center[prev_time_index].y;
    people_info[people_id].dx = people_info[people_id].dx*(n-1)/n + new_dx/n;  //computing moving average
    people_info[people_id].dy = (int)((1 - .9) * people_info[people_id].dy + .9 * new_dy);  //computing moving average
    
    people_info[people_id].seen[current_time_index] = true;
    people_info[people_id].isLeft[current_time_index] = isLeft(door1, center[head_id], door2);
    people_info[people_id].seen_count++;
    if (people_info[people_id].seen_count >= add_threshold)
    {
      if(people_info[people_id].seen_count >= remove_threshold)
        people_info[people_id].seen_count = remove_threshold;
      people_info[people_id].in_the_scene = true;
      
    }
    
    total_people_matched++;
    total_heads_matched++;
    
    if(total_people_matched == total_people || total_heads_matched == total_heads)
      break;
    
  }
  
  //now we have to handle 2 cases
  
  //case 1: people in the previous frames that didn't have any matched head
  //update people who are not seen in this frame
  for (int i = total_people - 1; i >= 0; i--)
  {
    if (people_chosen[i])
      continue;
    int prev_time_index = people_info[i].current_time_index;
    people_info[i].current_time_index = (people_info[i].current_time_index + 1)%PAST_HORIZON_SIZE;
    int current_time_index = people_info[i].current_time_index;
    people_info[i].center[current_time_index].x = people_info[i].center[prev_time_index].x + people_info[i].dx;  //change?
    people_info[i].center[current_time_index].y = people_info[i].center[prev_time_index].y + people_info[i].dy;  //change?
    people_info[i].radius[current_time_index] = people_info[i].radius[prev_time_index];
    people_info[i].height[current_time_index] = people_info[i].height[prev_time_index];
    people_info[i].seen[current_time_index] = false;
    people_info[i].isLeft[current_time_index] = people_info[i].isLeft[prev_time_index];
    people_info[i].seen_count--;
    if (people_info[i].seen_count <= 0)//remove this person
    {
      people_info.erase(people_info.begin() + i);
    }
  }
  
  
  //case 2: add the heads that were not matched with any previously found people
  for (int i = 0; i < total_heads; i++)
  {
    if (head_chosen[i] == true)
      continue;
    
    if(print_debug_info)
      cout << "Adding a new person...." << endl;
    struct person_info person_info;
    
    person_info.current_time_index = 0;
    person_info.center[0] = center[i];
    person_info.radius[0] = radius[i];
    //        person_info.height[0] = depth[i];
    person_info.seen[0] = true;
    person_info.dx = 0;
    person_info.dy = 0;
    person_info.total_samples = 0;
    memset(person_info.isLeft, 0, sizeof(person_info.isLeft));
    person_info.isLeft[0] = isLeft(door1, center[i], door2);
    person_info.seen_count = 1;
    person_info.in_the_scene = false;
    people_info.push_back(person_info);
  }
  /*
   int total_people_in_the_scene = 0;
   //printing people in the scene
   cout << "Total people in the scene: ";
   for (int i = 0; i < people_info.size(); i++)
   {
   if(people_info[i].in_the_scene)
   total_people_in_the_scene++; //cout << " " << i;
   }
   //cout << endl;
   cout << total_people_in_the_scene << endl;
   */
}


void estimate_occupancy(int frame_no, int offset)
{
  int in_now = 0;
  int out_now = 0;
  
  int total_people = (int)people_info.size();
  
  if (match_by_velocity) {
    for (int i = 0; i < total_people; i++) {
      if (!people_info[i].in_the_scene)
        continue;
      int i_dx = people_info[i].dx;
      int i_dy = people_info[i].dy;
      
      for (int j = 0; j < total_people; j++) {
        if (j == i || !people_info[j].in_the_scene)
          continue;
        
        int i_current_time_index = people_info[i].current_time_index;
        int j_current_time_index = people_info[j].current_time_index;
        
        int j_dx = people_info[j].dx;
        int j_dy = people_info[j].dy;
        int ab_diff = abs(i_dx - j_dx) + abs(i_dy - j_dy);
        int y_ab_diff = abs(i_dy - j_dy);
        //                cout << "y_diff: " << y_ab_diff << endl;
        printf("dy: %d, dy: %d ", i_dy, j_dy);
        
        printf("ab_diff: %d FOUND SAME PEOPLE!!!\n", y_ab_diff);
        if (y_ab_diff < 5 && ((i_dy > 0 && j_dy > 0) || (i_dy < 0 && j_dy < 0))) {
          //                    printf("dy: %d, dy: %d ", i_dy, j_dy);
          //
          //                    printf("ab_diff: %d FOUND SAME PEOPLE!!!\n", y_ab_diff);
          if (people_info[i].center[i_current_time_index].y > people_info[j].center[j_current_time_index].y)
            people_info[i].in_the_scene = false;
          else
            people_info[j].in_the_scene = false;
        }
      }
    }
  }
  
  
  for(int i = 0; i < total_people; i++)
  {
    if (!people_info[i].in_the_scene)
      continue;
    
    int total_state_check_passed = 0;
    //is this person entered into a room?
    //he has to change his state at this frame and the next (2-1) frames..so, start frame is actually one frame behind from current
    //his state needs to be different in the last 2 frames
    
    int current_time_index = people_info[i].current_time_index;
    int start_index = (current_time_index - prev_state_count + PAST_HORIZON_SIZE) % PAST_HORIZON_SIZE;
    
    //if (!people_info[i].seen[start_index])
    //   continue;
    int start_state = people_info[i].isLeft[start_index];
    int next_state = start_state*(-1);
    
    
    //need to deal with in_the_scene
    if(print_debug_info)
    {
      cout << "people " << i << " state: " ;
      
      cout <<"start and next state: " << start_state << " " << next_state << endl;
      cout << "start index:" << start_index << " , current index: " << current_time_index << endl;
      cout << "isLeft: " ;
      for (int k = 0; k < PAST_HORIZON_SIZE; k++)
        cout << " " << people_info[i].isLeft[k];
      cout << endl;
    }
    
    int j = start_index;
    
    for(int k = 0; k < prev_state_count; k++, j = (j+1)%PAST_HORIZON_SIZE)
    {
      //if (!people_info[i].seen[j])
      //    continue;
      if(print_debug_info)
        cout << " " << people_info[i].isLeft[j];
      if (people_info[i].isLeft[j] == start_state)
        total_state_check_passed++;
    }
    
    for(int k = 0; k < next_state_count; k++, j = (j+1)%PAST_HORIZON_SIZE)
    {
      //if (!people_info[i].seen[j])
      //    continue;
      if(print_debug_info)
        cout << " " << people_info[i].isLeft[j];
      if (people_info[i].isLeft[j] == next_state)
        total_state_check_passed++;
    }
    if(print_debug_info)
    {
      cout << endl;
      cout << ">>>Person: " << i << " , total check passed: " << total_state_check_passed << endl;
    }
    
    if (total_state_check_passed == (prev_state_count + next_state_count))
    {
      if(start_state == 1 && next_state == -1) //he entered into the room
      {
        if(!count_opposite_way)
          occupancy++, in_now++;
        else
          occupancy--, out_now++;
        
      }
      if(start_state == -1 && next_state == 1) //he left the room
      {
        if(!count_opposite_way)
          occupancy--, out_now++;
        else
          occupancy++, in_now++;
      }
    }
  }
  
  if(generate_ground_truth_of_occupancy_estimation)
  {
    in[frame_no] = in_now; //in[frame_no - next_state_count] = in_now;
    out[frame_no] = out_now; //out[frame_no - next_state_count] = out_now;
  }
  
  if(estimate_accuracy_of_occupancy_estimation)
  {
    if(in_now > 0 || in[frame_no] > 0) // we detect some people entering in this frame
    {
      true_entrance_detected +=  min(in_now, in[frame_no]);
      if (in_now > in[frame_no]){
        false_entrance_detected += in_now - in[frame_no];
        cout << "False entrance detected: " << in_now - in[frame_no] << endl;
        waitKey(0);
      }
      if( in[frame_no] > in_now)
      {
        cout << "Missed people in entrance: " << in[frame_no] - in_now << endl;
        waitKey(0);
        
      }
      
    }
    if(out_now > 0 || out[frame_no] > 0) // we detect some people leaving out this frame
    {
      true_leave_detected +=  min(out_now, out[frame_no]);
      if ( out_now > out[frame_no])
      {
        false_leave_detected += out_now - out[frame_no];
        cout << "False leaving detected: " << out_now - out[frame_no];
        waitKey(0);
      }
      if (out[frame_no] > out_now)
      {
        cout << "Missed people leaving: " << out[frame_no] - out_now << endl;
        waitKey(0);
      }
      
    }
    
  }
}


void find_minimum_enclosing_circle_no_canny(Mat src, vector<Point2f> &center, vector<float> &radius, vector<vector<Point> > &contours_poly,double threshold_val) // Find the 2D heads by using findContours.
{
  // Step 1: Threshold the input image with the current depth level provided.(6...5.5...5...4.5...)
  Mat threshold_output;
  int height_diff = max_height_limit - threshold_val;
  //    convert_to_grayscale_8bit(src);
  src.convertTo(src, CV_16S);
  threshold(src, threshold_output, height_diff, 1, THRESH_TOZERO_INV);
  if(!suppress_images){
    Mat threshold_output2 = threshold_output * 100;
    drawImageNoDivide(threshold_output2, "Output: One Thresh Image");
  }
  vector<Vec4i> hierarchy;
  
  // Step 2: Find contours in the thresholded image.
  vector<vector<Point>> contours;
  threshold_output.convertTo(threshold_output, CV_8U);
  findContours(threshold_output, contours, RETR_LIST, CV_CHAIN_APPROX_NONE);//CV_RETR_TREE
  if (display_polys) {
    RNG rng(12345);
    Mat drawing = Mat::zeros(src.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
    {
      Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
      drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }
    poly_queue->push(drawing);
  }
  
  center.resize(contours.size());
  radius.resize(contours.size());
  contours_poly.resize(contours.size());
  
  // Step 3: Find the center and radius of the contours.
  for( int i = 0; i < contours.size(); i++ )
  {
    approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
    minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
  }
  
}

void find_minimum_enclosing_circle(Mat src, vector<Point2f> &center, vector<float> &radius, vector<vector<Point> > &contours_poly, int frame)
{
  Mat src_gray;
  
  //cout << src << endl;
  src.convertTo(src_gray, CV_8U);
  
  //drawImageNoDivide(src_gray, "Image before getting contours");
  
  /// Detect edges using Threshold
  //    Mat threshold_output;
  //    threshold( src_gray, threshold_output, threshold_for_find_contours, 255, THRESH_BINARY_INV );
  //
  //drawImageNoDivide(threshold_output, "Image before getting contours after thresholding");
  
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  
  // findContours( src_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );//CV_RETR_TREE
  findContours(src_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );//CV_RETR_TREE
  //vector<Rect> boundRect( contours.size() );
  RNG rng(12345);
  Mat drawing = Mat::zeros(src.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
  {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
  }
  //    imshow( "image with contours", drawing );
  poly_queue->push(drawing);
  center.resize(contours.size());
  radius.resize(contours.size());
  contours_poly.resize(contours.size());
  
  //    cout << "Total contours: " << contours.size() << endl;
  
  for( int i = 0; i < contours.size(); i++ )
  {
    approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
    //boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
    //cout << "circle: " << i + 1 << " center: " << center[i].x << " " << center[i].y << " " << " radius: " << radius[i] << endl;
  }
  
  if(print_debug_info)
    cout << "Canny OP type : " << src_gray.type() << endl;
}

bool check_center_in_ROI(int centx,int centy)
{
  bool is_in_ROI = true;
  
  if (centx < top_left.x || centy < top_left.y || centx > bottom_right.x || centy > bottom_right.y)
    is_in_ROI = false;
  
  return is_in_ROI;
  
}

// EFFECT: Function to write a CSV file for the calculated error.
// REQUIRES: writeCSVfile boolean to be enabled.
void writeCSV(int fileType, int frameno, double centx, double centy,double rad,int ishead)
{
  if(!writeCSVfile)
    return;
  
  if (fileType == 0)
    head << frameno << "," << int(centx) << "," << int(centy) << "," << int(rad) << "," << ishead << endl;
  
  else if (fileType == 1)
    noHead << frameno << "," << int(centx) << "," << int(centy) << "," << int(rad) << "," << ishead <<endl;
  
  else if (fileType == 2)
    allError << frameno << "," << int(centx) << "," << int(centy) << "," << int(rad) << "," << ishead << endl;
  
}

Mat add_circles(Mat src, vector<Point2f> &center, vector<float> &radius, vector<vector<Point> > &contours_poly , int frame)
{
  Mat src_gray = convert_to_grayscale_8bit(src);
  
  Mat drawing;
  src_gray.convertTo(drawing, CV_8UC3);
  
  for( int i = 0; i< contours_poly.size(); i++ )
  {
    
    double area = contourArea(contours_poly[i]);
    
    if(print_debug_info)
      cout << "radius:" << radius[i] << ", area: " << area << endl;
    
    
    Scalar color = Scalar( 0,0,0 );
    drawContours(drawing, contours_poly, i, color, 2, 8, vector<Vec4i>(), 0, Point() );
    circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    circle( drawing, center[i], 4, color, 2, 8, 0 ); //highlight the center
  }
  
  drawDoor(drawing);
  return drawing;
}

void drawTrackingLines(Mat image, vector<struct person_info> past_people)
{
  int total_people = int(past_people.size()); //number of people based on previous frames
  
  for(int i = 0; i < total_people; i++)
  {
    int current_time_index = past_people[i].current_time_index;
    for (int j = 0; j < current_time_index; j++)
    {
      cv::line(image, past_people[i].center[j], past_people[i].center[j+1], Scalar(128, 128, 128),2,CV_AA);
      // out of index possibbly j + 1
    }
  }
}

void display_occupancy_info(Mat img, vector<struct person_info> past_people, int past_count)
{
  //char buf[20];
  //itoa(total_people_in_the_room, buf, 10);
  string text = "People Inside: " + to_string(past_count);
  string head_text = "Head: " + to_string(head_counter);
  string shoulder_text = "Shoulder: " + to_string(shoulder_counter);
  
  int fontFace = CV_FONT_HERSHEY_TRIPLEX; //ONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 1;
  int thickness = 1;
  
  int baseline=0;
  //Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
  baseline += thickness;
  
  // center the text
  //Point textOrg((img.cols - textSize.width)/2, (img.rows + textSize.height)/2);
  Point textOrg(110,25);
  Point textOrg1(25,60);
  Point textOrg2(220,60);
  
  // draw the box
  //rectangle(img, textOrg + Point(0, baseline), textOrg + Point(textSize.width, -textSize.height), Scalar(0,0,255));
  
  // ... and the baseline first
  //line(img, textOrg + Point(0, thickness), textOrg + Point(textSize.width, thickness), Scalar(0, 0, 255));
  
  // then put the text itself
  //putText(img, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
  if(print_debug_info)
  {
    putText(img, text, textOrg, fontFace, fontScale, Scalar::all(0), thickness, 8);
    putText(img, head_text, textOrg1, fontFace, fontScale, Scalar::all(0), thickness, 2);
    putText(img, shoulder_text, textOrg2, fontFace, fontScale, Scalar::all(0), thickness, 2);
  }
  else
  {
    textOrg.x = 110;
    textOrg.y = 65;
    putText(img, text, textOrg, fontFace, fontScale, Scalar::all(0), thickness, 8);
    
  }
  img = convert_to_grayscale_8bit(img);
  drawTrackingLines(img, past_people);
  namedWindow( "Occupancy Estimation", CV_WINDOW_AUTOSIZE );
  imshow( "Occupancy Estimation", img );
}

Mat convert_to_rgb(Mat people_image)
{
  Mat person_image;
  people_image.convertTo(person_image, CV_8UC1, 255.0/IMAGE_DIVIDE);
  drawDoor(person_image);
  
  Mat out;
  cv::Mat in[] = {person_image, person_image, person_image};
  merge(in, 3, out);
  
  //cout << "type: " << person_image.type() << " " << out.type() <<  " " << occ_image.type() << endl;
  return out;
}

void compute_frequency_distribution(Mat image, int freq[])
{
  float height_limits[50];
  
  height_limits[0] = INF_HEIGHT;
  for(int i = 0; i < total_diff_heights; i++)
  {
    freq[i] = 0;
    float height_val = (min_people_height + height_decrement*i)*12*2.54*10;
    height_limits[i+1] = max_height_limit - height_val; //height_limits is decreasing for higher indices
  }
  
  /*
   cout << "Different heights: ";
   for (int i = 0; i < total_diff_heights; i++)
   cout << " " << height_limits[i]/(12*2.54*10);
   cout << endl;
   */
  
  //cout << "In freq distribution, image type: " << image.type() << " " << CV_16U << " " << CV_32F << " " << endl;
  //    assert(image.type() == CV_16S);
  
  for(int i = 0; i < image.rows; i++)
  {
    for(int j = 0; j < image.cols; j++)
    {
      int val = (int)image.at<ushort>(i,j);
      for(int k = 0; k < total_diff_heights; k++)
      {
        
        if(val >= height_limits[k+1] && val < height_limits[k])
          freq[k]++;
      }
    }
  }
}

Mat filter_based_on_people_height(Mat input_image, double min_height)
{
  Mat fixed_image = input_image.clone();
  
  int height_diff = max_height_limit - min_height;
  
  if(!discard_height_data_above)
  {
    for (int i = 0; i < fixed_image.rows; i++)
    {
      for(int j = 0; j < fixed_image.cols; j++)
      {
        if( fixed_image.at<short>(i,j) > height_diff )
        {
          fixed_image.at<short>(i,j) = (short)max_height_limit;
          //cout << i << " " << j << " " << fixed_image.at<short>(i,j) << " " << max_height_limit << " " << height_diff <<  endl;
        }
      }
    }
  }
  else
  {
    int height_diff2 = max_height_limit - min_height - height_prune_above_threshold;
    
    for (int i = 0; i < fixed_image.rows; i++)
      for(int j = 0; j < fixed_image.cols; j++)
      {
        if( fixed_image.at<ushort>(i,j) > height_diff || fixed_image.at<int>(i,j) < height_diff2)
          fixed_image.at<ushort>(i,j) = max_height_limit;
      }
  }
  return fixed_image;
}

void find_centers_of_current_height(Mat height_fixed_image, vector<Point2f> &current_center, vector<float> &current_radius, vector<vector<Point> > &current_contours_poly, int frame)
{
  if (!suppress_images)
    drawImage(height_fixed_image, "Height Fixed Difference");
  //    cvWaitKey(150);
  //    Mat image = canny_edge_detector(height_fixed_image, "Canny Edge Detector of Filtered Image",true, 200, 255 * 2, 5, false);
  find_minimum_enclosing_circle_no_canny(height_fixed_image, current_center, current_radius, current_contours_poly, FT_TO_MM(min_people_height));
  //    find_minimum_enclosing_circle(image, current_center, current_radius, current_contours_poly, frame);
  
}

// calculate the average depth of whatever is bounded by the polygon contour
double calculate_shoulder_average(Mat &src, vector<Point> contours) {
  double average = 0;
  int points_added = 0;
  
  int x_min = INT_MAX, y_min = INT_MAX;
  int x_max = INT_MIN, y_max = INT_MIN;
  
  for (int i = 0; i < contours.size(); ++i) {
    if (contours[i].y < x_min)
      x_min = contours[i].y;
    
    if (contours[i].y > x_max)
      x_max = contours[i].y;
    
    if (contours[i].x < y_min)
      y_min = contours[i].x;
    
    if (contours[i].x > y_max)
      y_max = contours[i].x;
  }
  
  for (int x = x_min; x <= x_max; x++) {
    for (int y = y_min; y <= y_max; y++) {
      Point2f test_point(y, x);
      int test_result = pointPolygonTest(contours, test_point, false);
      
      if (test_result == 1) {
        ushort test_point_depth = src.at<ushort>(x, y);
        
        if (test_point_depth == 0 || (test_point_depth > (FT_TO_MM(9) - FT_TO_MM(min_people_height))) || (test_point_depth < (FT_TO_MM(9) - FT_TO_MM(max_people_height)))) {
          //                    cout << "not good: " << test_point_depth << endl;
          continue;
        }
        points_added++;
        average += test_point_depth;
      }
    }
  }
  return average / points_added;
}

bool update_centers(Mat src, vector<Point2f> &current_center, vector<float> &current_radius, vector<vector<Point> > &current_contours_poly, vector<Point2f> &center, vector<float> &radius, vector<float> &depth, vector<vector<Point> > &contours_poly, bool enable_update, int frame)
{
  vector <int> samples;
  samples.resize(center.size(), 0);
  if (frame <= number_of_frames_for_background_determination)
    return false;
  bool is_updated = false;
  
  for (int j = 0; j < current_center.size(); j++)
  {
    if (current_radius[j] >= min_radius_threshold && current_radius[j] <= max_radius_threshold)
    {
      int total_centers = (int)center.size();
      
      int k = 0;
      for(; k < total_centers; k++)
      {
        double xx = center[k].x - current_center[j].x;
        double yy = center[k].y - current_center[j].y;
        double dist = sqrt((xx * xx) + (yy * yy));
        
        if (dist < max_center_diff) {// bigger counter always gets a distance big is 20, small is 14
          double curr_poly_depth = calculate_shoulder_average(src, current_contours_poly[j]); // bigger counter a go
          double old_poly_depth = calculate_shoulder_average(src, contours_poly[k]); //smaller counter no go
          
          if (abs(curr_poly_depth - old_poly_depth) < same_shoulder_thresh)
          {
            
            center[k].x = (center[k].x + current_center[j].x) / 2;
            center[k].y = (center[k].y + current_center[j].y) / 2;
            
            cout << "TADAAAAA!!" << endl;
            break;
          }
          if (curr_poly_depth < old_poly_depth) {
            center[k].x = (center[k].x + current_center[j].x) / 2;
            center[k].y = (center[k].y + current_center[j].y) / 2;
            break;
          }
          else {
            if (samples[k]) {
              radius[k] = (radius[k] + current_radius[j]) / 2;
              center[k].x = (center[k].x + current_center[j].x) / 2;
              center[k].y = (center[k].y + current_center[j].y) / 2;
              samples[k]++;
            }
            else {
              radius[k] = current_radius[j];
              center[k] = current_center[j];
              contours_poly[k] = current_contours_poly[j];
              depth[k] = curr_poly_depth;
            }
            
            break;
          }
        }
      }
      if (k == total_centers) // all the previous centers are away from max_center_diff
      {
        is_updated = true;
        
        double min_pixel = -1;
        
        if (enable_update)
        {
          center.push_back(current_center[j]);
          radius.push_back(current_radius[j]);
          depth.push_back(calculate_shoulder_average(src, current_contours_poly[j]));
          samples.resize(center.size(), 0);
          
          if(print_debug_info)
            cout << "...Depth value: " << min_pixel << endl;
          contours_poly.push_back(current_contours_poly[j]);
        }
      }
    }
  }
  if (center.size())
    store_frame = true;
  return is_updated;
}

void find_centers_using_different_heights(Mat height_fixed_image, int background_freq[], vector<Point2f> &center, vector<float> &radius, vector<float> &depth, vector<vector<Point> > &contours_poly, int frame, int frame_no, Mat people_image)
{
  //we are iterating from max height to min height..in this way, if we find a circle at the top, we will discard any other circles that are very close to that head at a smaller height
  int current_freq[HEIGHT_ARRAY_SIZE];
  int freq_after_background_subtraction[HEIGHT_ARRAY_SIZE];
  
  if (!background_image_determined) {
    compute_frequency_distribution(height_fixed_image, current_freq);
    
    for (int j = 0; j < total_diff_heights; j++)
    {
      background_freq[j] = min(background_freq[j], current_freq[j]);
      freq_after_background_subtraction[j] = current_freq[j] - background_freq[j];
    }
  }
  
  if(perform_background_subtraction && !background_image_determined)
  {
    if(frame_no > number_of_frames_for_background_determination)
    {
      int sum_of_background_diff = 0;
      for (int j = 1; j < total_diff_heights; j++)  //j = 0 contains lots of noises..
      {
        sum_of_background_diff += freq_after_background_subtraction[j];
      }
      if(print_debug_info)
        cout << "---------->Background diff: " << sum_of_background_diff << endl;
      
      if (sum_of_background_diff <= background_determination_threshold)
      {
        background_image_determined = true;
        background_image = height_fixed_image.clone();
      }
    }
  }
  if(print_debug_info)
  {
    cout << "----------FREQ DISTRIBUTION------------: ";
    for (int j = 0; j < total_diff_heights; j++)
      cout << freq_after_background_subtraction[j] << " " ;
    cout << endl;
  }
  
  vector<Point2f> current_center;
  vector<float> current_radius;
  vector<vector<Point> > current_contours_poly;
  
  find_centers_of_current_height(height_fixed_image, current_center, current_radius, current_contours_poly, frame_no);
  
  if (current_radius.size() > 0)
    update_centers(people_image, current_center, current_radius, current_contours_poly, center, radius, depth, contours_poly, true, frame);
}

// Used to jump to the required start frame in the ground truth file if we do not start frames from 1.
int jump_To_Start_Frame(string w)
{
  // Remove the first line that contains all the string info
  string word = w;
  getline(grndTruth, word, ','); // cx
  getline(grndTruth, word, ','); // cy
  getline(grndTruth, word, ','); // rad
  getline(grndTruth, word, ','); // err
  getline(grndTruth, word, ','); // total pix
  getline(grndTruth, word,'\r'); // is head
  getline(grndTruth, word, ','); // frame no
  
  int fn = stoi(word);
  
  while (fn < startFrame)
  {
    getline(grndTruth, word, ','); // cx
    getline(grndTruth, word, ','); // cy
    getline(grndTruth, word, ','); // rad
    getline(grndTruth, word, ','); // err
    getline(grndTruth, word, ','); // total pix
    getline(grndTruth, word,'\r'); // is head
    getline(grndTruth, word, ','); // frame no
    fn = stoi(word);
  }
  return fn;
}

// Calculates and prints the parameters that determine the efficiency of the algorithm
void write_Accuracy(int check_no)
{
  if(!findAccuracy)
    return;
  
  int total_head_count;
  total_head_count = truePositive[check_no] + trueNegative[check_no] + falseNegative[check_no] + falsePositive[check_no];
  
  switch (check_no)
  {
    case 0:
      if(!throat_check)
        return;
      cout << "----- Throat Drop readings ---- " << endl << endl;
      break;
      
    case 1:
      if(!head_check)
        return;
      cout << "----- Sphere fit readings -----" << endl << endl;
      break;
      
    case 2:
      if(!shoulder_check)
        return;
      cout << "----- Shoulder Drop readings -----" << endl << endl;
      break;
      
    case 3:
      cout << "----- Total readings -----" << endl << endl;
      break;
  }
  cout << "True Positive : " << truePositive[check_no] << endl;
  cout << "False Positive : " << falsePositive[check_no] << endl;
  cout << "True Negative : " << trueNegative[check_no] << endl;
  cout << "False Negative : " << falseNegative[check_no] << endl;
  cout << "Total Head Count : " << totalHeadCount << ", " << total_head_count << endl;
  cout << "True Head Count : " << totalTrueHeads  << endl;
  
  double pre = (float(truePositive[check_no]) / (truePositive[check_no] + falsePositive[check_no])) * 100;
  cout << endl <<  "Precision : " << pre << endl;
  double rec = (float(truePositive[check_no]) / (truePositive[check_no] + falseNegative[check_no])) * 100;
  cout << "Recall : " << rec << endl;
  double acc = (float(truePositive[check_no] + trueNegative[check_no]) / (total_head_count)) * 100;
  cout << "Accuracy : " << acc << endl;
  double F = (float(2 * pre * rec) / (pre + rec));
  cout << "F score : " << F << endl << endl;
}

int read_CSV_and_first_element()
{
  if(!findAccuracy)
    return 0;
  
  grndTruth.open(groundTruthFile);
  getline(grndTruth, temp_name, ',');
  return jump_To_Start_Frame(temp_name);
  
}

void init_CSV()
{
  
  if(!writeCSVfile)
    return;
  
  string no_head_file = "/Users/Apple/desktop/Bosch/non_heads_rad1.csv";
  string head_file = "/Users/Apple/desktop/Bosch/heads_rad1.csv";
  string only_error = "/Users/Apple/desktop/Bosch/all_error1.csv";
  
  
  allError.open(only_error);
  noHead.open(no_head_file);
  head.open(head_file);
  
  // Writing CSV
  
  allError << "Frame" << "," << "Center.x" << "," << "Center.y" << ","  << "Radius" << "," << "Error" << "," << "Total Pixels" << "," << "Is Head" << endl;
  
  head << "Frame" << "," << "Center.x" << "," << "Center.y" << ","  << "Radius" << "," << "Error" << "," << "Total Pixels" << "," << "Is Head" << endl;
  
  noHead << "Frame" << "," << "Center.x" << "," << "Center.y" << ","  << "Radius" << "," << "Error" << "," << "Total Pixels" << "," << "Is Head" << endl;
  
}

void write_video_in_a_file(VideoWriter output, Mat people_image, Mat video_mat)
{
  
  Mat people_image_rgb = convert_to_rgb(people_image);
  //    Mat background_image_rgb = convert_to_rgb(diff_image_backup);
  
  Mat temp;
  //    hconcat(people_image_rgb, background_image_rgb, temp);
  //    hconcat(temp, image_with_cirlces, video_mat);
  //    hconcat(temp, background_image_rgb, video_mat);
  
  namedWindow( "concatenated image", WINDOW_AUTOSIZE ); // Create a window for display.
  
  imshow( "concatenated image", video_mat );                // Show our image inside it.
  
  //cout << "types: " << people_image_rgb.type() << " " << background_image_rgb.type() <<  " " << temp.type() << " " << occ_image.type() << " " << video_mat.type() << endl;
  output.write(video_mat); //image3
  
}

Mat hough_lines(Mat image2, Mat image_original)
{
  bool use_hough_line_p = false; //determines whether to use HoughLines or HoughLinesP
  Mat image;
  image2.convertTo(image, CV_8U);
  
  //drawImageNoDivide(image, "Trying hough lines");
  
  //Mat dest = image_original.clone();
  Mat dest = image.clone();
  
  //dest.create(image2.size(), CV_8U);
  //dest = Scalar::all(255);
  
  /// Using Canny's output as a mask, we display our result
  //dest.create(src.size(), CV_32F);
  //dest = Scalar::all(max_height_limit);
  
  if( !use_hough_line_p)
  {
    vector<Vec2f> lines;
    //HoughLines(image, lines, 1, CV_PI/180, 1000, 0, 0 );
    HoughLines(image, lines, 5, CV_PI/180, 125, 0, 0 );
    
    if(print_debug_info)
      cout << "....Total lines: " << lines.size() << endl;
    
    for( size_t i = 0; i < lines.size(); i++ ) //lines.size()
    {
      float rho = lines[i][0], theta = lines[i][1];
      Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      pt1.x = cvRound(x0 + 1000*(-b));
      pt1.y = cvRound(y0 + 1000*(a));
      pt2.x = cvRound(x0 - 1000*(-b));
      pt2.y = cvRound(y0 - 1000*(a));
      //line( dest, pt1, pt2, Scalar(255,0,255), 3, CV_AA);
      
      float slope_y = pt2.y - pt1.y;
      float slope_x = pt2.x - pt1.x;
      
      if(slope_x == 0)
        continue;
      float slope = slope_y/slope_x;
      if(print_debug_info)
        cout << "slope: " << slope << endl;
      float threshold = 0.087; //tan(5 degree)
      if(slope > -threshold && slope < threshold){
        line(dest, pt1, pt2, Scalar(100, 0, 0),2,CV_AA);
        
        //update the door
        door_x1 = pt1.x; //190
        door_y1 = pt1.y;
        
        door_x2 = pt2.x; //370 minimal
        door_y2 = pt2.y;
        
        
        door1.x = door_x1;
        door1.y = door_y1;
        door2.x = door_x2;
        door2.y = door_y2;
        
        cout << "New door: " << door_x1 << " " << door_y1 << " " << door_x2 << " " << door_y2 << endl;
        
        break;
      }
      
    }
  }
  else
  {
    vector<Vec4i> lines;
    //HoughLinesP(image, lines, 1, CV_PI/180, 50, 50, 10 );
    HoughLinesP(image, lines, 5, CV_PI/180, 20, 25, 5 );
    if(print_debug_info)
      cout << "....Total lines: " << lines.size() << endl;
    
    for( size_t i = 0; i < lines.size(); i++ ) //lines.size()
    {
      Vec4i l = lines[i];
      //line( dest, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,255), 1, CV_AA);
      line(dest, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(100, 0,0),2,8);
      if(print_debug_info)
        cout << "point: " << i << ": " << l[0] << " " << l[1] << " " << l[2] << " " << l[3] << endl;
      
    }
  }
  return dest;
  
}
void detect_door(Mat image)
{
  Mat im2 = image.clone();
  Mat im = filter_based_on_max_depth2(im2);
  
  //drawImage(im2, "Door detection input");
  Mat canny_image_filtered = canny_edge_detector(im, "Canny Edge Detector of Filtered Image",true, 1000, 2000, 5, true);
  //drawImageNoDivide(canny_image_filtered, "Door detection Canny");
  Mat hough_line_image = hough_lines(canny_image_filtered, image);
  //drawImageNoDivide(hough_line_image, "Door detection Canny + hough");
  
}

float compute_floor_depth(Mat image)
{
  float bin_size = 0.5*12*2.54*10.0f; //0.5 feet = 152.4 mm
  
  double min, max;
  cv::minMaxLoc(image, &min, &max);
  
  if(max > MAX_DEPTH)
    max = MAX_DEPTH;
  int total_bins = ceil(max/bin_size);
  
  if(print_debug_info)
  {
    cout << "image type: " << image.type() << " " << CV_16U << endl;
    cout << "short size: " << sizeof(unsigned short) << endl;
  }
  
  assert(image.type() == CV_16U);
  
  int bins[500] = {0}; //assuming the number of bins does not exceed 500
  for(int i = 0; i < image.rows; i++)
  {
    for(int j = 0; j < image.cols; j++)
    {
      int val = (int)image.at< uint16_t >(i,j); //at<float>? at<unsigned short>, or at<ushort>
      if (val > MAX_DEPTH)
        continue;
      else{
        int index = (int)floor(val/bin_size);
        //cout << val << " ";
        bins[index]++;
      }
    }
    //cout << endl;
  }
  
  int max_val = -1;
  int max_index = -1;
  if(print_debug_info)
    cout << "Floor depths: ";
  for(int i = 1; i < total_bins; i++) //ignoring i = 0, which may contain a lot of 0s and not contain the floor
  {
    if(print_debug_info)
      cout << " " << bins[i];
    if(bins[i] > max_val)
    {
      max_val = bins[i];
      max_index = i;
    }
  }
  if(print_debug_info)
    cout << endl;
  
  if(max_index == -1)
    return -1;
  else
    return (max_index+1)*bin_size;
  
}

void write_ground_truth_of_occupancy_estimation()
{
  ofstream file_gt;
  
  file_gt.open(occGroundTruthFile);
  
  file_gt << "FrameNo, In, Out\n";
  
  for(int i = startFrame; i <= endFrame; i++ )
    file_gt << i << "," << in[i] << "," << out[i] << "\n";
  
  file_gt.close();
  
}
void  read_ground_truth_of_occupancy_estimation()
{
  ifstream file_in;
  file_in.open(occGroundTruthFile);
  
  if(!file_in.is_open())
  {
    cout << "Can not open the occupancy estimation ground truth file." << endl;
    return;
    
  }
  //char temp[200];
  //file_in >> temp; //header of the ground truth file
  
  string line;
  getline(file_in, line);
  
  int frame_no, in_no, out_no;
  char c;
  
  int total_in = 0, total_out = 0;
  while(file_in >> frame_no >> c >> in_no >> c >> out_no) // c is for reading commas
  {
    //cout << frame_no << " " << in_no << " " << out_no << endl;
    in[frame_no] = in_no;
    out[frame_no] = out_no;
    
    total_in +=in_no;
    total_out +=out_no;
  }
  
  cout << "From the whole ground truth file, total in: " << total_in << " " <<", total out: " << total_out << endl;
  
}

void display_accuracy_of_occupancy_estimation()
{
  int total_in_gt = 0;
  int total_out_gt = 0;
  
  for(int i = startFrame; i <= endFrame; i++)
  {
    total_in_gt += in[i];
    total_out_gt += out[i];
  }
  
  cout << "Total people entered in ground truth: " << total_in_gt << endl;
  cout << "Total people left in ground truth: " << total_out_gt << endl;
  
  
  cout << "True entrance detected: " <<  true_entrance_detected << endl;
  cout << "False entrance detected: " << false_entrance_detected << endl;
  cout << "True leave detected: " <<  true_leave_detected << endl;
  cout << "False leave detected: " << false_leave_detected << endl;
  
  cout << "Accuracy of entrance detection: " << true_entrance_detected*100.0f/total_in_gt << endl;
  cout << "Accuracy of leaving detection: " << true_leave_detected*100.0f/total_out_gt << endl;
  
}

static void init_camera(rs::context &handle) {
  if (!handle.get_device_count())
    throw "error opening realsense device. check if it's connected";
  
  dev = handle.get_device(0); // grab first realsense device
  // 16-bit depth stream, 30 fps
  dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);
  
  dev->start();
  
  frame_height = dev->get_stream_height(rs::stream::depth);
  frame_width = dev->get_stream_width(rs::stream::depth);
  
  cout << "device serial: " << dev->get_serial() << endl;
  cout << "device firmware: " << dev->get_firmware_version() << endl;
  
  // wait for image stabilization
  for (int i = 0; i < 30; i++){
    dev->wait_for_frames();
  }
}

static void init_threading_structs() {
  pthread_mutex_init(&im_lock, nullptr);
  pthread_cond_init(&im_cond, nullptr);
  
  pthread_mutex_init(&frame_lock, nullptr);
  pthread_cond_init(&frame_cond, nullptr);
  
  pthread_mutex_init(&poly_lock, nullptr);
  pthread_cond_init(&poly_cond, nullptr);
}

static void queue_frames(const Mat &filtered_image, const Mat &people_image,
                         string &timestamp) {
  pthread_mutex_lock(&frame_lock);
  people_image_queue->push(make_tuple(people_image, timestamp));
  filtered_image_queue->push(filtered_image);
  pthread_cond_signal(&frame_cond);
  pthread_mutex_unlock(&frame_lock);
}

static void capture_frame(uint32_t &frame_no) {
  dev->wait_for_frames();
  const void *depth_frame = dev->get_frame_data(rs::stream::depth);
  string timestamp = logger->get_timestamp("time");
  
  // TDODO: correclty implement decrease fps function
//  if (decrease_fps) {
//    if (frame_no++ % fps_divisor)
//      continue;
//  }
  Mat people_image = Mat(frame_height, frame_width, CV_16U, (void *)depth_frame);
  Mat filtered_image = filter_based_on_max_depth(people_image);
  
  queue_frames(filtered_image, people_image, timestamp);
}

//void *frame_capture_thread(void *) {
//    if (use_realsense) {
//        rs::context handle; // manages all connected realsense devices
//        
//        if(!handle.get_device_count())
//        {
//            std::cout << "error opening realsense device. check if it's connected" << std::endl;
//            exit(-1);
//        }
//        
//        dev = handle.get_device(0); // grab first realsense device
//        // 16-bit depth stream, 30 fps
//        dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);
//        
//        dev->start();
//        
//        std::cout << "device serial: " << dev->get_serial() << std::endl;
//        std::cout << "device firmware: " << dev->get_firmware_version() << std::endl;
//        
//        // wait for image stabilization
//        for (int i = 0; i < 30; i++){
//            dev->wait_for_frames();
//        }
//        
//        frame_height = dev->get_stream_height(rs::stream::depth);
//        frame_width = dev->get_stream_width(rs::stream::depth);
//        
//        uint32_t frame_no = 0;
//        
//        while (true) {
//            dev->wait_for_frames();
//            const void *depth_frame = dev->get_frame_data(rs::stream::depth);
//            
//            if (decrease_fps) {
//                if (frame_no++ % fps_divisor)
//                    continue;
//            }
//            Mat people_image = Mat(frame_height, frame_width, CV_16U, (void *)depth_frame);
//            Mat filtered_image = filter_based_on_max_depth(people_image);
//            
////            if (background_image_determined)
////            {
////                for(int i = 0; i < filtered_image.rows; i++)
////                {
////                    for(int j = 0; j < filtered_image.cols; j++)
////                    {
////                        int h1 = filtered_image.at<ushort>(i,j);
////                        int h2 = background_image.at<ushort>(i,j);
////                        int diff = h1 - h2;
////                        if (abs(diff) < 1350)
////                            filtered_image.at<short>(i,j) = max_height_limit;
////                    }
////                }
////            }
//
//            pthread_mutex_lock(&frame_lock);
//            people_image_queue->push(people_image);
//            filtered_image_queue->push(people_image);
//            pthread_cond_signal(&frame_cond);
//            pthread_mutex_unlock(&frame_lock);
//        }
//    }
//    else {
//        for (int i = startFrame; i <= endFrame; i++) {
//            if (decrease_fps) {
//                if (i % fps_divisor)
//                    continue;
//            }
//            char image_loc_people[256];
//            sprintf(image_loc_people, IMG_LOC, i);
//            Mat people_image = readImage(image_loc_people, "Depth Data", false);
//            Mat filtered_image = filter_based_on_max_depth(people_image);
//            
//            
////            if (background_image_determined)
////            {
////                for(int i = 0; i < filtered_image.rows; i++)
////                {
////                    for(int j = 0; j < filtered_image.cols; j++)
////                    {
////                        int h1 = filtered_image.at<ushort>(i,j);
////                        int h2 = background_image.at<ushort>(i,j);
////                        int diff = h1 - h2;
////                        if (abs(diff) < 1350)
////                            filtered_image.at<short>(i,j) = max_height_limit;
////                    }
////                }
////            }
//
//            pthread_mutex_lock(&frame_lock);
//            people_image_queue->push(people_image);
//            filtered_image_queue->push(filtered_image);
//            pthread_cond_signal(&frame_cond);
//            pthread_mutex_unlock(&frame_lock);
//        }
//        done_capturing = true;
//    }
//    pthread_exit(NULL);
//}

static void display_frame() {
  pthread_mutex_lock(&im_lock);
  
  while (im_queue->empty())
    pthread_cond_wait(&im_cond, &im_lock);
  
  Mat img = im_queue->front();
  
  vector<struct person_info> temp_people = past_people_queue->front();
  int past_count = past_people_count.front();
  
  past_people_queue->pop();
  im_queue->pop();
  past_people_count.pop();
  
  pthread_mutex_unlock(&im_lock);
  
  display_occupancy_info(img, temp_people, past_count);
  cvWaitKey(1);
  if (display_polys)
    if (!poly_queue->empty()) {
      imshow("polys", poly_queue->front());
      poly_queue->pop();
    }
}

// TODO: make separate threads for frame capture and frame display
// TODO: make a decrease fps func
int main(int argc, char** argv) {
  signal(SIGINT, signal_handler);
  
  init_threading_structs();
  
  if (log_data)
    logger = new ForkLogger(stop_logger);
  
  pthread_create(&threads[0], nullptr, &process_frames, nullptr);
  
  if (use_realsense) {
    rs::context handle; // manages all connected realsense devices
    
    try {
      init_camera(handle);
    }
    catch (exception &e) {
      cerr << e.what();
      return EXIT_FAILURE;
    }
    
    uint32_t frame_no = 0;
    
    while (!(done_processing && im_queue->empty())) {
      if (!stop_frame_capture)
        capture_frame(frame_no);
      
      if (display_occupancy)
        display_frame();
    }
  }
  else {
    uint32_t frame_no = 0;
    
    for (int i = startFrame; i <= endFrame; i++) {
      char image_loc_people[256];
      sprintf(image_loc_people, IMG_LOC, i);
      
      if (decrease_fps) {
        if (frame_no++ % fps_divisor)
          continue;
      }
      Mat people_image = readImage(image_loc_people, "Depth Data", false);
      Mat filtered_image = filter_based_on_max_depth(people_image);
      
      string timestamp = logger->get_timestamp("time");
      
      queue_frames(filtered_image, people_image, timestamp);
      
      display_frame();
    }
    done_capturing = true;
  }
  kill_threads();
}

// retrieves pre-captured frames and computes occupancy count
void *process_frames(void *)
{
  assert(total_diff_heights <= MAX_HEIGHT_LEVELS);
  
  init_CSV(); // to generate the ground truth files for precision, recall computation
  
  // Initializing parameters for Accuracy calculations
  
  cout << "Total Number of CPUs: " << getNumberOfCPUs() << endl;
  cout << "Total threads running: " << getNumThreads() << endl;
  
  //setNumThreads(6);
  
  int background_freq[HEIGHT_ARRAY_SIZE];
  
  for (int i = 0; i < total_diff_heights; i++)
    background_freq[i] = INF_HEIGHT;
  
  int frame_no = 0;
  
  if(estimate_accuracy_of_occupancy_estimation)
    read_ground_truth_of_occupancy_estimation();
  
  int i = startFrame;
  clock_t t1,t2;
  t1 = clock();
  
  time_t time_start;
  time_start = time(NULL);
  
  clock_t clock_start = clock();
  clock_t time_vals[FPS_MAX_LEN] = {0};
  
  while (!(done_capturing && filtered_image_queue->empty())) {
    pthread_mutex_lock(&frame_lock);
    
    while(filtered_image_queue->empty()) {
      pthread_cond_wait(&frame_cond, &frame_lock);
    }
    
    
    std::tuple<Mat, string> raw_frame_info = people_image_queue->front();
    Mat people_image = get<0>(raw_frame_info);
    string timestamp = get<1>(raw_frame_info);
    
    Mat height_fixed_image = filtered_image_queue->front();
    
    people_image_queue->pop();
    filtered_image_queue->pop();
    
    pthread_mutex_unlock(&frame_lock);
    vector<Point2f> center;
    vector<float> radius;
    vector<float> depth;
    vector<vector<Point> > contours_poly;
    
    //height_fixed_image.convertTo(height_fixed_image, CV_32F);
    
    if (background_image_determined)
    {
      for(int i = 0; i < height_fixed_image.rows; i++)
      {
        for(int j = 0; j < height_fixed_image.cols; j++)
        {
          int h1 = height_fixed_image.at<ushort>(i,j);
          int h2 = background_image.at<ushort>(i,j);
          int diff = h1 - h2;
          if (abs(diff) < 1350)
            height_fixed_image.at<short>(i,j) = max_height_limit;
        }
      }
    }
    
    find_centers_using_different_heights(height_fixed_image, background_freq, center, radius, depth, contours_poly,i, frame_no, people_image);
    
    Mat image_with_circles = add_circles(height_fixed_image, center, radius, contours_poly, i);
    if(!suppress_images) {
      drawImageNoDivide(image_with_circles, "Image with circles");
    }
    
    // Update info of people
    update_people_info(center, radius, depth); //it will update the vector: people_info
    //update the estimate of occupancy
    estimate_occupancy(people_image_no, 1); //it will use people_info and update
    cout << "PEOPLE inside: " << occupancy << endl;
    
    //display occupancy info
    if (display_occupancy) {
      pthread_mutex_lock(&im_lock);
      im_queue->push(image_with_circles);
      past_people_queue->push(people_info);
      past_people_count.push(occupancy);
      pthread_mutex_unlock(&im_lock);
      pthread_cond_signal(&im_cond);
    }
    
    if (store_frame) {
      logger->log_data(image_with_circles, occupancy,
                       logger->get_timestamp("time"));
    }
    store_frame = false;
    
    if (occupancy < 0)
      occupancy = 0;
    
    // Calculating average number of frames per second
    frame_no++;
    clock_t current_time = clock();
    float time_diff_all = (float)(current_time - clock_start)/CLOCKS_PER_SEC;
    float frames_per_sec_all = (float)(frame_no)/time_diff_all;
    
    float time_diff_20_frames = (float) (current_time - time_vals[frame_no%FPS_MAX_LEN])/CLOCKS_PER_SEC;
    time_vals[frame_no%FPS_MAX_LEN] = current_time;
    float frames_per_sec_20 = (float)(FPS_MAX_LEN)/time_diff_20_frames;
    
    cout << "FPS: overall, current: " << frames_per_sec_all <<"," << frames_per_sec_20 << endl; //frames per sec
    i++;
  }
  
  // Calculate total time required to run the algorithm
  t2 = clock();
  float diff = ((float)t2-(float)t1);
  int seconds = diff / CLOCKS_PER_SEC;
  cout<< "Time Taken : " << seconds/60 << " Minutes , " << seconds%60 << " Seconds " << endl;
  
  time_t time_end = time(NULL);
  long time_diff = time_end - time_start; //in seconds
  cout <<"Actual time passed: " << time_diff/60 << " Minutes,  " << time_diff%60 << " Seconds." << endl;
  
  done_processing = true;
  
  pthread_exit(NULL);
}
