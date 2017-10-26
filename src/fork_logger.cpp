//
//  fork_logger.cpp
//  fork
//
//  Created by Jonathan Appiagyei on 10/19/17.
//  Copyright © 2017 Carnegie Mellon WiSE Lab. All rights reserved.
//

#include "fork_logger.hpp"

using namespace cv;
using namespace std;

namespace fork_logger {

  ForkLogger::ForkLogger(string folder_name){
    char dir_buff[FILENAME_MAX];
    _cwd = getcwd(dir_buff, FILENAME_MAX);
    _init_time = get_timestamp("time");
    _init_date = get_timestamp("date");
    
    if (folder_name == "none")
      _folder_name = _init_time;
    else
      _folder_name = folder_name;
    
    boost::filesystem::path dir(_cwd + '/' + _folder_name);
    
    if (!boost::filesystem::exists(dir)) {
      boost::filesystem::create_directory(dir);
    }
    _log_out.open(_cwd + '/' + _folder_name + '/' + "realsense.log",
                  ofstream::out |
                  ofstream::trunc);
    
    pthread_mutex_init(&_queue_lock, nullptr);
    pthread_cond_init(&_queue_cond, nullptr);
    
    pthread_create(&_log_thread, nullptr, &start_thread, this);
  }
  
  string ForkLogger::get_timestamp(string format) {
    time_t curr_time = time(&curr_time);
    string date_time;
    struct tm local_time;
    localtime_r(&curr_time, &local_time);
    char time_array[25];
    string date;
    if (format == "time") {
//      strftime(time_array, 50, "%Y-%m-%d_%H:%M:%S", &local_time);
      boost::posix_time::ptime tod =
      boost::posix_time::microsec_clock::local_time();
      date_time = boost::posix_time::to_iso_extended_string(tod);
    }
    else {
      strftime(time_array, 50, "%Y-%m-%d", &local_time);
      date_time = string(time_array);
    }
    
    return date_time;
  }
  
  void ForkLogger::log_data(Mat img, int count, string timestamp) {
    tuple<Mat, int, string> info = make_tuple(img, count, timestamp);
    pthread_mutex_lock(&_queue_lock);
    _log_queue.push(info);
    pthread_cond_signal(&_queue_cond);
    pthread_mutex_unlock(&_queue_lock);
  }
  
  void *ForkLogger::start_thread(void *This) {
    ((ForkLogger *)This)->process_queue();
    return nullptr;
  }
  
  void ForkLogger::process_queue() {
    int prev_count = 0;
    // TODO: terminate this loop correctly
    while (true) {
      pthread_mutex_lock(&_queue_lock);
      while (_log_queue.empty())
        pthread_cond_wait(&_queue_cond, &_queue_lock);
      
      std::tuple<cv::Mat, int, std::string> info = _log_queue.front();
      _log_queue.pop();
      pthread_mutex_unlock(&_queue_lock);
      
      Mat img = get<0>(info);
      int count = get<1>(info);
      string timestamp = get<2>(info);
      
      if (count != prev_count)
        _log_out << timestamp << ' ' << count << endl;
      imwrite(_cwd + '/' + _folder_name + '/' + timestamp + ".png", img);
      prev_count = count;
    }
    _log_out.close();
  }
}


