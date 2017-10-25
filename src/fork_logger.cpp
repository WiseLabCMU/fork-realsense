//
//  fork_logger.cpp
//  fork
//
//  Created by Jonathan Appiagyei on 10/19/17.
//  Copyright © 2017 Carnegie Mellon WiSE Lab. All rights reserved.
//

#include "fork_logger.hpp"

using namespace std;
using namespace cv;

namespace fork_logger {

  ForkLogger::ForkLogger(string filename) : _filename(filename) {
    pthread_mutex_init(&_queue_lock, nullptr);
    pthread_cond_init(&_queue_cond, nullptr);
    
    pthread_create(&_log_thread, nullptr, &start_thread, this);
  }
  
  string ForkLogger::get_timestamp() {
    time_t curr_time = time(nullptr);
    struct tm *local_time = localtime_r(&curr_time, nullptr);
    char time_array[25];
    
    strftime(time_array, 50, "%Y-%m-%d_%H:%M:S", local_time);
    string date_time(time_array);
    
    return date_time;
  }
  
  void ForkLogger::log_data(std::tuple<cv::Mat, int, std::string> info) {
    _log_queue.push(info);
  }
  
  void *ForkLogger::start_thread(void *This) {
    ((ForkLogger *)This)->process_queue();
    return nullptr;
  }
  
  void ForkLogger::process_queue() {
    // TODO: terminate this loop correctly
    while (true) {
      pthread_mutex_lock(&_queue_lock);
      while (_log_queue.empty())
        pthread_cond_wait(&_queue_cond, &_queue_lock);
      
      std::tuple<cv::Mat, int, std::string> info = _log_queue.front();
      pthread_mutex_unlock(&_queue_lock);
      
      Mat img = get<0>(info);
      int count = get<1>(info);
      string timestamp = get<2>(info);
      _log_out << timestamp << ' ' << count << endl;
      // TODO: create directories by date
      imwrite(timestamp + ".png", img);
    }
  }
}


