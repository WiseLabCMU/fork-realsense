//
//  fork_logger.hpp
//  fork
//
//  Created by Jonathan Appiagyei on 10/19/17.
//  Copyright © 2017 Carnegie Mellon WiSE Lab. All rights reserved.
//

#ifndef fork_logger_hpp
#define fork_logger_hpp

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pthread.h>
#include <queue>
#include <stdio.h>
#include <string>
#include <tuple>
#include <unistd.h>

namespace fork_logger {
  
  // threaded logger to log occupancy count including the corresponding image
  // and timestamp for the count
  class ForkLogger {
  public:
    ForkLogger(bool *stop_logger, std::string folder_name = "none");
    std::string get_timestamp(std::string format);
    pthread_cond_t _queue_cond;

    // accessor for client
    // enqueues image, occupancy count, and timestamp
    void log_data(cv::Mat img, int count, std::string timestamp);
  private:
    std::string _cwd;
    std::string _folder_name; // where counts are logged
    std::string _init_date; // when logger started
    std::string _init_time; // when logger initialized
    pthread_t _log_thread;
    pthread_mutex_t _queue_lock;
    std::ofstream _log_out; // log file
    std::queue<std::tuple<cv::Mat, int, std::string>> _log_queue;
    bool *_stop_logger;
    static void * start_thread(void *This);
    // unpacks data to be logged and saves items to disk
    void process_queue();
  };
}

#endif /* fork_logger_hpp */

