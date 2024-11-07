#include "stdio.h"
#include <vector>
#include <pthread.h> 

class RpLidar{

public:
    RpLidar();
    ~RpLidar();

    std::vector<double>& get_scan();
    void stop();

private:
    std::vector<double> scan{};    
    std::vector<double> empty{}; 
    pthread_t ptid; 

};