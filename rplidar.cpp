#include "rplidar.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <cmath>

// подключаем sdk
#include "sl_lidar.h" 
#include "sl_lidar_driver.h"

#include <stdio.h> 
#include <stdlib.h> 
#include <vector>
#include <mutex>

std::mutex copy_mutex;
  
int value = 0;
bool run = true;
bool new_scan = false;

#ifndef _countof
#define _countof(_Array) (int)(sizeof(_Array) / sizeof(_Array[0]))
#endif

#ifdef _WIN32
#include <Windows.h>
#define delay(x)   ::Sleep(x)
#else
#include <unistd.h>
static inline void delay(sl_word_size_t ms){
    while (ms>=1000){
        usleep(1000*1000);
        ms-=1000;
    };
    if (ms!=0)
        usleep(ms*1000);
}
#endif

using namespace sl;

std::vector<double> sc;

static constexpr double wrapAngle(double angle) {

    // Находим количество полных оборотов в угле и домножаем на 2π.
    const double a = int((__builtin_fabs(angle) + M_PI) * (1.0 / (2 * M_PI))) * (2 * M_PI);

    if (angle > M_PI) {
        angle = angle - a;
        return angle;
    }

    if (angle < -M_PI) {
        angle = angle + a;
        return angle;
    }

    return angle;
}

int lidar_main(int argc, const char * argv[]) {
	const char *opt_channel = NULL;
    const char *opt_channel_param_first = NULL;
    sl_u32      opt_channel_param_second = 0;
    sl_result   op_result;
	int         opt_channel_type = CHANNEL_TYPE_SERIALPORT;

    IChannel* _channel;

    printf("%s\n", argv[0]);
    printf("%s\n", argv[1]);
    printf("%s\n", argv[2]);
    printf("%s\n", argv[3]);
    printf("%s\n", argv[4]);


	const char * opt_is_channel = argv[1];
	if(strcmp(opt_is_channel, "--channel")==0)
	{
		opt_channel = argv[2];
		if(strcmp(opt_channel, "-s")==0||strcmp(opt_channel, "--serial")==0)
		{
			opt_channel_param_first = argv[3];
			if (argc>4) opt_channel_param_second = strtoul(argv[4], NULL, 10);
		}
		else if(strcmp(opt_channel, "-u")==0||strcmp(opt_channel, "--udp")==0)
		{
			opt_channel_param_first = argv[3];
			if (argc>4) opt_channel_param_second = strtoul(argv[4], NULL, 10);
			opt_channel_type = CHANNEL_TYPE_UDP;
		}
		else
		{
			
			return -1;
		}
	}
    else
	{
		
		return -1;
	}

    // create the driver instance
	ILidarDriver * drv = *createLidarDriver();

    if (!drv) {
        fprintf(stderr, "insufficent memory, exit\n");
        exit(-2);
    }

    sl_lidar_response_device_health_t healthinfo;
    sl_lidar_response_device_info_t devinfo;
    
    // try to connect
    if (opt_channel_type == CHANNEL_TYPE_SERIALPORT) {
        _channel = (*createSerialPortChannel(opt_channel_param_first, opt_channel_param_second));
    }
    else if (opt_channel_type == CHANNEL_TYPE_UDP) {
        _channel = *createUdpChannel(opt_channel_param_first, opt_channel_param_second);
    }
    
    if (SL_IS_FAIL((drv)->connect(_channel))) {
        switch (opt_channel_type) {	
            case CHANNEL_TYPE_SERIALPORT:
                fprintf(stderr, "Error, cannot bind to the specified serial port %s.\n"
                    , opt_channel_param_first);
                return -1;
            case CHANNEL_TYPE_UDP:
                fprintf(stderr, "Error, cannot connect to the ip addr %s with the udp port %u.\n"
                    , opt_channel_param_first, opt_channel_param_second);
                return -1;
        }
    }

    // retrieving the device info
    ////////////////////////////////////////
    op_result = drv->getDeviceInfo(devinfo);

    if (SL_IS_FAIL(op_result)) {
        if (op_result == SL_RESULT_OPERATION_TIMEOUT) {
            // you can check the detailed failure reason
            fprintf(stderr, "Error, operation time out.\n");
        } else {
            fprintf(stderr, "Error, unexpected error, code: %x\n", op_result);
            // other unexpected result
        }
        return -1;
    }

    // print out the device serial number, firmware and hardware version number..
    printf("SLAMTEC LIDAR S/N: ");
    for (int pos = 0; pos < 16 ;++pos) {
        printf("%02X", devinfo.serialnum[pos]);
    }

    printf("\n"
            "Version:  %s \n"
            "Firmware Ver: %d.%02d\n"
            "Hardware Rev: %d\n"
            , "SL_LIDAR_SDK_VERSION"
            , devinfo.firmware_version>>8
            , devinfo.firmware_version & 0xFF
            , (int)devinfo.hardware_version);


    // check the device health
    ////////////////////////////////////////
    op_result = drv->getHealth(healthinfo);
    if (SL_IS_OK(op_result)) { // the macro IS_OK is the preperred way to judge whether the operation is succeed.
        printf("Lidar health status : ");
        switch (healthinfo.status) 
        {
            case SL_LIDAR_STATUS_OK:
                printf("OK.");
                break;
            case SL_LIDAR_STATUS_WARNING:
                printf("Warning.");
                return -1;
            case SL_LIDAR_STATUS_ERROR:
                printf("Error.");
                return -1;
        }
        printf(" (errorcode: %d)\n", healthinfo.error_code);

    } else {
        fprintf(stderr, "Error, cannot retrieve the lidar health code: %x\n", op_result);
        return -1;
    }


    if (healthinfo.status == SL_LIDAR_STATUS_ERROR) {
        fprintf(stderr, "Error, slamtec lidar internal error detected. Please reboot the device to retry.\n");
        // enable the following code if you want slamtec lidar to be reboot by software
        // drv->reset();
        return -1;
    }

    switch (opt_channel_type) 
    {	
        case CHANNEL_TYPE_SERIALPORT:
            drv->setMotorSpeed();
        break;
    }

    // take only one 360 deg scan and display the result as a histogram
    ////////////////////////////////////////////////////////////////////////////////
    if (SL_IS_FAIL(drv->startScan( 0,1 ))) // you can force slamtec lidar to perform scan operation regardless whether the motor is rotating
    {
        fprintf(stderr, "Error, cannot start the scan operation.\n");
        return -1;
    }

    while(run) {
		delay(10);

        sl_result ans;
        
        sl_lidar_response_measurement_node_hq_t nodes[8192];
        size_t   count = _countof(nodes);    

        ans = drv->grabScanDataHq(nodes, count, 0);

        std::lock_guard<std::mutex> guard(copy_mutex);
        if (SL_IS_OK(ans)) {

            drv->ascendScanData(nodes, count);            
            
            sc.clear();
            for (int pos = 0 ; pos < (int)count; ++pos) {
                const double alpha = wrapAngle(nodes[pos].angle_z_q14 * (M_PI_2 / 16384.0));                
                const double distance = nodes[pos].dist_mm_q2/1000.f/4.0f;

                const double x = distance * cos(alpha);
                const double y = distance * sin(alpha);
                sc.push_back(x);
                sc.push_back(y);
                sc.push_back(0);
            }

        } 
        else { 
            // обработки ошибок нет
            // printf("error code: %x\n", ans);
        }

    };

    drv->stop();
    switch (opt_channel_type) 
	{	
		case CHANNEL_TYPE_SERIALPORT:
			delay(20);
			drv->setMotorSpeed(0);
		break;
	}
    if(drv) {
        delete drv;
        drv = NULL;
    }
    return 0;
}


// функция для потока, в котором будет обрабатываться лидар
static void* thread_function(void* arg) 
{    
    // настройки для лидара 
    const char* cmd_line[5];
    cmd_line[0] = "./simple_grabber";
    cmd_line[1] = "--channel";
    cmd_line[2] = "--serial";
    cmd_line[3] = "/dev/ttyUSB0";
    cmd_line[4] = "256000";
    
    lidar_main(sizeof(cmd_line), cmd_line);

    
    printf("Thread exit\n");
    pthread_exit(NULL);
}   

// в конструкторе создадим поток в котором будут обрабатываться данные лидара 
RpLidar::RpLidar(){    
    printf("Lidar init.\n");
    pthread_create(&ptid, NULL,thread_function, NULL);     
}

// в деструкторе освободим ресурсы
RpLidar::~RpLidar(){
    this->stop();
}

// если новых данных нет, то будем возвращать пустой массив
std::vector<double>& RpLidar::get_scan(){

    if(sc.size() == 0){
        return this->empty;
    }

    // копируем данные и очищаем исходный вектор
    std::lock_guard<std::mutex> guard(copy_mutex);
    this->scan = sc;
    sc.clear();
    return this->scan;

}

// здесь мы ожидаем завершение потока и освобождения ресурсов выделенных под лидар
void RpLidar::stop(){    
    printf("Lidar exit.\n");
    run = false;
    pthread_join(ptid, NULL);    
}