#ifndef _GRASSHOPPER_HPP_
#define _GRASSHOPPER_HPP_

// main() function with minimal example program (using OpenCV highgui)
//#define _STANDALONE

///////////////////////////////////////////////////////////////////////////////
// Triggering Info
//
// SOFTWARE_TRIGGER: A register on the cameras will be written to initiate the capture of a frame.
// FIREWIRE_TRIGGER: The cameras synchronize over the firewire bus. Therefore they have to be on the same bus (no dual-bus).
// HARDWARE_TRIGGER: The cameras receive an external signal to trigger frame capturing at a specified GPIO port.
// 
// To use the Firewire Trigger the cameras have to be on the same bus (no dual-bus).
// Cameras on the same bus should synchronize automatically! The difference between
// the Firewire trigger and no trigger is the call to FlyCapture:
// "StartSyncCapture()" vs. "StartCapture()"
//
// Using the Firewire trigger resulted in some polling problems, so I recommend to use
// no trigger at all, when the cameras are on the same bus.
///////////////////////////////////////////////////////////////////////////////


// Makro for Flycapture Videomodes
#define VIDEOMODE(width,height,encoding) VIDEOMODE_##width##x##height##encoding

///////////////////////////////////////////////////////////////////////////////
// Possible video modes
//
// VIDEOMODE_160x120YUV444 		160x120 YUV444.
// VIDEOMODE_320x240YUV422 		320x240 YUV422.
// VIDEOMODE_640x480YUV411 		640x480 YUV411.
// VIDEOMODE_640x480YUV422 		640x480 YUV422.
// VIDEOMODE_640x480RGB 		640x480 24-bit RGB.
// VIDEOMODE_640x480Y8 			640x480 8-bit.
// VIDEOMODE_640x480Y16 		640x480 16-bit.
// VIDEOMODE_800x600YUV422 		800x600 YUV422.
// VIDEOMODE_800x600RGB 		800x600 RGB.
// VIDEOMODE_800x600Y8 			800x600 8-bit.
// VIDEOMODE_800x600Y16 		800x600 16-bit.
// VIDEOMODE_1024x768YUV422 	1024x768 YUV422.
// VIDEOMODE_1024x768RGB 		1024x768 RGB.
// VIDEOMODE_1024x768Y8 		1024x768 8-bit.
// VIDEOMODE_1024x768Y16 		1024x768 16-bit.
// VIDEOMODE_1280x960YUV422 	1280x960 YUV422.
// VIDEOMODE_1280x960RGB 		1280x960 RGB.
// VIDEOMODE_1280x960Y8 		1280x960 8-bit.
// VIDEOMODE_1280x960Y16 		1280x960 16-bit.
// VIDEOMODE_1600x1200YUV422 	1600x1200 YUV422.
// VIDEOMODE_1600x1200RGB 		1600x1200 RGB.
// VIDEOMODE_1600x1200Y8 		1600x1200 8-bit.
// VIDEOMODE_1600x1200Y16 		1600x1200 16-bit.
// VIDEOMODE_FORMAT7
///////////////////////////////////////////////////////////////////////////////

#include "FlyCapture2.h"

#include <vector>
#include <iostream>
#include <unistd.h>
#include <sstream>
#include <map>
#include <algorithm>

#include <opencv2/core/core.hpp>
#ifdef _STANDALONE
	#include <opencv2/highgui/highgui.hpp>
#endif
//#include <dc1394/dc1394.h>

#include <sys/time.h>

#ifdef _WITH_OPENCL
    #ifdef __APPLE__
    	#include "OpenCL/opencl.h" // not tested! is this correct?
    #else
    	#include <CL/opencl.h>
    #endif
#endif

#ifdef _WITH_OPENCL
static const char* errorToString(cl_int);
//static void clLoadProgram(const char*, char**, size_t*);
static void clPrintBuildLog(cl_program, cl_device_id);
#endif

using namespace FlyCapture2;

class Grasshopper
{
public:
	static const int NO_TRIGGER = 0;
	static const int SOFTWARE_TRIGGER = 1;
	static const int FIREWIRE_TRIGGER = 2;
	static const int HARDWARE_TRIGGER = 3;

	Grasshopper(int triggerSwitch = NO_TRIGGER, bool BGRtoRGB = false);

	// Initialize each connected PointGrey Grasshopper camera.
	bool initCameras(const int width, const int height, const std::string& encoding, const float& framerate);
	bool initCameras(VideoMode videoMode, FrameRate frameRate);

	// Close the connection to all cameras.
	bool stopCameras();

	// triggering and retrieving frames
	bool getNextFrame();
	cv::Mat getImage(const int i = 0);

	// printing informations
	void printInfo();
	void printCamInfo(CameraInfo* pCamInfo);
	void printVideoModes(const int i = 0);
	void printImageMetadata(const int i = 0); // embedded data

	// changing and monitoring camera properties
	bool setShutter(const int milliseconds = 20);
	bool distributeCamProperties(const unsigned int master); // if the master is changed, you should first restore the default properties
	bool restoreDefaultProperties(const int i = -1);
	bool testPropertiesForManualMode();
	std::string getProperty(const PropertyType& propType, const int i); // Shutter, Gain, etc.

	// region of interest -- experimental!
	// (Changing the ROI currently takes about 1 second, so it's much to slow to do it
	// in each iteration. This is because you have to stop the cameras, set the settings,
	// and start them again.)
	bool setROI(const cv::Rect& roi, const unsigned int cam);
	bool setROI(const int x, const int y, const int width, const int height, const unsigned int cam);

	// some additional features
	TimeStamp getTimestamp(const int i = 0);
	unsigned int getCycleCount(const int i = 0) const;
	bool saveImages(const int imgNum = 0); // very primitive
	Image getFlyCapImage(const int i = 0);
	int getCameraSerialNumber(int index);

	// display frames per second
	double tickFPS(); // Use this one time in your main loop
					  // to get the processed frames per second
	double getProcessedFPS() { return fps; };
	std::string getProcessedFPSString();

	// getter and setter
	int getNumCameras() { return numCameras; };
	int getChannels() { return numCameras; };

private:
	unsigned int numCameras;
	int GPIO_TRIGGER_SOURCE_PIN;
	int TRIGGER_MODE_NUMBER;
	bool BGRtoRGB;

	int width, height;
	std::string encoding;
	float framerate;

	// Camera properties and flag if they can be used in manual mode
	std::map<PropertyType, bool> manualProp;

	Error error;
    BusManager busMgr;
    Camera** ppCameras;
    Image* images;
    void printError( Error error ) { error.PrintErrorTrace(); };
    bool PollForTriggerReady( Camera* pCam );
    bool CheckSoftwareTriggerPresence( Camera* pCam );
    bool FireSoftwareTrigger( Camera** pCam );
  	static std::string toString(const VideoMode& vm);
	static std::string toString(const FrameRate& fps);
	static std::string toString(const PropertyType& prop);
	static VideoMode getVideoMode(const int width, const int height, const std::string& encoding);
	static FrameRate getFrameRate(const float& fps);
	static void getCameraParameters(const VideoMode& vm, const FrameRate& fr, int& width, int& height, std::string& encoding, float& framerate);

	// embed information in the first few pixels
	bool embedTimestamp, embedGain, embedShutter,
		 embedBrightness, embedExposure, embedWhiteBalance,
		 embedFrameCounter, embedStrobePattern,
    	 embedGPIOPinState, embedROIPosition;

   	// timestamp calculation
    double old_ts, fps;

	// trigger mode
	int triggerSwitch;

#ifdef _WITH_OPENCL
	bool useGPU;
	cl_context clContext;
    cl_command_queue clCommandQueue;
    cl_device_id clDevice;
    cl_program clProgram;
    cl_kernel clKernel;
    cl_mem dYuv, dRgb;
    bool initializeOpenCL();
    void cleanupOpenCL();
    void yuv422toRGB_gpu(const cv::Mat& yuv, cv::Mat& rgb, const bool BGRtoRGB = false);
#endif

    void yuv422toRGB(const cv::Mat& yuv, cv::Mat& rgb, const bool BGRtoRGB = false);

	Grasshopper(const Grasshopper&) = delete; /**< -Weffc++ */
	Grasshopper& operator=(const Grasshopper&) = delete; /**< -Weffc++ */
};

#endif
