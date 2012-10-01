#include "camGrasshopper.h"
#include "grasshopper.h"
#include <thread>



// This is your module's constructor.
// Please do not change its signature as it is called by the framework (so the
// framework actually creates your module) and the framework assigns the unique
// identifier and gives you access to its config.
// However, you should use it to create your data structures etc.
camGrasshopper::camGrasshopper(const std::string id, const BVS::Info& bvs)
	: BVS::Module()
	, id(id)
	, logger(id)
	, bvs(bvs)
	, outputs()
	, camOrder()
	, config("capture", 0, nullptr, "camGrasshopperConfig.txt")
	, g(config.getValue<int>(id + ".trigger", 2))
	, numCameras(0)
	, masterCam(config.getValue<int>(id + ".masterCam", -1))
	, shutterSpeed(config.getValue<int>(id + ".shutterSpeed", -1))
	, triggerRunning(false)
	, triggerExit(false)
	, mutex()
	, masterLock(mutex)
	, triggerCond()
	, trigger()
{
	// initialize cameras with defined video mode and frame rate
	//g.initCameras(FlyCapture2::VIDEOMODE_1024x768RGB, FlyCapture2::FRAMERATE_7_5);
	//g.initCameras(FlyCapture2::VIDEOMODE_1600x1200RGB, FlyCapture2::FRAMERATE_7_5);
	//g.initCameras(FlyCapture2::VIDEOMODE_1600x1200Y8, FlyCapture2::FRAMERATE_15);
	g.initCameras(FlyCapture2::VIDEOMODE_1024x768Y8, FlyCapture2::FRAMERATE_15);

	//g.printVideoModes(0);
	if (shutterSpeed > 0)
	{
		g.setShutter(shutterSpeed);
	}
	numCameras = g.getNumCameras();

	for (unsigned int i = 0; i < numCameras; ++i)
	{
		outputs.push_back( new BVS::Connector<cv::Mat>(std::string("out")+std::to_string(i+1), BVS::ConnectorType::OUTPUT) );
	}
	g.getNextFrame();
	
	std::map<int, int> remap;
	for (unsigned int i=0; i<numCameras; i++) remap[g.getCameraSerialNumber(i)]=i;
	for (auto& it: remap) camOrder.push_back(it.second);

	triggerRunning = true;
	trigger = std::thread(&camGrasshopper::triggerCameras, this);
}



// This is your module's destructor.
// See the constructor for more info.
camGrasshopper::~camGrasshopper()
{
	triggerExit = true;
	triggerRunning = true;
	masterLock.unlock();
	triggerCond.notify_one();
	if (trigger.joinable()) trigger.join();

	g.restoreDefaultProperties();
    g.stopCameras();
}



// Put all your work here.
BVS::Status camGrasshopper::execute()
{
	triggerCond.wait(masterLock, [&](){ return !triggerRunning; });

	cv::Mat img;
	for (unsigned int i = 0; i < numCameras; ++i)
	{
		img = g.getImage(camOrder[i]);
		outputs[i]->send(img);
	}

	triggerRunning = true;
	triggerCond.notify_one();

	return BVS::Status::OK;
}



void camGrasshopper::triggerCameras()
{
	BVS::nameThisThread("camGH.trigger");
	std::unique_lock<std::mutex> triggerLock(mutex);
	while (!triggerExit)
	{
		triggerCond.wait(triggerLock, [&](){ return triggerRunning; });
		if (masterCam >= 0) g.distributeCamProperties(masterCam);
		g.getNextFrame();
		triggerRunning = false;
		triggerCond.notify_one();
	}
	triggerCond.notify_one();
}



// UNUSED
BVS::Status camGrasshopper::debugDisplay()
{
	return BVS::Status::OK;
}



// This function is called by the framework upon creating a module instance of
// this class. It creates the module and registers it within the framework.
// DO NOT CHANGE OR DELETE
extern "C" {
	int bvsRegisterModule(std::string id, BVS::Info& bvs)
	{
		registerModule(id, new camGrasshopper(id, bvs));

		return 0;
	}
}

