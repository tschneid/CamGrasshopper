#include "camGrasshopper.h"
#include "grasshopper.h"
#include <thread>

camGrasshopper::camGrasshopper(BVS::ModuleInfo info, const BVS::Info& bvs)
	: BVS::Module()
	, info(info)
	, logger(info.id)
	, bvs(bvs)
	, outputs()
	, camOrder()
	, g(bvs.config.getValue<int>(info.conf + ".trigger", 0), true)
	, numCameras(0)
	, resolution()
	, encoding(bvs.config.getValue<std::string>(info.conf + ".encoding", "Y8"))
	, framerate(bvs.config.getValue<float>(info.conf + ".framerate", 15))
	, masterCam(bvs.config.getValue<int>(info.conf + ".masterCam", -1))
	, shutter(bvs.config.getValue<int>(info.conf + ".shutter", -1))
	, triggerThread(bvs.config.getValue<bool>(info.conf + ".triggerThread", true))
	, triggerRunning(false)
	, triggerExit(false)
	, mutex()
	, masterLock(mutex)
	, triggerCond()
	, trigger()
{
	bvs.config.getValue<int>(info.conf + ".resolution", resolution);
	if (resolution.size() != 2) resolution = {1024, 768};

	if (!g.initCameras(resolution[0], resolution[1], encoding, framerate))
		LOG(1, "Something went wrong while initializing the cameras!");

	//g.printVideoModes(0);
	if (shutter > 0)
	{
		g.setShutter(shutter);
	}
	
	numCameras = g.getNumCameras();
	if (numCameras == 0)
		LOG(1, "No cameras detected!");


	for (unsigned int i = 0; i < numCameras; ++i)
	{
		outputs.push_back( new BVS::Connector<cv::Mat>(std::string("out")+std::to_string(i+1), BVS::ConnectorType::OUTPUT) );
	}
	g.getNextFrame();
	
	std::map<int, int> remap;
	for (unsigned int i=0; i<numCameras; i++) remap[g.getCameraSerialNumber(i)]=i;
	for (auto& it: remap) camOrder.push_back(it.second);

	if (triggerThread)
	{
		triggerRunning = true;
		trigger = std::thread(&camGrasshopper::startTriggerThread, this);
	}
}



camGrasshopper::~camGrasshopper()
{
	if (triggerThread)
	{
		triggerExit = true;
		triggerRunning = true;
		masterLock.unlock();
		triggerCond.notify_one();
		if (trigger.joinable()) trigger.join();
	}

	g.restoreDefaultProperties();
    g.stopCameras();
}



BVS::Status camGrasshopper::execute()
{
	if (triggerThread) triggerCond.wait(masterLock, [&](){ return !triggerRunning; });
	else triggerCameras();

	cv::Mat img;
	for (unsigned int i = 0; i < numCameras; ++i)
	{
		img = g.getImage(camOrder[i]);
		outputs[i]->send(img);
	}

	if (triggerThread)
	{
		triggerRunning = true;
		triggerCond.notify_one();
	}

	return BVS::Status::OK;
}



void camGrasshopper::triggerCameras()
{
	if (masterCam >= 0) g.distributeCamProperties(masterCam);
	g.getNextFrame();
}



void camGrasshopper::startTriggerThread()
{
	BVS::nameThisThread("camGH.trigger");
	std::unique_lock<std::mutex> triggerLock(mutex);
	while (!triggerExit)
	{
		triggerCond.wait(triggerLock, [&](){ return triggerRunning; });
		triggerCameras();
		triggerRunning = false;
		triggerCond.notify_one();
	}
	triggerCond.notify_one();
}


BVS::Status camGrasshopper::debugDisplay()
{
	return BVS::Status::OK;
}