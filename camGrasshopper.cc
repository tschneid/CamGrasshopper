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
	, g(bvs.config.getValue<int>(id + ".trigger", 0))
	, numCameras(0)
	, resolution()
	, encoding(bvs.config.getValue<std::string>(id + ".encoding", "Y8"))
	, framerate(bvs.config.getValue<float>(id + ".framerate", 15))
	, masterCam(bvs.config.getValue<int>(id + ".masterCam", -1))
	, shutter(bvs.config.getValue<int>(id + ".shutter", -1))
	, triggerRunning(false)
	, triggerExit(false)
	, mutex()
	, masterLock(mutex)
	, triggerCond()
	, trigger()
{
	bvs.config.getValue<int>(id + ".resolution", resolution);
	if (resolution.size() != 2) resolution = {1024, 768};

	g.initCameras(resolution[0], resolution[1], encoding, framerate);


	//g.printVideoModes(0);
	if (shutter > 0)
	{
		g.setShutter(shutter);
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

