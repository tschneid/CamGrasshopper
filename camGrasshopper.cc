#include "camGrasshopper.h"
#include "grasshopper.h"


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
	, config("capture", 0, nullptr, "camGrasshopperConfig.txt")
	, g()
	, numCameras(0)
	, masterCam(config.getValue<int>(id + ".masterCam", -1))
	, shutterSpeed(config.getValue<int>(id + ".shutterSpeed", -1))
{
	// initialize cameras with defined video mode and frame rate
	g.initCameras(FlyCapture2::VIDEOMODE_1024x768RGB, FlyCapture2::FRAMERATE_7_5);

	g.printVideoModes(0);
	if (shutterSpeed > 0)
	{
		g.setShutter(shutterSpeed);
	}
	numCameras = g.getNumCameras();

	for (unsigned int i = 0; i < numCameras; ++i)
	{
		outputs.push_back( new BVS::Connector<cv::Mat>(std::string("out")+std::to_string(i+1), BVS::ConnectorType::OUTPUT) );
	}
}



// This is your module's destructor.
// See the constructor for more info.
camGrasshopper::~camGrasshopper()
{
	g.restoreDefaultProperties();
    g.stopCameras();
}



// Put all your work here.
BVS::Status camGrasshopper::execute()
{
	// to log messages to console or file, use the LOG(...) macro
	//LOG(3, "Execution of " << id << "!");

	LOG(3, "Execution of " << id << "!");

	// Various settings and information
	// in some config:
	// [thisModuleId]
	// foo = 42
	//int foo = bvs.getValue<int>(id + ".myInteger, 23);
	//unsigned long long round = bvs.round;
	//int lastRoundModuleDuration = bvs.moduleDurations.find(id)->second.count();
	//int lastRoundDuration = bvs.lastRoundDuration.count();

	// Simple Connector Example
	//int incoming;
	//std::string message;
	//if (input.receive(incoming))
	//{
	//	message = "received" + std::to_string(incoming);
	//}
	//else
	//{
	//	message = "no input received!";
	//}
	//output.send(message);

	if (masterCam >= 0)
	{
		g.distributeCamProperties(masterCam);
	}
	
	g.getNextFrame();
	
	for (int i = 0; i < numCameras; ++i)
	{
		cv::Mat img = g.getImage(i);
		outputs[i]->send(img);
	}

	// Advanced Connector Example (do not forget to unlock the connection or
	// you will cause deadlocks)
	//std::string s2 = "This";
	//output.lockConnection();
	//*output = s2;
	//*output = *output + " is an";
	//s2 = " example!";
	//*output += s2;
	//output.unlockConnection();

	return BVS::Status::OK;
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

