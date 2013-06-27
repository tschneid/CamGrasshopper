#ifndef CAMGRASSHOPPER_H
#define CAMGRASSHOPPER_H

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "bvs/module.h"
#include "grasshopper.h"


class camGrasshopper : public BVS::Module
{
	public:

		camGrasshopper(BVS::ModuleInfo info, const BVS::Info& bvs);
		~camGrasshopper();
		BVS::Status execute();
		BVS::Status debugDisplay();

	private:
		const BVS::ModuleInfo info;
		BVS::Logger logger;
		const BVS::Info& bvs;

		void triggerCameras();
		void startTriggerThread();

		std::vector<BVS::Connector<cv::Mat>* > outputs;
		std::vector<int> camOrder;

		camGrasshopper(const camGrasshopper&) = delete; /**< -Weffc++ */
		camGrasshopper& operator=(const camGrasshopper&) = delete; /**< -Weffc++ */

		Grasshopper g;
		unsigned int numCameras;

		std::vector<int> resolution;
		std::string encoding;
		float framerate;

		int masterCam; /**< Index for master cam. Slave cams will get image properties from master. */
		int shutter; /**< Define shutter speed for higher frame rate. */


		bool triggerThread;
		bool triggerRunning;
		bool triggerExit;
		std::mutex mutex;
		std::unique_lock<std::mutex> masterLock;
		std::condition_variable triggerCond;
		std::thread trigger;
};

/** This calls a macro to create needed module utilities. */
BVS_MODULE_UTILITIES(camGrasshopper)

#endif //CAMGRASSHOPPER_H

