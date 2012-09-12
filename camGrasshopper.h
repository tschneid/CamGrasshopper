#ifndef CAMGRASSHOPPER_H
#define CAMGRASSHOPPER_H

#include<condition_variable>
#include<mutex>
#include<thread>

#include "bvs/module.h"
#include "grasshopper.h"



/** This is the camGrasshopper class.
 * Please add sufficient documentation to enable others to use it.
 * Include information about:
 * - Dependencies
 * - Inputs
 * - Outputs
 * - Configuration Options
 */
class camGrasshopper : public BVS::Module
{
	public:
		/** Your module constructor.
		 * Please do not change the signature, as it will be called by the
		 * framework.
		 * You can use the constructor/destructor pair to create/destroy your data.
		 * @param[in] id Your modules unique identifier, will be set by framework.
		 * @param[in] bvs Reference to framework info for e.g. config option retrieval.
		 */
		camGrasshopper(const std::string id, const BVS::Info& bvs);

		/** Your module destructor. */
		~camGrasshopper();

		/** Execute function doing all the work.
		 * This function is executed exactly once and only once upon each started
		 * round/step of the framework. It is supposed to contain the actual work
		 * of your module.
		 */
		BVS::Status execute();

		/** UNUSED
		 * @return Module's status.
		 */
		BVS::Status debugDisplay();

	private:
		const std::string id; /**< Your unique module id, set by framework. */

		/** Your logger instance.
		 * @see Logger
		 */
		BVS::Logger logger;

		/** Your Info recerence;
		 * @see Info
		 */
		const BVS::Info& bvs;

		/** Example Connector used to retrieve/send data from/to other modules.
		 * @see Connector
		 */

		void triggerCameras();

		std::vector<BVS::Connector<cv::Mat>* > outputs;

		camGrasshopper(const camGrasshopper&) = delete; /**< -Weffc++ */
		camGrasshopper& operator=(const camGrasshopper&) = delete; /**< -Weffc++ */

		BVS::Config config; /**< Private config object. */

		Grasshopper g;
		unsigned int numCameras;

		int masterCam; /**< Index for master cam. Slave cams will get image properties from master. */
		int shutterSpeed; /**< Define shutter speed for higher frame rate. */


		bool triggerRunning;
		bool triggerExit;
		std::mutex masterMutex;
		std::unique_lock<std::mutex> masterLock;
		std::mutex triggerMutex;
		std::unique_lock<std::mutex> triggerLock;
		std::condition_variable triggerCond;
		std::thread trigger;
};

#endif //CAMGRASSHOPPER_H

