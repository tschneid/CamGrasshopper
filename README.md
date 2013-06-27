CamGrasshopper ( + BVS Module )
===============================

Simple wrapper for FlyCapture2 to control PointGrey Grasshopper Cameras.

----
## Dependencies
* [PointGrey FlyCapture2 SDK](http://www.ptgrey.com/support/downloads/)
* [OpenCV](http://opencv.org/)
* optional: [OpenCL](http://www.khronos.org/opencl/)

----
## Features
The Grasshopper class contains

* Triggering many cameras by Software- or Hardwaretrigger for simultaneous capturing
* OpenCL YUV422 to RGB/BGR conversion
* Distribution of camera properties from one camera to another (e.g., shutter, gain, ...) 
* Printing out camera information
* ...

----
## Usage
For an example program using the Grasshopper class, have a look at `main()` in `grasshopper.cc`.

----
## BVS Module
You can additionally run the CamGrasshopper module inside the [BVS framework](https://github.com/nilsonholger/bvs).
For more information on how to use the module with BVS, see [BVS/BVS Module List](https://github.com/nilsonholger/bvs#bvs-module-list).

