#include "grasshopper.h"
#include "FlyCapture2.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "yuv422toRgb.h" // defines const char clProgramCode[]

using namespace FlyCapture2;

Grasshopper::Grasshopper(int triggerSwitch, bool BGRtoRGB)
: // cameras
  numCameras(0),
  // source pin for hardware trigger
  GPIO_TRIGGER_SOURCE_PIN(0),
  TRIGGER_MODE_NUMBER(14),
  BGRtoRGB(BGRtoRGB),
  width(0),
  height(0),
  encoding(""),
  framerate(0),
  manualProp(),
  error(),
  busMgr(),
  ppCameras(nullptr),
  images(nullptr),
  // information embedded in each image
  embedTimestamp(true),
  embedGain(false),
  embedShutter(false),
  embedBrightness(false),
  embedExposure(false),
  embedWhiteBalance(false),
  embedFrameCounter(false),
  embedStrobePattern(false),
  embedGPIOPinState(false),
  embedROIPosition(false),
  // timestamp
  old_ts(-1),
  fps(-1),
  triggerSwitch(triggerSwitch)
#ifdef _WITH_OPENCL
  ,useGPU(true), clContext(), clCommandQueue(), clDevice(), clProgram(), clKernel(), dYuv(), dRgb()
#endif
{
    
}


bool Grasshopper::initCameras(const int width, const int height, const std::string& encoding, const float& framerate)
{
    return initCameras(getVideoMode(width,height,encoding), getFrameRate(framerate));
}


bool Grasshopper::initCameras(VideoMode videoMode, FrameRate frameRate)
{
    getCameraParameters(videoMode, frameRate, width, height, encoding, framerate);
#ifdef _WITH_OPENCL
    if (!initializeOpenCL())
    {
        std::cout << "Failed to initialize OpenCL, falling back to CPU implementation\n";
        useGPU = false;
    }
#endif

    error = busMgr.GetNumOfCameras(&numCameras);
    if (error != PGRERROR_OK)
    {
        printError( error );
        return false;
    }
    //printf( "Number of cameras detected: %u\n\n", numCameras );
    if ( numCameras < 1 )
    {
        printf( "No cameras detected!\n" );
        return false;
    }

    bool errorState = false; // indicate error and return false

    ppCameras = new Camera*[numCameras];
    images = new Image[numCameras];

    #pragma omp parallel for
    for (unsigned int i = 0; i < numCameras; ++i)
    {
        ppCameras[i] = new Camera();

        PGRGuid guid;
        error = busMgr.GetCameraFromIndex( i, &guid );
        if (error != PGRERROR_OK)
        {
            printError( error );
            errorState = true; 
        }

        if (!errorState)
        {

        }
        // Connect to a camera
        error = ppCameras[i]->Connect( &guid );
        if (error != PGRERROR_OK)
        {
            printError( error );
            errorState = true;
        }

        // Get the camera information
        CameraInfo camInfo;
        error = ppCameras[i]->GetCameraInfo( &camInfo );
        if (error != PGRERROR_OK)
        {
            printError( error );
            errorState = true;
        }
        //std::cout << i <<":\n";
        // printCamInfo(&camInfo); 

        // Set all cameras to a specific mode and frame rate so they
        // can be synchronized.
        error = ppCameras[i]->SetVideoModeAndFrameRate( videoMode, frameRate );
        if (error != PGRERROR_OK)
        {
            printError( error );
            printf( 
                // Video Mode not supported
                "Error starting cameras. \n"
                "The given video mode is supported by the camera. \n");
            errorState = true;
        }

        // Embed information in the first few pixels of the image.
        EmbeddedImageInfo embeddedInfo;
        error = ppCameras[i]->GetEmbeddedImageInfo( &embeddedInfo );
        if ( error != PGRERROR_OK )
        {
            printError( error );
            errorState = true;
        }
        if (embedTimestamp)     embeddedInfo.timestamp.onOff = true;
        if (embedGain)          embeddedInfo.gain.onOff = true;
        if (embedShutter)       embeddedInfo.shutter.onOff = true;
        if (embedBrightness)    embeddedInfo.brightness.onOff = true;
        if (embedExposure)      embeddedInfo.exposure.onOff = true;
        if (embedWhiteBalance)  embeddedInfo.whiteBalance.onOff = true;
        if (embedFrameCounter)  embeddedInfo.frameCounter.onOff = true;
        if (embedStrobePattern) embeddedInfo.strobePattern.onOff = true;
        if (embedGPIOPinState)  embeddedInfo.GPIOPinState.onOff = true;
        if (embedROIPosition)   embeddedInfo.ROIPosition.onOff = true;

        ppCameras[i]->SetEmbeddedImageInfo( &embeddedInfo );
    }

    if (errorState)
        return false;

    // test if propertiers can be written manually
    testPropertiesForManualMode();


    if (triggerSwitch==FIREWIRE_TRIGGER)
    {
        error = Camera::StartSyncCapture( numCameras, (const Camera**)ppCameras );
        if (error != PGRERROR_OK)
        {
            printError( error );
            printf( 
                    // Cameras could not be synchronized over the Firewire bus
                    "Error starting cameras. \n"
                    "Are the cameras on the same bus? (Not dual-bus!). \n");
            return false;
        }
    }
    else
    if (triggerSwitch==SOFTWARE_TRIGGER || triggerSwitch==HARDWARE_TRIGGER)
    {
        const unsigned int millisecondsToSleep = 100;
        unsigned int regVal = 0;

#pragma omp parallel for
        for ( unsigned int i = 0; i < numCameras; ++i )
        {
            // Power on the cameras
            const unsigned int k_cameraPower = 0x610;
            const unsigned int k_powerVal = 0x80000000;
            error  = ppCameras[i]->WriteRegister( k_cameraPower, k_powerVal );
            if (error != PGRERROR_OK)
            {
                printError( error );
                errorState = true;
            }


            // Wait for cameras to complete power-up
            do 
            {
                usleep(millisecondsToSleep * 1000);
                error = ppCameras[i]->ReadRegister(k_cameraPower, &regVal);
                if (error != PGRERROR_OK)
                {
                    printError( error );
                    errorState = true;
                }
            } while ((regVal & k_powerVal) == 0);


            if (triggerSwitch==HARDWARE_TRIGGER)
            {
                // Check for external trigger support
                TriggerModeInfo triggerModeInfo;
                error = ppCameras[i]->GetTriggerModeInfo( &triggerModeInfo );
                if (error != PGRERROR_OK)
                {
                    printError( error );
                    errorState = true;
                }

                if ( triggerModeInfo.present != true )
                {
                    printf( "Camera does not support external trigger!\n" );
                    errorState = true;
                }
            }

            // Get current trigger settings
            TriggerMode triggerMode;
            error = ppCameras[i]->GetTriggerMode( &triggerMode );
            if (error != PGRERROR_OK)
            {
                printError( error );
                errorState = true;
            }

            // Set camera to trigger mode 0
            // Trigger_Mode_0 (“Standard External Trigger Mode”)
            // Trigger_Mode_0 is best described as the standard external trigger mode. When the camera is put
            // into Trigger_Mode_0, the camera starts integration of the incoming light from external trigger input
            // falling/rising edge. The SHUTTER register describes integration time. No parameter is required. The
            // camera can be triggered in this mode using the GPIO pins as external trigger or the SOFTWARE_
            // TRIGGER (62Ch) register.

            // It is not possible to trigger the camera the full frame rate using Mode_0;
            // however, this is possible using Trigger_Mode_14.


            triggerMode.onOff = true;
            triggerMode.mode = TRIGGER_MODE_NUMBER;
            triggerMode.parameter = 0;
            // A source of 7 means software trigger
            if (triggerSwitch==SOFTWARE_TRIGGER) triggerMode.source = 7;
            // Triggering the camera externally using specified source pin.
            if (triggerSwitch==HARDWARE_TRIGGER) triggerMode.source = GPIO_TRIGGER_SOURCE_PIN;

            error = ppCameras[i]->SetTriggerMode( &triggerMode );
            if (error != PGRERROR_OK)
            {
                printError( error );
                errorState = true;
            }

            // Poll to ensure camera is ready
            bool retVal = PollForTriggerReady( ppCameras[i] );
            if( !retVal )
            {
                printf("\nError polling for trigger ready!\n");
                errorState = true;
            }

            // Get the camera configuration
            FC2Config config;
            error = ppCameras[i]->GetConfiguration( &config );
            if (error != PGRERROR_OK)
            {
                printError( error );
                errorState = true;
            } 
            // Set the grab timeout to 5 seconds
            // grabTimeout = Time in milliseconds that RetrieveBuffer()
            // and WaitForBufferEvent() will wait for an image before
            // timing out and returning. 
            config.grabTimeout = 5000;
            // config.grabMode = BUFFER_FRAMES; // tried for a higher frame rate... not working
            // Set the camera configuration
            error = ppCameras[i]->SetConfiguration( &config );
            if (error != PGRERROR_OK)
            {
                printError( error );
                errorState = true;
            }

            // Cameras are ready, start capturing images
            error = ppCameras[i]->StartCapture();
            if (error != PGRERROR_OK)
            {
                printError( error );
                errorState = true;
            }

            if (triggerSwitch==SOFTWARE_TRIGGER && !CheckSoftwareTriggerPresence( ppCameras[i] ))
            {
                printf( "SOFT_ASYNC_TRIGGER not implemented on this camera! Stopping application\n");
                errorState = true;
            }
        }

        if (errorState)
            return false;
    }
    else // no trigger
    {
        for ( unsigned int i = 0; i < numCameras; ++i )
        {
            error = ppCameras[i]->StartCapture();
            if (error != PGRERROR_OK)
            {
                printError( error );
            }
        }
    }

    return true;
}



bool Grasshopper::stopCameras()
{
    if (triggerSwitch==SOFTWARE_TRIGGER || triggerSwitch==HARDWARE_TRIGGER)
    {
        // Turn trigger mode off.
        for (unsigned int i = 0; i < numCameras; ++i)
        {
            TriggerMode triggerMode;
            error = ppCameras[i]->GetTriggerMode( &triggerMode );
            if (error != PGRERROR_OK)
            {
                printError( error );
                //exit(-1);
            }
            triggerMode.onOff = false;

            error = ppCameras[i]->SetTriggerMode( &triggerMode );
            if (error != PGRERROR_OK)
            {
                printError( error );
                //exit(-1);
            }
        }
    }
    for ( unsigned int i = 0; i < numCameras; i++ )
    {
        ppCameras[i]->StopCapture();
        ppCameras[i]->Disconnect();
        delete ppCameras[i];
    }
    delete [] ppCameras;
    delete [] images;

#ifdef _WITH_OPENCL
    cleanupOpenCL();
#endif

    return true;
}



bool Grasshopper::getNextFrame()
{
    if (triggerSwitch==SOFTWARE_TRIGGER)
    {
        // Fire software trigger
#ifdef _WITH_TIMER
        OKAPI_TIMER_START("grasshopper: FireSoftwareTrigger()");
#endif
        bool retVal = FireSoftwareTrigger(ppCameras);
#ifdef _WITH_TIMER
        OKAPI_TIMER_STOP("grasshopper: FireSoftwareTrigger()");
#endif
        /* for (unsigned int i = 0; i < numCameras; ++i)
           {
           retVal &= FireSoftwareTrigger( ppCameras[i] );
           }*/
        if ( !retVal )
        {
            printf("Error firing software trigger!\n");
            return false;
        }
    }
#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: RetrieveBuffer() of all cameras");
#endif
    for (unsigned int i = 0; i < numCameras; ++i)
    {
#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: RetrieveBuffer() of one cameras");
#endif
        // Write the frame in images
        error = ppCameras[i]->RetrieveBuffer( &images[i] );
        if (error != PGRERROR_OK)
        {
            printError( error );
        }
#ifdef _WITH_TIMER
    OKAPI_TIMER_STOP("grasshopper: RetrieveBuffer() of one cameras");
#endif

    }
#ifdef _WITH_TIMER
    OKAPI_TIMER_STOP("grasshopper: RetrieveBuffer() of all cameras");
#endif

    return true;
}



Image Grasshopper::getFlyCapImage(const int i)
{
    return images[i];
}


cv::Mat Grasshopper::getImage(const int i)
{
#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: getImage()");
#endif

    unsigned int rows, cols;
    rows = images[i].GetRows();
    cols = images[i].GetCols();
    unsigned int bpp = images[i].GetBitsPerPixel();
    unsigned int channels = bpp/8;
  
    cv::Mat img(rows,cols,CV_8UC(channels));

    // Set the pointer of the cv::Mat data to the image data
    img.data = images[i].GetData();

    // The image is actually BGR and we have to
    // change B and R channel
    if (channels == 3 && BGRtoRGB)
    {
        cv::cvtColor(img,img,CV_BGR2RGB);
#ifdef _WITH_TIMER
        OKAPI_TIMER_STOP("grasshopper: getImage()");
#endif
        return img;
    }

    // The image is probably YUV422 and we have
    // to convert it to RGB.
    // (Or it is Y16, then this should lead to some
    // interesting results...)
    if (channels == 2) 
    {
        cv::Mat imgRGB(rows, cols, CV_8UC3);
#ifdef _WITH_OPENCL
        if (useGPU) yuv422toRGB_gpu(img, imgRGB, BGRtoRGB);
        else yuv422toRGB(img, imgRGB, BGRtoRGB);
#else
        yuv422toRGB(img, imgRGB, BGRtoRGB);
#endif
#ifdef _WITH_TIMER
        OKAPI_TIMER_STOP("grasshopper: getImage()");
#endif
        return imgRGB;
    }
    
#ifdef _WITH_TIMER
        OKAPI_TIMER_STOP("grasshopper: getImage()");
#endif
    // The image is Y8 (grayscale)
    return img;
}



bool Grasshopper::distributeCamProperties(const unsigned int master)
{
#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: distributeCamProperties()");
#endif
    
    for (std::map<PropertyType,bool>::iterator it = manualProp.begin(); it != manualProp.end(); ++it)
    {
        if ((*it).second) // flag if property can be set manually
        {
            // get properties from master camera
            Property masterProp;
            masterProp.type = (*it).first;
            error = ppCameras[master]->GetProperty(&masterProp);
            if (error != PGRERROR_OK)
            {
                printError(error);
                return false;
            }

            for (unsigned int i = 0; i < numCameras; ++i)
            {
                if (i != master)
                {

                    // set properties for all slave cameras
                    Property slaveProp;
                    slaveProp.type = masterProp.type;
                    slaveProp.onOff = true;
                    slaveProp.autoManualMode = false;

                    switch (slaveProp.type)
                    {
                        case WHITE_BALANCE:
                            slaveProp.valueA = masterProp.valueA;
                            slaveProp.valueB = masterProp.valueB;
                            break;
                        case SHARPNESS:
                            slaveProp.valueA = masterProp.valueA;
                            break;
                        default:
                            slaveProp.absControl = true;
                            slaveProp.absValue = masterProp.absValue;
                            break;
                    }
                    
                    error = ppCameras[i]->SetProperty(&slaveProp);
                    if (error != PGRERROR_OK)
                    {
                        printError(error);
                        return false;
                    }
                }
            }
        }
    }

#ifdef _WITH_TIMER
    OKAPI_TIMER_STOP("grasshopper: distributeCamProperties()");
#endif

    return true;
}


bool Grasshopper::restoreDefaultProperties(const int i)
{
#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: restoreDefaultProperties()");
#endif
    if (i < 0)
    {
        // restore defaults of each connected cam
        for (unsigned int cam = 0; cam < numCameras; ++cam)
        {
            error = ppCameras[cam]->RestoreFromMemoryChannel(0);
            if (error != PGRERROR_OK)
            {
                printError(error);
            }
        }
    }
    else
    {
        if ((unsigned int)i < numCameras)
        {
            error = ppCameras[i]->RestoreFromMemoryChannel(0);
            if (error != PGRERROR_OK)
            {
                printError(error);
            }
        }
        else
        {
            std::cout << "restoreDefaultProperties(): Wrong camera index!";
            return false;
        }
    }
#ifdef _WITH_TIMER
    OKAPI_TIMER_STOP("grasshopper: restoreDefaultProperties()");
#endif
    return true;
}


bool Grasshopper::testPropertiesForManualMode()
{
    bool output = false; // cout some information

    // offset, controls level of black in an image
    manualProp.insert(std::pair<PropertyType, bool>(BRIGHTNESS, false)); // no auto mode supported
    // allows the camera to automatically control shutter and/or gain
    // (only if SHUTTER and GAIN autoManual--Bits are set)
    manualProp.insert(std::pair<PropertyType, bool>(AUTO_EXPOSURE, false));
    // sharpness adjustment --> average upon a 3x3 block of pixels, only
    // applied to the green component of the Bayer tiled pattern
    manualProp.insert(std::pair<PropertyType, bool>(SHARPNESS, false));
    // color correction to account for different lighting conditions, by modifying
    // the relative gain of R, G and B.
    manualProp.insert(std::pair<PropertyType, bool>(WHITE_BALANCE, false)); // not present
    // hue adjustment defines the color phase of images (in degrees)
    manualProp.insert(std::pair<PropertyType, bool>(HUE, false)); // no manual mode supported
    // saturation adjustment of color values (in %)
    manualProp.insert(std::pair<PropertyType, bool>(SATURATION, false)); // no manual mode supported
    // non--linear mapping of raw image intensity from the image sensor
    manualProp.insert(std::pair<PropertyType, bool>(GAMMA, false)); // no auto mode supported
    // time of exposure
    manualProp.insert(std::pair<PropertyType, bool>(SHUTTER, false));
    // the sensitiviy of the image sensor
    manualProp.insert(std::pair<PropertyType, bool>(GAIN, false));

    // Iterate through all properties and save if they should be
    // considered later on (i.e. if they are present and if they
    // can be set to manual mode)
    // @TODO This assumes, that the properties are the same for each camera!
    //std::cout << "*** CAMERA PROPERTIES ***\n";

    for (std::map<PropertyType,bool>::iterator it = manualProp.begin(); it != manualProp.end(); ++it)
    {
        PropertyInfo propInfo;
        propInfo.type = (*it).first;
        error = ppCameras[0]->GetPropertyInfo(&propInfo);
        if (error != PGRERROR_OK)
        {
            printError( error );
            return false;
        }

        // do not output rest
        // continue;

        // Only care about a property which is present and
        // can be set to auto mode.
        if (output) std::cout << toString((*it).first);
        if (propInfo.present)
        {
            if (propInfo.autoSupported)
            {
                Property prop;
                prop.type = (*it).first;
                if (propInfo.manualSupported)
                {
                    // This property has to be considered later on.
                    (*it).second = true;
                    if (output) std::cout << " [will be controlled by master camera]\n";
                }
                else
                {
                    if (output) std::cout << " [no manual mode supported]\n";
                }
            }
            else
            {
                if (output) std::cout << " [no auto mode supported]\n";
            }
        }
        else
        {
            if (output) std::cout << " [not present on this camera type]\n";
        }
    }

    return true;
}


void Grasshopper::printInfo()
{
    std::string triggerMode;
    std::string triggerExplanation;
    switch (triggerSwitch)
    {
        case 0: triggerMode = "NO TRIGGER";
                triggerExplanation = "the cameras capture images independently and are not synchronized";
                break;
        case 1: triggerMode = "SOFTWARE TRIGGER";
                triggerExplanation = "the cameras are synchronized by writing a camera internal register";
                break;
        case 2: triggerMode = "FIREWIRE TRIGGER";
                triggerExplanation = "the cameras are automatically synchronized by the firewire bus (therefore they have to be on the same bus)";
                break;
        case 3: triggerMode = "HARDWARE TRIGGER";
                triggerExplanation = "the cameras are synchronized by a trigger pulse on GPIO pin " + std::to_string(GPIO_TRIGGER_SOURCE_PIN);
                break;
        default: break;
    }

    std::cout << "*** INFORMATION ***\n"
              << numCameras << " connected cameras found.\n"
              << "Resolution: " << width << "x" << height << ", Encoding: " << encoding << ", Framerate: " << framerate << "\n"
              << "Trigger: " << triggerMode << " (trigger mode " << TRIGGER_MODE_NUMBER << "), i.e., " << triggerExplanation << ".\n"
              << "Properties to distribute: ";
    for (std::map<PropertyType,bool>::iterator it = manualProp.begin(); it != manualProp.end(); ++it)
    {
        if ((*it).second) std::cout << toString((*it).first) << " ";
    }
    std::cout << "\n";
#ifdef _WITH_OPENCL
    std::cout << "The YUV422 to RGB conversion will be calculated on the GPU using OpenCL\n";
#endif
#ifdef _WITH_TIMER
    std::cout << "Okapi Timer is activated\n";
#endif
}

void Grasshopper::printCamInfo( CameraInfo* pCamInfo )
{
    printf(
        "*** CAMERA INFORMATION ***\n"
        "Serial number - %u\n"
        "Camera model - %s\n"
        "Camera vendor - %s\n"
        "Sensor - %s\n"
        "Resolution - %s\n"
        "Firmware version - %s\n"
        "Firmware build time - %s\n\n",
        pCamInfo->serialNumber,
        pCamInfo->modelName,
        pCamInfo->vendorName,
        pCamInfo->sensorInfo,
        pCamInfo->sensorResolution,
        pCamInfo->firmwareVersion,
        pCamInfo->firmwareBuildTime );
}


void Grasshopper::printVideoModes(const int i)
{
    int numFrameRates = 7;
    std::vector<FrameRate> frameRate;
    frameRate.push_back(FRAMERATE_3_75);
    frameRate.push_back(FRAMERATE_7_5);
    frameRate.push_back(FRAMERATE_15);
    frameRate.push_back(FRAMERATE_30);
    frameRate.push_back(FRAMERATE_60);
    frameRate.push_back(FRAMERATE_120);
    frameRate.push_back(FRAMERATE_240);

    int numVideoModes = 24;
    std::vector<VideoMode> videoMode;
    videoMode.push_back(VIDEOMODE_FORMAT7);
    videoMode.push_back(VIDEOMODE_160x120YUV444);
    videoMode.push_back(VIDEOMODE_320x240YUV422);
    videoMode.push_back(VIDEOMODE_640x480YUV422);
    videoMode.push_back(VIDEOMODE_800x600YUV422);
    videoMode.push_back(VIDEOMODE_1024x768YUV422);
    videoMode.push_back(VIDEOMODE_1280x960YUV422);
    videoMode.push_back(VIDEOMODE_1600x1200YUV422);
    videoMode.push_back(VIDEOMODE_640x480YUV411);
    videoMode.push_back(VIDEOMODE_640x480RGB);
    videoMode.push_back(VIDEOMODE_800x600RGB);
    videoMode.push_back(VIDEOMODE_1024x768RGB);
    videoMode.push_back(VIDEOMODE_1280x960RGB);
    videoMode.push_back(VIDEOMODE_1600x1200RGB);
    videoMode.push_back(VIDEOMODE_640x480Y8);
    videoMode.push_back(VIDEOMODE_800x600Y8);
    videoMode.push_back(VIDEOMODE_1024x768Y8);
    videoMode.push_back(VIDEOMODE_1280x960Y8);
    videoMode.push_back(VIDEOMODE_1600x1200Y8);
    videoMode.push_back(VIDEOMODE_640x480Y16);
    videoMode.push_back(VIDEOMODE_800x600Y16);
    videoMode.push_back(VIDEOMODE_1024x768Y16);
    videoMode.push_back(VIDEOMODE_1280x960Y16);
    videoMode.push_back(VIDEOMODE_1600x1200Y16);

    std::cout << "\n*** POSSIBLE VIDEO MODES ***\n";

    for (int iterVM = 0; iterVM < numVideoModes; ++iterVM)
    {
        std::stringstream ss;
        for (int iterFR = 0; iterFR < numFrameRates; ++iterFR)
        {
            bool supported;
            ppCameras[i]->GetVideoModeAndFrameRateInfo(videoMode[iterVM], frameRate[iterFR], &supported);
            if (supported) ss << toString(frameRate[iterFR]) << " ";
        }
        if (ss.str() != "")
            std::cout << toString(videoMode[iterVM]) << ": " << ss.str() << "\n";
    }

    std::cout << "\n";
}


TimeStamp Grasshopper::getTimestamp(const int i)
{
    return images[i].GetTimeStamp();
}


unsigned int Grasshopper::getCycleCount(const int i) const
{
    return images[i].GetTimeStamp().cycleCount;
}


double Grasshopper::tickFPS()
{
    double new_ts;
    struct timeval now;
    gettimeofday(&now, NULL);
    new_ts = ((double)(now.tv_usec)) / 1000000.0; // in seconds
    if (old_ts < 0)
        old_ts = new_ts - 0.1;
    double new_fps = 1 / (new_ts - old_ts);
    if (fps < 0)
        fps = new_fps;
    else
        fps = 0.95 * fps + 0.05 * new_fps;
    old_ts = new_ts;
    return fps;
}


std::string Grasshopper::getProcessedFPSString()
{
    char buf[1000];
    sprintf(buf, "%4.1f fps\n", fps);
    std::string fpsString = buf;
    return fpsString;
}


void Grasshopper::printImageMetadata(const int i)
{
    // timestamp, shutter, brightness, exposure, whiteBalance is embedded
    ImageMetadata meta = images[i].GetMetadata();
    
    /*union _absValueConversion
    {
        unsigned long ulVal;
        float fVal;
    } absValueConversion;

    absValueConversion.ulVal = meta.embeddedShutter;
    float f = absValueConversion.fVal;
    float* p_f = reinterpret_cast<float*>(&shutter);

    std::cout << f << " " << *p_f <<"\n";*/

    std::stringstream ss;
    if (embedTimestamp)     ss << "Timestamp: " << meta.embeddedTimeStamp << "\n";
    if (embedGain)          ss << "Gain: " << meta.embeddedGain << "\n";
    if (embedShutter)       ss << "Shutter: " << meta.embeddedShutter << "\n";
    if (embedBrightness)    ss << "Brightness: " << meta.embeddedBrightness << "\n";
    if (embedExposure)      ss << "Exposure: " << meta.embeddedExposure << "\n";
    if (embedWhiteBalance)  ss << "WhiteBalance: " << meta.embeddedWhiteBalance << "\n";
    if (embedFrameCounter)  ss << "FrameCounter: " << meta.embeddedFrameCounter << "\n";
    if (embedStrobePattern) ss << "StrobePattern: " << meta.embeddedStrobePattern << "\n";
    if (embedGPIOPinState)  ss << "GPIOPinState: " << meta.embeddedGPIOPinState << "\n";
    if (embedROIPosition)   ss << "ROIPosition: " << meta.embeddedROIPosition << "\n";
    if (ss.str() != "")
        std::cout << ss.str();
    else
        std::cout << "No information embedded.\n";
}


bool Grasshopper::setShutter(const int milliseconds)
{
    Property shutter;
    shutter.type = SHUTTER;
    shutter.onOff = true;
    shutter.autoManualMode = false; // auto-adjust mode off
    shutter.absControl = true; // write property with absolute value
    shutter.absValue = milliseconds;

    Property gain;
    gain.type = GAIN;
    gain.onOff = true;
    gain.autoManualMode = true;

    // autoManualMode of
    //   SHUTTER is off
    //   GAIN is on
    // The auto exposure algorithm will regulate gain
    // to maintain image brightness.

    // @TODO auto_exposure_range for minimal and maximal shutter time?    

    for (unsigned int i = 0; i < numCameras; ++i)
    {
        error = ppCameras[i]->SetProperty(&shutter);
        if (error != PGRERROR_OK)
        {
            printError( error );
            return false;
        }

        error = ppCameras[i]->SetProperty(&gain);
        if (error != PGRERROR_OK)
        {
            printError( error );
            return false;
        }
    }
    return true;   
}


std::string Grasshopper::getProperty(const PropertyType& propType, const int i)
{
#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: getProperty()");
#endif

    Property prop;
    prop.type = propType;

    PropertyInfo propInfo;
    propInfo.type = propType;

    std::stringstream propString;

    
    ppCameras[i]->GetProperty(&prop);
    ppCameras[i]->GetPropertyInfo(&propInfo);

    if (propInfo.present)
    {
        if (propInfo.absValSupported && propInfo.readOutSupported)
        {

            switch (propType)
            {
                case BRIGHTNESS: propString << "Brightness"; break;
                case AUTO_EXPOSURE: propString << "Auto_Exposure"; break;
                case SHARPNESS: propString << "Sharpness"; break;
                case WHITE_BALANCE: propString << "White_Balance"; break;
                case HUE: propString << "Hue"; break;
                case SATURATION: propString << "Saturation"; break;
                case GAMMA: propString << "Gamma"; break;
                case IRIS: propString << "Iris"; break;
                case FOCUS: propString << "Focus"; break;
                case ZOOM: propString << "Zoom"; break;
                case PAN: propString << "Pan"; break;
                case TILT: propString << "Tilt"; break;
                case SHUTTER: propString << "Shutter"; break;
                case GAIN: propString << "Gain"; break;
                case TRIGGER_MODE: propString << "Trigger_Mode"; break;
                case TRIGGER_DELAY: propString << "Trigger_Delay"; break;
                case FRAME_RATE: propString << "Frame_Rate"; break;
                case TEMPERATURE: propString << "Temperature"; break;
                case UNSPECIFIED_PROPERTY_TYPE: propString << "Unspecified"; break;
                default: break;
            }
            propString << ": " << prop.absValue << " " << propInfo.pUnitAbbr;
        }
        else
        {
            printf("Could not read out value.\n");
            return "";
        }
    }
    else
    {
        printf("The property is not present.\n");
        return "";
    }
    
#ifdef _WITH_TIMER
    OKAPI_TIMER_STOP("grasshopper: getProperty()");
#endif
    return propString.str();
}



int Grasshopper::getCameraSerialNumber(int index)
{
	CameraInfo camInfo;
	error = ppCameras[index]->GetCameraInfo(&camInfo);
	if (error != PGRERROR_OK)
	{
		printError( error );
	}
	return camInfo.serialNumber;
}



#ifdef _STANDALONE
void Grasshopper::resetBus()
{
    // reset all cameras/buses for a clean start
    okapi::Cam1394b::dc1394ext_reset_bus();
}


void Grasshopper::printBusInfo()
{
    // print all connected cameras
    okapi::Cam1394b::printConnectedCams();
    std::cout << "\n";
    
    // print the unique IDs of all connected cameras
    std::vector<okapi::Cam1394b::unique_cam_id_t> uids;
    uids = okapi::Cam1394b::getConnectedCamIDs();
    std::cout << "Unique IDs of connected cameras: ";
    for (size_t i = 0; i < uids.size(); i++)
        std::cout << uids[i] << " ";
    std::cout << "\n";
    
    // get the unique IDs of all connected cameras with the given vendorname/modelname
    std::string vendorname = "*";
    
    std::string modelname = "Grasshopper GRAS-50S5C"; // Grasshopper 2, Model# GS2-FW-14S5M/C-C
    uids = okapi::Cam1394b::getConnectedCamIDs(vendorname,modelname);
    std::cout << "Unique IDs of connected cameras by vendor=" << vendorname << "/model=" << modelname << ": ";
    for (size_t i = 0; i < uids.size(); i++)
        std::cout << uids[i] << " ";
    std::cout << std::endl;
}
#endif


bool Grasshopper::saveImages(const int imgNum)
{
    for (unsigned int cam = 0; cam < numCameras; ++cam)
    {
        char filename[512];
        sprintf(filename, "image-%u_cam-%d.jpg", imgNum, cam);
        Error error = images[cam].Save(filename);
        if (error != PGRERROR_OK)
        {
            printError(error);
            return false;
        }
    }
    return true;
}


bool Grasshopper::setROI(const int x, const int y, const int width, const int height, const unsigned int cam)
{
    return setROI(cv::Rect(x,y,width,height), cam);
}


bool Grasshopper::setROI(const cv::Rect& _roi, const unsigned int cam)
{
#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: setROI()");
#endif

    // VideoMode currVideoMode;
    // FrameRate currFrameRate;
    // Format7ImageSettings currFmt7Settings;
    // unsigned int currPacketSize = 0;

    // error = ppCameras[cam]->GetVideoModeAndFrameRate( &currVideoMode, &currFrameRate );        
    // if ( error != PGRERROR_OK )
    // {
    //     printError(error);
    // }

    // if ( currVideoMode == VIDEOMODE_FORMAT7 )
    // {
    //     // Get the current Format 7 settings
    //     float percentage; // Don't need to keep this
    //     error = ppCameras[cam]->GetFormat7Configuration( &currFmt7Settings, &currPacketSize, &percentage );
    //     if ( error != PGRERROR_OK )
    //     {
    //         printError(error);
    //         return false;
    //     }
    // }

    const Mode k_fmt7Mode = MODE_0;
    const PixelFormat k_fmt7PixFmt = PIXEL_FORMAT_MONO8;
    
    // Query for available Format 7 modes
    Format7Info fmt7Info;
    bool supported;
    fmt7Info.mode = k_fmt7Mode;
    error = ppCameras[cam]->GetFormat7Info( &fmt7Info, &supported );
    if (error != PGRERROR_OK)
    {
        printError( error );
        return false;
    }

    // printf(
    //     "Max image pixels: (%u, %u)\n"   // 2448, 2048
    //     "Image Unit size: (%u, %u)\n"    // 8, 2
    //     "Offset Unit size: (%u, %u)\n"   // 2, 2
    //     "Pixel format bitfield: 0x%08x\n",
    //     fmt7Info.maxWidth,
    //     fmt7Info.maxHeight,
    //     fmt7Info.imageHStepSize,
    //     fmt7Info.imageVStepSize,
    //     fmt7Info.offsetHStepSize,
    //     fmt7Info.offsetVStepSize,
    //     fmt7Info.pixelFormatBitField );

    // if ( (k_fmt7PixFmt & fmt7Info.pixelFormatBitField) == 0 )
    // {
    //     // Pixel format not supported!
    //     printf("Pixel format is not supported\n");
    //     return false;
    // }

    // change roi values to the nearest allowed values
    // (it will always be bigger than the original one)
    cv::Rect roi = _roi;
    if (roi.x % fmt7Info.offsetHStepSize != 0)
        roi.x = int(roi.x / fmt7Info.offsetHStepSize) * fmt7Info.offsetHStepSize;
    if (roi.x < 0) roi.x = 0;
    if (roi.y % fmt7Info.offsetVStepSize != 0)
        roi.y = int(roi.y / fmt7Info.offsetVStepSize) * fmt7Info.offsetVStepSize;
    if (roi.y < 0) roi.y = 0;
    if (roi.width + roi.x > (int)fmt7Info.maxWidth) roi.width = fmt7Info.maxWidth - roi.x;
    if (roi.width % fmt7Info.imageHStepSize != 0)
        roi.width = int(roi.width / fmt7Info.imageHStepSize) * fmt7Info.imageHStepSize;
    if (roi.height + roi.y > (int)fmt7Info.maxHeight) roi.height = fmt7Info.maxHeight - roi.y;
    if (roi.height % fmt7Info.imageVStepSize != 0)
        roi.height = int(roi.height / fmt7Info.imageVStepSize) * fmt7Info.imageVStepSize;

    Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = k_fmt7Mode;
    fmt7ImageSettings.offsetX = roi.x;
    fmt7ImageSettings.offsetY = roi.y;
    fmt7ImageSettings.width = roi.width;//fmt7Info.maxWidth;
    fmt7ImageSettings.height = roi.height;//fmt7Info.maxHeight;
    fmt7ImageSettings.pixelFormat = k_fmt7PixFmt;

    bool valid;
    Format7PacketInfo fmt7PacketInfo;

    // Validate the settings to make sure that they are valid
    error = ppCameras[cam]->ValidateFormat7Settings(&fmt7ImageSettings, &valid, &fmt7PacketInfo);
    if (error != PGRERROR_OK)
    {
        printError( error );
        return false;
    }

    if ( !valid )
    {
        // Settings are not valid
        printf("Format7 settings are not valid\n");
        return false;
    }

#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: setROI(): stop capture");
#endif

    ppCameras[cam]->StopCapture(); // @TODO: This takes really long! (unlike in flycap GUI)

#ifdef _WITH_TIMER
    OKAPI_TIMER_STOP("grasshopper: setROI(): stop capture");
#endif

    //std::cout << "==== bytesPerPacket ====\n"
    //          << "recommended: " << fmt7PacketInfo.recommendedBytesPerPacket << "\n"
    //          << "max: " << fmt7PacketInfo.maxBytesPerPacket << "\n";
    
    // Set the settings to the camera
    error = ppCameras[cam]->SetFormat7Configuration(&fmt7ImageSettings, fmt7PacketInfo.recommendedBytesPerPacket);
    if (error != PGRERROR_OK)
    {
        printError( error );
        return false;
    }

    ppCameras[cam]->StartCapture();



#ifdef _WITH_TIMER
    OKAPI_TIMER_STOP("grasshopper: setROI()");
#endif
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Internal Helpers
///////////////////////////////////////////////////////////////////////////////

bool Grasshopper::PollForTriggerReady( Camera* pCam )
{
    const unsigned int k_softwareTrigger = 0x62C;
    Error error;
    unsigned int regVal = 0;

    do 
    {
        error = pCam->ReadRegister( k_softwareTrigger, &regVal );
        if (error != PGRERROR_OK)
        {
            printError( error );
            return false;
        }

    } while ( (regVal >> 31) != 0 );

    return true;
}


bool Grasshopper::CheckSoftwareTriggerPresence( Camera* pCam )
{
    const unsigned int k_triggerInq = 0x530;

    Error error;
    unsigned int regVal = 0;

    error = pCam->ReadRegister( k_triggerInq, &regVal );

    if (error != PGRERROR_OK)
    {
        printError( error );
        return false;
    }

    if( ( regVal & 0x10000 ) != 0x10000 )
    {
        return false;
    }

    return true;
}


bool Grasshopper::FireSoftwareTrigger( Camera** ppCam )
{
    const unsigned int k_softwareTrigger = 0x62C;
    const unsigned int k_fireVal = 0x80000000;
    Error error;

    for (unsigned int i = 0; i < numCameras; ++i)
    {
        error = ppCam[i]->WriteRegister( k_softwareTrigger, k_fireVal );
        if (error != PGRERROR_OK)
        {
            printError( error );
            return false;
        }
    }
    
    return true;
}


std::string Grasshopper::toString(const FrameRate& fps)
{
    switch (fps)
    {
        case FRAMERATE_3_75: return "3.75";
        case FRAMERATE_7_5: return "7.5";
        case FRAMERATE_15: return "15";
        case FRAMERATE_30: return "30";
        case FRAMERATE_60: return "60";
        case FRAMERATE_120: return "120";
        case FRAMERATE_240: return "240";
        default: break;
    }
    return "";
}


std::string Grasshopper::toString(const VideoMode& vm)
{
    switch (vm)
    {
        case VIDEOMODE_FORMAT7: return "FORMAT7";
        case VIDEOMODE_160x120YUV444: return "160x120 YUV444";
        case VIDEOMODE_320x240YUV422: return "320x240 YUV422";
        case VIDEOMODE_640x480YUV422: return "640x480 YUV422";
        case VIDEOMODE_800x600YUV422: return "800x600 YUV422";
        case VIDEOMODE_1024x768YUV422: return "1024x768 YUV422";
        case VIDEOMODE_1280x960YUV422: return "1280x960 YUV422";
        case VIDEOMODE_1600x1200YUV422: return "1600x1200 YUV422";
        case VIDEOMODE_640x480YUV411: return "640x480 YUV411";
        case VIDEOMODE_640x480RGB: return "640x480 RGB";
        case VIDEOMODE_800x600RGB: return "800x600 RGB";
        case VIDEOMODE_1024x768RGB: return "1024x768 RGB";
        case VIDEOMODE_1280x960RGB: return "1280x960 RGB";
        case VIDEOMODE_1600x1200RGB: return "1600x1200 RGB";
        case VIDEOMODE_640x480Y8: return "640x480 Y8";
        case VIDEOMODE_800x600Y8: return "800x600 Y8";
        case VIDEOMODE_1024x768Y8: return "1024x768 Y8";
        case VIDEOMODE_1280x960Y8: return "1280x960 Y8";
        case VIDEOMODE_1600x1200Y8: return "1600x1200 Y8";
        case VIDEOMODE_640x480Y16: return "640x480 Y16";
        case VIDEOMODE_800x600Y16: return "800x600 Y16";
        case VIDEOMODE_1024x768Y16: return "1024x768 Y16";
        case VIDEOMODE_1280x960Y16: return "1280x960 Y16";
        case VIDEOMODE_1600x1200Y16: return "1600x1200 Y16";
        default: break;
    }
    return "";
}

std::string Grasshopper::toString(const PropertyType& propType)
{
    switch (propType)
    {
        case BRIGHTNESS: return "BRIGHTNESS";
        case AUTO_EXPOSURE: return "AUTO_EXPOSURE";
        case SHARPNESS: return "SHARPNESS";
        case WHITE_BALANCE: return "WHITE_BALANCE";
        case HUE: return "HUE";
        case SATURATION: return "SATURATION";
        case GAMMA: return "GAMMA";
        case SHUTTER: return "SHUTTER";
        case GAIN: return "GAIN";
        default: break;
    }
    return "";
}

VideoMode Grasshopper::getVideoMode(const int width, const int height, const std::string& _encoding)
{
    // YUV422, Y8, RGB Y16YUV 444 FORMAT7 YUV411
    std::string encoding = _encoding;
    std::transform(encoding.begin(), encoding.end(), encoding.begin(), ::toupper);
    if      (encoding.compare("FORMAT7") == 0)                                return VIDEOMODE_FORMAT7;
    else if (encoding.compare("YUV444") == 0 && width==160 && height==120)    return VIDEOMODE_160x120YUV444;
    else if (encoding.compare("YUV444") == 0 && width==640 && height==480)    return VIDEOMODE_640x480YUV411;
    else if (encoding.compare("YUV422") == 0 && width==320 && height==240)    return VIDEOMODE_320x240YUV422;
    else if (encoding.compare("YUV422") == 0 && width==640 && height==480)    return VIDEOMODE_640x480YUV422;
    else if (encoding.compare("YUV422") == 0 && width==800 && height==600)    return VIDEOMODE_800x600YUV422;
    else if (encoding.compare("YUV422") == 0 && width==1024 && height==768)   return VIDEOMODE_1024x768YUV422;
    else if (encoding.compare("YUV422") == 0 && width==1280 && height==960)   return VIDEOMODE_1280x960YUV422;
    else if (encoding.compare("YUV422") == 0 && width==1600 && height==1200)  return VIDEOMODE_1600x1200YUV422;
    else if (encoding.compare("RGB") == 0    && width==640 && height==480)    return VIDEOMODE_640x480RGB;
    else if (encoding.compare("RGB") == 0    && width==800 && height==600)    return VIDEOMODE_800x600RGB;
    else if (encoding.compare("RGB") == 0    && width==1024 && height==768)   return VIDEOMODE_1024x768RGB;
    else if (encoding.compare("RGB") == 0    && width==1280 && height==960)   return VIDEOMODE_1280x960RGB;
    else if (encoding.compare("RGB") == 0    && width==1600 && height==1200)  return VIDEOMODE_1600x1200RGB;
    else if (encoding.compare("Y8") == 0     && width==640 && height==480)    return VIDEOMODE_640x480Y8;
    else if (encoding.compare("Y8") == 0     && width==800 && height==600)    return VIDEOMODE_800x600Y8;
    else if (encoding.compare("Y8") == 0     && width==1024 && height==768)   return VIDEOMODE_1024x768Y8;
    else if (encoding.compare("Y8") == 0     && width==1280 && height==960)   return VIDEOMODE_1280x960Y8;
    else if (encoding.compare("Y8") == 0     && width==1600 && height==1200)  return VIDEOMODE_1600x1200Y8;
    else if (encoding.compare("Y16") == 0    && width==640 && height==480)    return VIDEOMODE_640x480Y16;
    else if (encoding.compare("Y16") == 0    && width==800 && height==600)    return VIDEOMODE_800x600Y16;
    else if (encoding.compare("Y16") == 0    && width==1024 && height==768)   return VIDEOMODE_1024x768Y16;
    else if (encoding.compare("Y16") == 0    && width==1280 && height==960)   return VIDEOMODE_1280x960Y16;
    else if (encoding.compare("Y16") == 0    && width==1600 && height==1200)  return VIDEOMODE_1600x1200Y16;
    else std::cout << "The specified video mode " << width << "x" << height << encoding << " is not supported.\n";
    return NUM_VIDEOMODES;
}

FrameRate Grasshopper::getFrameRate(const float& fps)
{
    if      (fps == 3.75) return FRAMERATE_3_75;
    else if (fps == 7.5)  return FRAMERATE_7_5;
    else if (fps == 15)   return FRAMERATE_15;
    else if (fps == 30)   return FRAMERATE_30;
    else if (fps == 60)   return FRAMERATE_60;
    else if (fps == 120)  return FRAMERATE_120;
    else if (fps == 240)  return FRAMERATE_240;
    else std::cout << "The specified framerate " << fps << "is not supported.\n";
    return NUM_FRAMERATES;
}

void Grasshopper::getCameraParameters(const VideoMode& vm, const FrameRate& fr, int& width, int& height, std::string& encoding, float& framerate)
{
    switch (vm)
    {
        case VIDEOMODE_FORMAT7:         width = 0; height = 0; encoding = ""; break;
        case VIDEOMODE_160x120YUV444:   width = 160; height = 120; encoding = "yuv444"; break;
        case VIDEOMODE_320x240YUV422:   width = 320; height = 240; encoding = "yuv422"; break;
        case VIDEOMODE_640x480YUV422:   width = 640; height = 480; encoding = "yuv422"; break;
        case VIDEOMODE_800x600YUV422:   width = 800; height = 600; encoding = "yuv422"; break;
        case VIDEOMODE_1024x768YUV422:  width = 1024; height = 768; encoding = "yuv422"; break;
        case VIDEOMODE_1280x960YUV422:  width = 1280; height = 960; encoding = "yuv422"; break;
        case VIDEOMODE_1600x1200YUV422: width = 1600; height = 1200; encoding = "yuv422"; break;
        case VIDEOMODE_640x480YUV411:   width = 640; height = 480; encoding = "yuv411"; break;
        case VIDEOMODE_640x480RGB:      width = 640; height = 480; encoding = "rgb"; break;
        case VIDEOMODE_800x600RGB:      width = 800; height = 600; encoding = "rgb"; break;
        case VIDEOMODE_1024x768RGB:     width = 1024; height = 768; encoding = "rgb"; break;
        case VIDEOMODE_1280x960RGB:     width = 1280; height = 960; encoding = "rgb"; break;
        case VIDEOMODE_1600x1200RGB:    width = 1600; height = 1200; encoding = "rgb"; break;
        case VIDEOMODE_640x480Y8:       width = 640; height = 480; encoding = "y8"; break;
        case VIDEOMODE_800x600Y8:       width = 800; height = 600; encoding = "y8"; break;
        case VIDEOMODE_1024x768Y8:      width = 1024; height = 768; encoding = "y8"; break;
        case VIDEOMODE_1280x960Y8:      width = 1280; height = 960; encoding = "y8"; break;
        case VIDEOMODE_1600x1200Y8:     width = 1600; height = 1200; encoding = "y8"; break;
        case VIDEOMODE_640x480Y16:      width = 640; height = 480; encoding = "y16"; break;
        case VIDEOMODE_800x600Y16:      width = 800; height = 600; encoding = "y16"; break;
        case VIDEOMODE_1024x768Y16:     width = 1024; height = 768; encoding = "y16"; break;
        case VIDEOMODE_1280x960Y16:     width = 1280; height = 960; encoding = "y16"; break;
        case VIDEOMODE_1600x1200Y16:    width = 1600; height = 1200; encoding = "y16"; break;
        default: width = 0; height = 0; encoding = ""; break;
    }

    switch (fr)
    {
        case FRAMERATE_3_75: framerate = 3.75; break;
        case FRAMERATE_7_5: framerate = 7.5; break;
        case FRAMERATE_15: framerate = 15; break;
        case FRAMERATE_30: framerate = 30; break;
        case FRAMERATE_60: framerate = 60; break;
        case FRAMERATE_120: framerate = 120; break;
        case FRAMERATE_240: framerate = 240; break;
        default: framerate = 0; break;
    }
}

///////////////////////////////////////////////////////////////////////////////
// conversion
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U> T clamp255(const U& value) 
{
    return value < 0 ? 0 : (value > 255 ? 255 : value); 
}

// not optimized, but better to read:
// static void yuv422toRGB_niceToRead(const cv::Mat& src, cv::Mat& dest)
// {
//     int r,g,b;
//     int rows = src.rows;
//     int cols = src.cols;
//     dest = cv::Mat(rows,cols,CV_8UC3);
//     for (int i = 0, j = 0; i < rows*cols*2; i = i+4, j = j+6)
//     {
//         int u = src.data[i];
//         int y1 = src.data[i+1];
//         int v = src.data[i+2];
//         int y2 = src.data[i+3];
//         int c = y1 - 16;
//         int d = u - 128;
//         int e = v - 128;
//         r = clamp<int, int>((298 * c + 409 * e + 128) >> 8);
//         g = clamp<int, int>((298 * c + 100 * d - 208 * e + 128) >> 8);
//         b = clamp<int, int>((298 * c + 516 * d + 128) >> 8);
//         // RGB 1
//         dest.data[j] = r;
//         dest.data[j+1] = g;
//         dest.data[j+2] = b;
//         c = y2 - 16;
//         r = clamp<int, int>((298 * c + 409 * e + 128) >> 8);
//         g = clamp<int, int>((298 * c + 100 * d - 208 * e + 128) >> 8);
//         b = clamp<int, int>((298 * c + 516 * d + 128) >> 8);
//         // RGB 2
//         dest.data[j+3] = r;
//         dest.data[j+4] = g;
//         dest.data[j+5] = b;
//     }
// }

void Grasshopper::yuv422toRGB(const cv::Mat& src, cv::Mat& dest, const bool BGRtoRGB)
{
#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: yuv422toRGB()");
#endif

    dest = cv::Mat(src.rows,src.cols,CV_8UC3);
    char channelSwitch = 0;
    if (BGRtoRGB) channelSwitch = 2;

    int numThreads = sysconf(_SC_NPROCESSORS_ONLN) - 1; // works for linux and osx > 10.4
    int rgbOffset = src.rows * src.cols * 3 / numThreads;
    int yuvOffset = src.rows * src.cols * 2 / numThreads;

    #pragma omp parallel for num_threads(numThreads)
    for (int t = 0; t < numThreads; ++t)
    {
        int tYuvOffset = t*yuvOffset;
        int tRgbOffset = t*rgbOffset;

        for (int i = 0, j = 0; i < yuvOffset; i = i+4, j = j+6)
        {
            // read first two bytes of yuv422 image
            unsigned char u = src.data[i + tYuvOffset];
            unsigned char y1 = src.data[i+1 + tYuvOffset];
            unsigned char v = src.data[i+2 + tYuvOffset];
            unsigned char y2 = src.data[i+3 + tYuvOffset];

            // do some optimization stuff
            int c = 298*(y1 - 16);
            int d = u - 128;
            int d1 = 100 * d;
            int d2 = 516 * d;
            int e = v - 128;
            int e1 = 409 * e;
            int e2 = 208 * e;
            int f1 = e1 + 128;
            int f2 = d1 - e2 + 128;
            int f3 = d2 + 128;

            // calculate first 3 RGB bytes
            unsigned char r = clamp255<unsigned char, int>((c + f1) >> 8);
            unsigned char g = clamp255<unsigned char, int>((c + f2) >> 8);
            unsigned char b = clamp255<unsigned char, int>((c + f3) >> 8);
            dest.data[j+channelSwitch + tRgbOffset] = r;
            dest.data[j+1 + tRgbOffset] = g;
            dest.data[j-channelSwitch+2 + tRgbOffset] = b;

            // calculate second 3 RGB bytes
            c = 298*(y2 - 16);
            r = clamp255<int>((c + f1) >> 8);
            g = clamp255<int>((c + f2) >> 8);
            b = clamp255<int>((c + f3) >> 8);
            dest.data[j+channelSwitch+3 + tRgbOffset] = r;
            dest.data[j+4 + tRgbOffset] = g;
            dest.data[j-channelSwitch+5 + tRgbOffset] = b;
        }
    }

#ifdef _WITH_TIMER
    OKAPI_TIMER_STOP("grasshopper: yuv422toRGB()");
#endif
}

#ifdef _WITH_OPENCL
static const char* errorToString(cl_int error)
{
    switch(error)
    {
#define CL_ERROR(x) case (x): return #x;
            CL_ERROR(CL_SUCCESS);
            CL_ERROR(CL_DEVICE_NOT_FOUND);
            CL_ERROR(CL_DEVICE_NOT_AVAILABLE);
            CL_ERROR(CL_COMPILER_NOT_AVAILABLE);
            CL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
            CL_ERROR(CL_OUT_OF_RESOURCES);
            CL_ERROR(CL_OUT_OF_HOST_MEMORY);
            CL_ERROR(CL_PROFILING_INFO_NOT_AVAILABLE);
            CL_ERROR(CL_MEM_COPY_OVERLAP);
            CL_ERROR(CL_IMAGE_FORMAT_MISMATCH);
            CL_ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
            CL_ERROR(CL_BUILD_PROGRAM_FAILURE);
            CL_ERROR(CL_MAP_FAILURE);
            CL_ERROR(CL_INVALID_VALUE);
            CL_ERROR(CL_INVALID_DEVICE_TYPE);
            CL_ERROR(CL_INVALID_PLATFORM);
            CL_ERROR(CL_INVALID_DEVICE);
            CL_ERROR(CL_INVALID_CONTEXT);
            CL_ERROR(CL_INVALID_QUEUE_PROPERTIES);
            CL_ERROR(CL_INVALID_COMMAND_QUEUE);
            CL_ERROR(CL_INVALID_HOST_PTR);
            CL_ERROR(CL_INVALID_MEM_OBJECT);
            CL_ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
            CL_ERROR(CL_INVALID_IMAGE_SIZE);
            CL_ERROR(CL_INVALID_SAMPLER);
            CL_ERROR(CL_INVALID_BINARY);
            CL_ERROR(CL_INVALID_BUILD_OPTIONS);
            CL_ERROR(CL_INVALID_PROGRAM);
            CL_ERROR(CL_INVALID_PROGRAM_EXECUTABLE);
            CL_ERROR(CL_INVALID_KERNEL_NAME);
            CL_ERROR(CL_INVALID_KERNEL_DEFINITION);
            CL_ERROR(CL_INVALID_KERNEL);
            CL_ERROR(CL_INVALID_ARG_INDEX);
            CL_ERROR(CL_INVALID_ARG_VALUE);
            CL_ERROR(CL_INVALID_ARG_SIZE);
            CL_ERROR(CL_INVALID_KERNEL_ARGS);
            CL_ERROR(CL_INVALID_WORK_DIMENSION);
            CL_ERROR(CL_INVALID_WORK_GROUP_SIZE);
            CL_ERROR(CL_INVALID_WORK_ITEM_SIZE);
            CL_ERROR(CL_INVALID_GLOBAL_OFFSET);
            CL_ERROR(CL_INVALID_EVENT_WAIT_LIST);
            CL_ERROR(CL_INVALID_EVENT);
            CL_ERROR(CL_INVALID_OPERATION);
            CL_ERROR(CL_INVALID_GL_OBJECT);
            CL_ERROR(CL_INVALID_BUFFER_SIZE);
            CL_ERROR(CL_INVALID_MIP_LEVEL);
#undef CL_ERROR
    default:
            return "Unknown error code";
    }
}

#define CL_RETURN_FALSE(expr, errmsg) {cl_int e=(expr);if(CL_SUCCESS!=e){std::cout<<"OpenCL Error: "<<errmsg<<" ["<<errorToString(e)<<"]"<<std::endl; return false; }}
#define CL_RETURN(expr, errmsg) {cl_int e=(expr);if(CL_SUCCESS!=e){std::cout<<"Error: "<<errmsg<<" ["<<errorToString(e)<<"]"<<std::endl; return; }}
#define SAFE_RELEASE_KERNEL(ptr) {if(ptr){ clReleaseKernel(ptr); ptr = NULL; }}
#define SAFE_RELEASE_PROGRAM(ptr) {if(ptr){ clReleaseProgram(ptr); ptr = NULL; }}
#define SAFE_RELEASE_MEMOBJECT(ptr) {if(ptr){ clReleaseMemObject(ptr); ptr = NULL; }}

static void clPrintBuildLog(cl_program Program, cl_device_id Device)
{
    cl_build_status buildStatus;
    clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL);
    if(buildStatus == CL_SUCCESS)
        return;

    //there were some errors.
    char* buildLog;
    size_t logSize;
    clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    buildLog = new char[logSize + 1];

    clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
    buildLog[logSize] = '\0';

    std::cout << "There were build errors:\n";
    std::cout << buildLog << "\n";

    delete [] buildLog;
}


void Grasshopper::yuv422toRGB_gpu(const cv::Mat& yuv, cv::Mat& rgb, const bool BGRtoRGB)
{
#ifdef _WITH_TIMER
    OKAPI_TIMER_START("grasshopper: yuv422toRGB_gpu()");
#endif
    // write image to device memory
    CL_RETURN(clEnqueueWriteBuffer(clCommandQueue, dYuv, CL_FALSE, 0, 2 * width * height, yuv.data, 0, NULL, NULL), "Failed to enqeue write buffer");
    // set kernel arguments
    CL_RETURN(clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void*) &dYuv), "Failed to set kernel arg 0");
    CL_RETURN(clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void*) &dRgb), "Failed to set kernel arg 1");
    uint switchChannels = 0; // after OpenCL specification you cannot pass "bool" to the kernel
    if (BGRtoRGB) switchChannels = 1;
    CL_RETURN(clSetKernelArg(clKernel, 2, sizeof(uint), (void*) &switchChannels), "Failed to set kernel arg 2");
    // define global and local work size
    size_t globalWorkSize = width * height * 3 / 6;  //1600 * 1200 * 3 / 6
    size_t localWorkSize = 256;
    // start computation
    clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    // read rgb image from device memory
    CL_RETURN(clEnqueueReadBuffer(clCommandQueue, dRgb, CL_TRUE, 0, 3 * width * height, rgb.data, 0, NULL, NULL), "Failed to read buffer from device");
#ifdef _WITH_TIMER
    OKAPI_TIMER_STOP("grasshopper: yuv422toRGB_gpu()");
#endif

}

bool Grasshopper::initializeOpenCL()
{
    cl_int clError;
    cl_platform_id clPlatform;
    CL_RETURN_FALSE(clGetPlatformIDs(1, &clPlatform, NULL), "Failed to get CL platform ID");
    CL_RETURN_FALSE(clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL), "No GPU found on this machine");
    //char deviceName[256];
    //clGetDeviceInfo(clDevice, CL_DEVICE_NAME, 256, &deviceName, NULL);
    //std::cout << "Device: " << deviceName << std::endl;
    clContext = clCreateContext(0, 1, &clDevice, NULL, NULL, &clError);
    CL_RETURN_FALSE(clError, "Failed to create OpenCL context");
    clCommandQueue = clCreateCommandQueue(clContext, clDevice, 0, &clError);
    CL_RETURN_FALSE(clError, "Failed to create command queue in the context");

    // device resources
    dYuv = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 2 * width * height, NULL, &clError);
    CL_RETURN_FALSE(clError, "Failed to create buffer");
    dRgb = clCreateBuffer(clContext, CL_MEM_READ_WRITE, 3 * width * height, NULL, &clError);
    CL_RETURN_FALSE(clError, "Failed to create buffer");

    // load kernel from file
    //char* programCode = NULL;
    //size_t programSize = 0;
    //clLoadProgram("yuv422toRgb.cl", &programCode, &programSize); // load source code from file
    //std::cout << "size: " << programSize <<"\n";
    //clProgram = clCreateProgramWithSource(clContext, 1, (const char**) &programCode, &programSize, &clError);
    size_t programSize = sizeof(clProgramCode)/sizeof(clProgramCode[0]);
    const char* programPointer = &clProgramCode[0];
    clProgram = clCreateProgramWithSource(clContext, 1, (const char**) &programPointer, &programSize, &clError);
    CL_RETURN_FALSE(clError, "Failed to create program");
    clError = clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL); // compile kernel
    if(clError != CL_SUCCESS)
    {
        clPrintBuildLog(clProgram, clDevice);
        return false;
    }
    clKernel = clCreateKernel(clProgram, "yuv422toRgb", &clError);
    CL_RETURN_FALSE(clError, "Failed to build kernel");
    return true;
}

void Grasshopper::cleanupOpenCL()
{
    SAFE_RELEASE_MEMOBJECT(dYuv);
    SAFE_RELEASE_MEMOBJECT(dRgb);
    SAFE_RELEASE_KERNEL(clKernel);
    SAFE_RELEASE_PROGRAM(clProgram);
    if(clCommandQueue) clReleaseCommandQueue(clCommandQueue);
    if(clContext) clReleaseContext(clContext);
}
#endif // _WITH_OPENCL


///////////////////////////////////////////////////////////////////////////////
// example program
///////////////////////////////////////////////////////////////////////////////
#ifdef _STANDALONE
int main(int argc, char** argv)
{
    bool gui = false;
    bool saveImages = false; // only works without gui
    int trigger = 0; 

    for(int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.compare("--gui") == 0  ||  arg.compare("-g") == 0)
            gui = true;
        if (arg.compare("--save") == 0)
            saveImages = true;
        if (arg.compare("--trigger") == 0)
            trigger = atoi(argv[i+1]);
    }       

    // The time from a software trigger to the start of the shutter consists of
    // 1. Register write request to register write response (approx. 49.85us)
    // 2. Trigger latency (approx. 6us)
    // 3. Shutter time
    // 4. Time until data transfer (appox. 1ms)
    // 5. Data transfer (approx. 30ms)

    // Time per call of RetrieveBuffer() equals for 1 or 2 cameras.
    // ==> Image acquisition limits the frame rate.
    // How can the image acquisition get faster?
    // ==> Shutter set to 20 ms
    // ==> Gain set to 1 dB

    // Maximum Frame Rate in External Trigger Mode:
    // Max_Frame_Rate_Trigger = 1 / ( Shutter + ( 1 / Max_Frame_Rate_Free_Running ) )
    // ( = 1 / (0.002 + (1 / 15)) = 14.56 with Shutter=20ms )

    // RetrieveBuffer() needs 95.971 ms / call
    // shutter = 20 ms
    // register write + latenxy = 56 us = 0.056 ms
    // time until transfer = 1 ms
    // => 96 ms - 21 ms = 75 ms data transfer?

    // 30 fps @ 1024x768 Y8 ?
    // 15 fps @ 1600x1200 YUV422

    // FPS = 1 / (Packets Per Frame * 125us)
    // Packets Per Frame = ImageSize * BytesPerPixel / BytesPerPacket  (estimate)


    if (gui)
    {
        // Initialize okapi GUI
        okapi::GuiThread gui;
        okapi::ImageWindow* imgWin = new okapi::ImageWindow("Display");
        gui.addClient(imgWin);
        okapi::WidgetWindow* widWin = new okapi::WidgetWindow("Settings");
        gui.addClient(widWin);
        gui.start();

        imgWin->resize(0,0,1500,700);
        widWin->setButton("Show Shutter", false);
        widWin->setButton("Show Gain", false);

        // Initialize cameras
        Grasshopper g(trigger);
        g.resetBus(); // does not change anything...


        if (!g.initCameras(1600,1200,"yuv422",15))
        {
            printf("Could not initialize the cameras! Exiting... \n");
            return -1;
        }

        // print possible video modes
        g.printVideoModes(0 /*cam index*/);

        g.printInfo();

        // set shutter to specified milliseconds,
        // gain will be autmatically set to auto
        g.setShutter(40);

        // get number of cameras
        int numCameras = g.getNumCameras();

        // set cam 0 to master and distribute its properties (shutter, gain, etc.)
        g.distributeCamProperties(0);

        // trigger and catch frames
        g.getNextFrame();

        int counter = 0;
        // main loop
        for (;;)
        {
            /*counter++;
            if (counter == 50)
            {
                g.setROI(0,0,1000,1000,1);
            }
            if (counter == 100)
            {
                g.setROI(400,400, 2000, 2000, 1);
            }
            if (counter == 150)
            {
                g.setROI(20,30,100,400,1);
            }*/

            // g.saveImages(i); // this would save them to disk

            // calculate frames per second
            g.tickFPS();
            std::string fps = g.getProcessedFPSString();


            for (int cam = 0; cam < numCameras; ++cam)
            {
                //Image img = g.getFlyCapImage(cam);
                cv::Mat img = g.getImage(cam);

                // write status in left upper image corner
                OKAPI_TIMER_START("main(): setting deco and image");
                okapi::ImageDeco deco(img);
                deco.setColor(0, 0, 0);
                deco.setThickness(3);
                deco.setTextFont(40);
                std::stringstream status; // status string shown inside the image
                status << fps;
                if (widWin->getButton("Show Shutter"))
                    status << g.getProperty(SHUTTER, cam) << "\n";
                if (widWin->getButton("Show Gain"))
                    status << g.getProperty(GAIN, cam) << "\n";
                deco.drawText(status.str(), 10, 50);

                // set images in gui
                imgWin->setImage(okapi::strprintf("camera %i", cam), img, 0.45f);
                OKAPI_TIMER_STOP("main(): setting deco and image");

            }

            // end application if okapi window is closed
            if (!imgWin->getWindowState() || !widWin->getWindowState()) break;

            // set cam 0 to master and distribute its properties (shutter, gain, etc.)
            g.distributeCamProperties(0);
            // trigger and catch frames
            g.getNextFrame();
        }

        std::cout << "Stopping cameras... \n";
        // restore default values before stopping
        g.restoreDefaultProperties();
        // stop cameras and clean up
        g.stopCameras();

        // Stop okapi GUI
        imgWin->setWindowState(false);
        widWin->setWindowState(false);
        gui.stop();
    }
    else
    {
        // minimal example program
        Grasshopper g(trigger);
        if (!g.initCameras(1600, 1200, "yuv422", 15))
        {
            printf("Could not initialize the cameras! Exiting... \n");
            return -1;
        }
        g.printVideoModes(0 /*cam index*/);
        g.setShutter(20);

        int numCameras = g.getNumCameras();
        int numImages = 200;
        for (int i = 0; i < numImages; ++i)
        {
            g.distributeCamProperties(0);
            g.getNextFrame();

            for (int cam = 0; cam < numCameras; ++cam)
            {
                cv::Mat img = g.getImage(cam);
            }

            if (saveImages) g.saveImages(i);

            double fps = g.tickFPS();
            std::cout << "frame " << i+1 << "/" << numImages <<" , fps: " << fps << "\n";
        }

        g.restoreDefaultProperties();
        g.stopCameras();
    }

    return 0;
}
#endif
