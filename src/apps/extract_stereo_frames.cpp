#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void printUsage(const char *progName)
{
    std::cout << "Usage:/n"
              << "  " << progName << " <left_video> <right_video> <output_dir>\n";
}

int main(int argc, char **argv)
{
    std::string leftVideoPath = "D:/doducnghia/Hcmut/Major/Computer_Vision/Images/stereo/leftCam_Detect.mp4";
    std::string rightVideoPath = "D:/doducnghia/Hcmut/Major/Computer_Vision/Images/stereo/rightCam_Detect.mp4";
    std::string outputDir = "D:/doducnghia/Hcmut/Major/Computer_Vision/Images/stereo_object_images";

    // Create output directory if it does not exist
    try
    {
        if (!fs::exists(outputDir))
        {
            fs::create_directories(outputDir);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to create output directory: " << e.what() << "/n";
        return 1;
    }

    cv::VideoCapture capL(leftVideoPath);
    cv::VideoCapture capR(rightVideoPath);

    if (!capL.isOpened())
    {
        std::cerr << "Cannot open left video: " << leftVideoPath << "\n";
        return 1;
    }
    if (!capR.isOpened())
    {
        std::cerr << "Cannot open right video: " << rightVideoPath << "\n";
        return 1;
    }

    int pairIndex = 0;
    std::cout << "Press 's' to save a stereo pair, 'q' or ESC to quit.\n";

    while (true)
    {
        cv::Mat frameL, frameR;

        // Grab both frames first for better time alignment
        if (!capL.grab() || !capR.grab())
        {
            std::cout << "End of one of the videos./n";
            break;
        }

        capL.retrieve(frameL);
        capR.retrieve(frameR);

        if (frameL.empty() || frameR.empty())
        {
            std::cout << "Empty frame encountered, stopping./n";
            break;
        }

        // Optionally resize for display if too large
        // Scale factor for display (0.5 = 50% size)
        const double displayScale = 0.5;

        // Resize frames for display only
        cv::Mat dispL, dispR;
        cv::resize(frameL, dispL, cv::Size(), displayScale, displayScale, cv::INTER_AREA);
        cv::resize(frameR, dispR, cv::Size(), displayScale, displayScale, cv::INTER_AREA);

        // Put small text overlay to indicate which is which
        cv::putText(dispL, "LEFT", cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(dispR, "RIGHT", cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // Show in two windows
        cv::imshow("Left", dispL);
        cv::imshow("Right", dispR);

        int key = cv::waitKey(30);
        if (key == 'q' || key == 27) // 'q' or ESC
        {
            break;
        }
        else if (key == 's')
        {
            fs::path outDirPath(outputDir);
            fs::path leftPath = outDirPath / ("left" + std::to_string(pairIndex) + ".png");
            fs::path rightPath = outDirPath / ("right" + std::to_string(pairIndex) + ".png");

            std::string leftName = leftPath.string();
            std::string rightName = rightPath.string();

            bool okL = cv::imwrite(leftName, frameL);
            bool okR = cv::imwrite(rightName, frameR);

            if (okL && okR)
            {
                std::cout << "Saved pair #" << pairIndex
                          << " -> " << leftName << " , " << rightName << "\n";
                pairIndex++;
            }
            else
            {
                std::cerr << "Failed to save one of the images.\n";
            }
        }
    }

    return 0;
}
