#include <opencv2/opencv.hpp>
#include <iostream>

struct StereoCalibrationConfig
{
    cv::Size boardSize; // number of inner corners (columns, rows), e.g. (9,6)
    float squareSize;   // in mm
};

bool loadImagePairs(const std::string &imgDir,
                    std::vector<std::string> &leftPaths,
                    std::vector<std::string> &rightPaths)
{
    // Use glob to find left*.png and right*.png
    std::vector<cv::String> leftFiles, rightFiles;
    cv::glob(imgDir + "/left*.png", leftFiles);
    cv::glob(imgDir + "/right*.png", rightFiles);

    if (leftFiles.empty() || rightFiles.empty())
    {
        std::cerr << "No left*.png or right*.png images found in directory: " << imgDir << "\n";
        return false;
    }

    // Sort them to keep consistent order
    std::sort(leftFiles.begin(), leftFiles.end());
    std::sort(rightFiles.begin(), rightFiles.end());

    if (leftFiles.size() != rightFiles.size())
    {
        std::cerr << "Number of left and right images does not match.\n";
        std::cerr << "Left: " << leftFiles.size()
                  << ", Right: " << rightFiles.size() << "\n";
        return false;
    }

    leftPaths.assign(leftFiles.begin(), leftFiles.end());
    rightPaths.assign(rightFiles.begin(), rightFiles.end());
    return true;
}

void createObjectPoints(const StereoCalibrationConfig &cfg,
                        std::vector<cv::Point3f> &objectCorners)
{
    objectCorners.clear();
    // Generate 3D points for the checkerboard in its own plane (Z=0)
    for (int j = 0; j < cfg.boardSize.height; ++j)
    {
        for (int i = 0; i < cfg.boardSize.width; ++i)
        {
            objectCorners.emplace_back(
                i * cfg.squareSize,
                j * cfg.squareSize,
                0.0f);
        }
    }
}

int main(int argc, char **argv)
{
    std::string imgDir = "D:/doducnghia/Hcmut/Major/Computer_Vision/Images/stereo_calib_images";
    std::string outputYaml = "D:/doducnghia/Hcmut/Major/Computer_Vision/Images/stereo_output/stereo_output.yaml";

    StereoCalibrationConfig cfg;
    cfg.boardSize = cv::Size(9, 6); // inner corners
    cfg.squareSize = 24.8f;         // mm

    std::vector<std::string> leftPaths, rightPaths;
    if (!loadImagePairs(imgDir, leftPaths, rightPaths))
    {
        return 1;
    }

    std::vector<std::vector<cv::Point2f>> imgPointsL, imgPointsR;
    std::vector<std::vector<cv::Point3f>> objPoints; // shared for both cameras

    std::vector<cv::Point3f> objectCorners;
    createObjectPoints(cfg, objectCorners);

    cv::Size imageSize;

    for (size_t i = 0; i < leftPaths.size(); ++i)
    {
        cv::Mat imgL = cv::imread(leftPaths[i], cv::IMREAD_GRAYSCALE);
        cv::Mat imgR = cv::imread(rightPaths[i], cv::IMREAD_GRAYSCALE);

        if (imgL.empty() || imgR.empty())
        {
            std::cerr << "Failed to read image pair index " << i << "\n";
            continue;
        }

        if (imageSize.width == 0)
        {
            imageSize = imgL.size();
        }

        std::vector<cv::Point2f> cornersL, cornersR;
        bool foundL = cv::findChessboardCorners(
            imgL, cfg.boardSize, cornersL,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        bool foundR = cv::findChessboardCorners(
            imgR, cfg.boardSize, cornersR,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (!foundL || !foundR)
        {
            std::cout << "Chessboard not found in pair index " << i << ", skipping.\n";
            continue;
        }

        // Refine corner locations for better accuracy
        cv::cornerSubPix(
            imgL, cornersL,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
        cv::cornerSubPix(
            imgR, cornersR,
            cv::Size(11, 11),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

        imgPointsL.push_back(cornersL);
        imgPointsR.push_back(cornersR);
        objPoints.push_back(objectCorners);

        std::cout << "Pair " << i << ": chessboard detected.\n";
    }

    if (imgPointsL.size() < 3)
    {
        std::cerr << "Not enough valid pairs with detected chessboard.\n";
        return 1;
    }

    std::cout << "Using " << imgPointsL.size() << " valid stereo pairs for calibration.\n";

    // Calibrate each camera individually (single-camera calibration)
    cv::Mat cameraMatrixL = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffsL = cv::Mat::zeros(8, 1, CV_64F);
    cv::Mat cameraMatrixR = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffsR = cv::Mat::zeros(8, 1, CV_64F);

    std::vector<cv::Mat> rvecsL, tvecsL;
    std::vector<cv::Mat> rvecsR, tvecsR;

    int camFlags = 0;

    double rmsL = cv::calibrateCamera(
        objPoints, imgPointsL, imageSize,
        cameraMatrixL, distCoeffsL, rvecsL, tvecsL,
        camFlags);
    double rmsR = cv::calibrateCamera(
        objPoints, imgPointsR, imageSize,
        cameraMatrixR, distCoeffsR, rvecsR, tvecsR,
        camFlags);

    std::cout << "Left camera RMS reprojection error:  " << rmsL << "\n";
    std::cout << "Right camera RMS reprojection error: " << rmsR << "\n";

    // Stereo calibration (refines R and T between cameras)
    cv::Mat R, T, E, F;

    // int flags = cv::CALIB_FIX_INTRINSIC; // keep intrinsics fixed, only refine extrinsics
    cv::TermCriteria stereoCriteria(
        cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
        100, 1e-5);

    int stereoFlags = 0;
    stereoFlags |= cv::CALIB_FIX_INTRINSIC;

    double rmsStereo = cv::stereoCalibrate(
        objPoints, imgPointsL, imgPointsR,
        cameraMatrixL, distCoeffsL,
        cameraMatrixR, distCoeffsR,
        imageSize, R, T, E, F,
        stereoFlags, stereoCriteria);

    std::cout << "Stereo calibration RMS reprojection error: " << rmsStereo << "\n";

    // Stereo rectification
    cv::Mat R1, R2, P1, P2, Q;
    cv::Rect validROI1, validROI2;

    cv::stereoRectify(
        cameraMatrixL, distCoeffsL,
        cameraMatrixR, distCoeffsR,
        imageSize, R, T,
        R1, R2, P1, P2, Q,
        cv::CALIB_ZERO_DISPARITY, -1, imageSize,
        &validROI1, &validROI2);

    // Save everything to YAML
    cv::FileStorage fs(outputYaml, cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open YAML file for writing: " << outputYaml << "\n";
        return 1;
    }

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    fs << "board_width" << cfg.boardSize.width;
    fs << "board_height" << cfg.boardSize.height;
    fs << "square_size" << cfg.squareSize;

    fs << "cameraMatrixL" << cameraMatrixL;
    fs << "distCoeffsL" << distCoeffsL;
    fs << "cameraMatrixR" << cameraMatrixR;
    fs << "distCoeffsR" << distCoeffsR;

    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;

    fs << "R1" << R1;
    fs << "R2" << R2;
    fs << "P1" << P1;
    fs << "P2" << P2;
    fs << "Q" << Q;

    fs << "validROI1" << validROI1;
    fs << "validROI2" << validROI2;

    fs.release();
    std::cout << "Stereo calibration parameters saved to: " << outputYaml << "\n";

    return 0;
}
