#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

struct StereoCalibData
{
    cv::Mat cameraMatrixL;
    cv::Mat distCoeffsL;
    cv::Mat cameraMatrixR;
    cv::Mat distCoeffsR;
    cv::Mat R; // rotation from left to right
    cv::Mat T; // translation from left to right
    cv::Size imageSize;
};

bool loadStereoCalib(const std::string &yamlPath, StereoCalibData &data)
{
    cv::FileStorage fs(yamlPath, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Cannot open stereo calibration file: " << yamlPath << "\n";
        return false;
    }

    int w = 0, h = 0;
    fs["image_width"] >> w;
    fs["image_height"] >> h;
    data.imageSize = cv::Size(w, h);

    fs["cameraMatrixL"] >> data.cameraMatrixL;
    fs["distCoeffsL"] >> data.distCoeffsL;
    fs["cameraMatrixR"] >> data.cameraMatrixR;
    fs["distCoeffsR"] >> data.distCoeffsR;

    fs["R"] >> data.R;
    fs["T"] >> data.T;

    fs.release();

    if (data.cameraMatrixL.empty() || data.cameraMatrixR.empty())
    {
        std::cerr << "Invalid camera matrices in YAML.\n";
        return false;
    }

    std::cout << "Loaded stereo calibration from: " << yamlPath << "\n";
    std::cout << "Image size: " << data.imageSize.width << " x "
              << data.imageSize.height << "\n";
    return true;
}

// Compute squared Euclidean distance between two 2D points
double squaredDist(const cv::Point2f &a, const cv::Point2f &b)
{
    double dx = static_cast<double>(a.x - b.x);
    double dy = static_cast<double>(a.y - b.y);
    return dx * dx + dy * dy;
}

// Sort 4 points of a quadrilateral into TL, TR, BR, BL order
// using sum and diff heuristics.
std::vector<cv::Point2f> sortQuadCorners(const std::vector<cv::Point2f> &pts)
{
    CV_Assert(pts.size() == 4);
    std::vector<cv::Point2f> sorted(4);

    // TL has min(x+y), BR has max(x+y)
    // TR has max(x-y), BL has min(x-y)
    double minSum = 1e12, maxSum = -1e12;
    double minDiff = 1e12, maxDiff = -1e12;
    int idxTL = 0, idxBR = 0, idxBL = 0, idxTR = 0;

    for (int i = 0; i < 4; ++i)
    {
        double sum = pts[i].x + pts[i].y;
        double diff = pts[i].x - pts[i].y;

        if (sum < minSum)
        {
            minSum = sum;
            idxTL = i;
        }
        if (sum > maxSum)
        {
            maxSum = sum;
            idxBR = i;
        }
        if (diff < minDiff)
        {
            minDiff = diff;
            idxBL = i;
        }
        if (diff > maxDiff)
        {
            maxDiff = diff;
            idxTR = i;
        }
    }

    sorted[0] = pts[idxTL]; // TL
    sorted[1] = pts[idxTR]; // TR
    sorted[2] = pts[idxBR]; // BR
    sorted[3] = pts[idxBL]; // BL

    return sorted;
}

// Detect the white object inside ROI and return 4 ordered corners (TL, TR, BR, BL)
// in full-image coordinates. Returns false if detection fails.
bool detectObjectQuad(const cv::Mat &imgRectified,
                      const cv::Rect &roi,
                      std::vector<cv::Point2f> &quadCorners,
                      bool debugShow = true,
                      const std::string &debugWindowName = "")
{
    CV_Assert(imgRectified.channels() == 1 || imgRectified.channels() == 3);

    cv::Mat roiImg = imgRectified(roi).clone();
    cv::Mat gray;
    if (roiImg.channels() == 3)
        cv::cvtColor(roiImg, gray, cv::COLOR_BGR2GRAY);
    else
        gray = roiImg;

    // Slight blur to reduce noise
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // Threshold: board is dark, object is bright.
    cv::Mat bin;
    cv::threshold(gray, bin, 190, 255, cv::THRESH_BINARY);

    // Morphological operations to clean up
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        std::cerr << "No contours found in ROI.\n";
        return false;
    }

    // Pick the largest contour by area
    size_t bestIdx = 0;
    double bestArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        double area = cv::contourArea(contours[i]);
        if (area > bestArea)
        {
            bestArea = area;
            bestIdx = i;
        }
    }

    if (bestArea < 10.0)
    {
        std::cerr << "Largest contour area too small, likely noise.\n";
        return false;
    }

    // Approximate contour with rotated rectangle and get its 4 vertices
    cv::RotatedRect rrect = cv::minAreaRect(contours[bestIdx]);
    cv::Point2f pts[4];
    rrect.points(pts);

    std::vector<cv::Point2f> localCorners(pts, pts + 4);

    // Convert local ROI coordinates to full-image coordinates
    for (auto &p : localCorners)
    {
        p.x += static_cast<float>(roi.x);
        p.y += static_cast<float>(roi.y);
    }

    quadCorners = sortQuadCorners(localCorners);

    if (debugShow)
    {
        cv::Mat dbg = imgRectified.clone();
        cv::cvtColor(dbg, dbg, cv::COLOR_GRAY2BGR);

        // Draw ROI
        cv::rectangle(dbg, roi, cv::Scalar(0, 255, 0), 2);

        // Draw quadrilateral
        for (int i = 0; i < 4; ++i)
        {
            cv::circle(dbg, quadCorners[i], 5, cv::Scalar(0, 0, 255), -1);
            cv::putText(dbg, std::to_string(i), quadCorners[i] + cv::Point2f(5, -5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        }

        // Scale down for display at 0.5x
        double dispScale = 0.5;
        cv::Mat dbgSmall;
        cv::resize(dbg, dbgSmall, cv::Size(), dispScale, dispScale, cv::INTER_AREA);

        if (!debugWindowName.empty())
            cv::imshow(debugWindowName, dbgSmall);
        else
            cv::imshow("Object detection", dbgSmall);
    }

    return true;
}

// Convert rotation matrix to yaw-pitch-roll (Z-Y-X Tait-Bryan angles).
// R = Rz(yaw) * Ry(pitch) * Rx(roll).
// Angles returned in radians.
cv::Vec3d rotationMatrixToYPR(const cv::Mat &R)
{
    CV_Assert(R.rows == 3 && R.cols == 3);

    double r00 = R.at<double>(0, 0);
    double r01 = R.at<double>(0, 1);
    double r02 = R.at<double>(0, 2);
    double r10 = R.at<double>(1, 0);
    double r11 = R.at<double>(1, 1);
    double r12 = R.at<double>(1, 2);
    double r20 = R.at<double>(2, 0);
    double r21 = R.at<double>(2, 1);
    double r22 = R.at<double>(2, 2);

    double yaw = std::atan2(r10, r00);  // around Z
    double pitch = std::asin(-r20);     // around Y
    double roll = std::atan2(r21, r22); // around X

    return cv::Vec3d(yaw, pitch, roll);
}

// Build object-to-camera rotation matrix from 4 ordered 3D corners.
// Corner order: 0=TL, 1=TR, 2=BR, 3=BL.
cv::Mat buildObjectRotation(const std::vector<cv::Point3d> &pts3D)
{
    CV_Assert(pts3D.size() == 4);

    cv::Point3d TL = pts3D[0];
    cv::Point3d TR = pts3D[1];
    cv::Point3d BR = pts3D[2];
    cv::Point3d BL = pts3D[3];

    // Local object axes:
    //  u: along width (TL -> TR)
    //  v: along height (TL -> BL)
    //  n: normal = u x v
    cv::Point3d u_vec = TR - TL;
    cv::Point3d v_vec = BL - TL;

    auto normalize3D = [](const cv::Point3d &v) -> cv::Point3d
    {
        double norm = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (norm < 1e-9)
            return cv::Point3d(0, 0, 0);
        return cv::Point3d(v.x / norm, v.y / norm, v.z / norm);
    };

    u_vec = normalize3D(u_vec);
    v_vec = normalize3D(v_vec);

    // Cross product u x v
    cv::Point3d n_vec(
        u_vec.y * v_vec.z - u_vec.z * v_vec.y,
        u_vec.z * v_vec.x - u_vec.x * v_vec.z,
        u_vec.x * v_vec.y - u_vec.y * v_vec.x);
    n_vec = normalize3D(n_vec);

    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

    // Columns of R are basis vectors in camera coordinates
    R.at<double>(0, 0) = u_vec.x;
    R.at<double>(1, 0) = u_vec.y;
    R.at<double>(2, 0) = u_vec.z;
    R.at<double>(0, 1) = v_vec.x;
    R.at<double>(1, 1) = v_vec.y;
    R.at<double>(2, 1) = v_vec.z;
    R.at<double>(0, 2) = n_vec.x;
    R.at<double>(1, 2) = n_vec.y;
    R.at<double>(2, 2) = n_vec.z;

    return R;
}

int main(int argc, char **argv)
{
    std::string yamlPath = "D:/doducnghia/Hcmut/Major/Computer_Vision/Images/stereo_output/stereo_output.yaml";
    std::string leftPath = "D:/doducnghia/Hcmut/Major/Computer_Vision/Images/stereo_object_images/left0.png";
    std::string rightPath = "D:/doducnghia/Hcmut/Major/Computer_Vision/Images/stereo_object_images/right0.png";

    StereoCalibData calib;
    if (!loadStereoCalib(yamlPath, calib))
    {
        return 1;
    }

    // Load left/right images
    cv::Mat leftImg = cv::imread(leftPath, cv::IMREAD_COLOR);
    cv::Mat rightImg = cv::imread(rightPath, cv::IMREAD_COLOR);

    if (leftImg.empty() || rightImg.empty())
    {
        std::cerr << "Failed to read left or right image.\n";
        return 1;
    }

    // DEBUG for size
    std::cout << "leftImg size  = " << leftImg.cols << " x " << leftImg.rows << "\n";
    std::cout << "rightImg size = " << rightImg.cols << " x " << rightImg.rows << "\n";
    std::cout << "calib size    = " << calib.imageSize.width
              << " x " << calib.imageSize.height << "\n";

    // Resize images to calibration size if needed
    if (leftImg.size() != calib.imageSize)
    {
        std::cout << "Resizing images to calibration size...\n";
        cv::resize(leftImg, leftImg, calib.imageSize, 0, 0, cv::INTER_AREA);
        cv::resize(rightImg, rightImg, calib.imageSize, 0, 0, cv::INTER_AREA);
    }

    // Work directly on original images
    cv::Mat leftRect = leftImg.clone();
    cv::Mat rightRect = rightImg.clone();

    // Convert to grayscale for processing
    cv::Mat leftRectGray, rightRectGray;
    cv::cvtColor(leftRect, leftRectGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightRect, rightRectGray, cv::COLOR_BGR2GRAY);

    const double roiDisplayScale = 0.5;

    std::cout << "Select ROI on the LEFT image (board area), then press ENTER.\n";

    // Create a resized version for easier viewing
    cv::Mat leftDisp;
    cv::resize(leftRect, leftDisp, cv::Size(), roiDisplayScale, roiDisplayScale, cv::INTER_AREA);

    cv::imshow("Left (for ROI)", leftDisp);

    cv::Rect roiDispL = cv::selectROI("Left (for ROI)", leftDisp);

    if (roiDispL.width <= 0 || roiDispL.height <= 0)
    {
        std::cerr << "Invalid LEFT ROI selected.\n";
        return 1;
    }

    // Scale ROI left back to original image coordinates
    cv::Rect roiL;
    roiL.x = static_cast<int>(roiDispL.x / roiDisplayScale);
    roiL.y = static_cast<int>(roiDispL.y / roiDisplayScale);
    roiL.width = static_cast<int>(roiDispL.width / roiDisplayScale);
    roiL.height = static_cast<int>(roiDispL.height / roiDisplayScale);

    // Clamp to ensure ROI is within image bounds
    roiL &= cv::Rect(0, 0, leftRect.cols, leftRect.rows);

    std::cout << "Using LEFT ROI (original coords): x=" << roiL.x << ", y=" << roiL.y
              << ", w=" << roiL.width << ", h=" << roiL.height << "\n";

    // Repeat for right image
    std::cout << "Select ROI on the RIGHT image (board area), then press ENTER.\n";

    cv::Mat rightDisp;
    cv::resize(rightRect, rightDisp, cv::Size(), roiDisplayScale, roiDisplayScale, cv::INTER_AREA);

    cv::imshow("Right (for ROI)", rightDisp);
    cv::Rect roiDispR = cv::selectROI("Right (for ROI)", rightDisp);

    if (roiDispR.width <= 0 || roiDispR.height <= 0)
    {
        std::cerr << "Invalid RIGHT ROI selected.\n";
        return 1;
    }

    // Scale ROI right back to original image coordinates
    cv::Rect roiR;
    roiR.x = static_cast<int>(roiDispR.x / roiDisplayScale);
    roiR.y = static_cast<int>(roiDispR.y / roiDisplayScale);
    roiR.width = static_cast<int>(roiDispR.width / roiDisplayScale);
    roiR.height = static_cast<int>(roiDispR.height / roiDisplayScale);

    // Clamp
    roiR &= cv::Rect(0, 0, rightRect.cols, rightRect.rows);

    std::cout << "Using RIGHT ROI (original coords): x=" << roiR.x << ", y=" << roiR.y
              << ", w=" << roiR.width << ", h=" << roiR.height << "\n";

    std::vector<cv::Point2f> cornersL, cornersR;

    if (!detectObjectQuad(leftRectGray, roiL, cornersL, true, "Left object"))
    {
        std::cerr << "Failed to detect object in LEFT image.\n";
        cv::waitKey(0);
        return 1;
    }
    if (!detectObjectQuad(rightRectGray, roiR, cornersR, true, "Right object"))
    {
        std::cerr << "Failed to detect object in RIGHT image.\n";
        cv::waitKey(0);
        return 1;
    }

    // Prepare points for triangulation (need in float and matched order)
    CV_Assert(cornersL.size() == 4 && cornersR.size() == 4);

    // Undistort points to normalized coordinates (left/right)
    std::vector<cv::Point2f> undL, undR;
    cv::undistortPoints(cornersL, undL, calib.cameraMatrixL, calib.distCoeffsL);
    cv::undistortPoints(cornersR, undR, calib.cameraMatrixR, calib.distCoeffsR);

    cv::Mat ptsL(2, 4, CV_64F);
    cv::Mat ptsR(2, 4, CV_64F);
    for (int i = 0; i < 4; ++i)
    {
        ptsL.at<double>(0, i) = undL[i].x;
        ptsL.at<double>(1, i) = undL[i].y;

        ptsR.at<double>(0, i) = undR[i].x;
        ptsR.at<double>(1, i) = undR[i].y;
    }

    // Build projection matrices P1, P2 for triangulation in left camera coords
    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F); // [I | 0]

    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
    calib.R.copyTo(P2(cv::Rect(0, 0, 3, 3))); // R in left-camera coords
    calib.T.copyTo(P2(cv::Rect(3, 0, 1, 3))); // T column

    // Triangulate using rectified projection matrices P1, P2
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, ptsL, ptsR, points4D);

    // Convert from homogeneous to 3D points (in camera coordinate system of left rectified camera)
    std::vector<cv::Point3d> pts3D(4);
    for (int i = 0; i < 4; ++i)
    {
        double X = points4D.at<double>(0, i);
        double Y = points4D.at<double>(1, i);
        double Z = points4D.at<double>(2, i);
        double W = points4D.at<double>(3, i);

        if (std::fabs(W) < 1e-12)
        {
            pts3D[i] = cv::Point3d(0, 0, 0);
        }
        else
        {
            pts3D[i] = cv::Point3d(X / W, Y / W, Z / W);
        }
    }

    // Compute center point
    cv::Point3d center(0, 0, 0);
    for (const auto &p : pts3D)
    {
        center.x += p.x;
        center.y += p.y;
        center.z += p.z;
    }
    center.x /= 4.0;
    center.y /= 4.0;
    center.z /= 4.0;

    // Compute width and height (average of opposite edges)
    auto euclideanDist3D = [](const cv::Point3d &a, const cv::Point3d &b) -> double
    {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        double dz = a.z - b.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    double width1 = euclideanDist3D(pts3D[0], pts3D[1]);  // TL-TR
    double width2 = euclideanDist3D(pts3D[3], pts3D[2]);  // BL-BR
    double height1 = euclideanDist3D(pts3D[0], pts3D[3]); // TL-BL
    double height2 = euclideanDist3D(pts3D[1], pts3D[2]); // TR-BR

    double widthAvg = 0.5 * (width1 + width2);
    double heightAvg = 0.5 * (height1 + height2);

    // Build rotation matrix of object relative to camera and compute yaw-pitch-roll
    cv::Mat R_obj = buildObjectRotation(pts3D);
    cv::Vec3d yprRad = rotationMatrixToYPR(R_obj);

    double yawDeg = yprRad[0] * 180.0 / CV_PI;
    double pitchDeg = yprRad[1] * 180.0 / CV_PI;
    double rollDeg = yprRad[2] * 180.0 / CV_PI;

    // Print results
    std::cout << "\n=== 3D measurement result (units consistent with calibration, typically mm) ===\n";
    std::cout << "Corner 3D points (TL, TR, BR, BL):\n";
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "  P" << i << " = [" << pts3D[i].x << ", "
                  << pts3D[i].y << ", " << pts3D[i].z << "]\n";
    }

    std::cout << "\nObject center (in left camera coords):\n";
    std::cout << "  C = [" << center.x << ", " << center.y << ", " << center.z << "]\n";

    std::cout << "\nEstimated object size:\n";
    std::cout << "  Width  ~= " << widthAvg << "\n";
    std::cout << "  Height ~= " << heightAvg << "\n";

    std::cout << "\nObject orientation (yaw-pitch-roll, degrees, Z-Y-X):\n";
    std::cout << "  Yaw   (around Z): " << yawDeg << " deg\n";
    std::cout << "  Pitch (around Y): " << pitchDeg << " deg\n";
    std::cout << "  Roll  (around X): " << rollDeg << " deg\n";

    std::cout << "\nDepth (Z of center from left camera): " << center.z << "\n";

    std::cout << "\nPress any key to close visualization windows.\n";
    cv::waitKey(0);

    return 0;
}
