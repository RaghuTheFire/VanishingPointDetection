#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>

// Threshold by which lines will be rejected wrt the horizontal
const double REJECT_DEGREE_TH = 4.0;

std::vector<cv::Mat> ReadImage(const std::string& InputImagePath) 
{
    std::vector<cv::Mat> Images;
    std::vector<std::string> ImageNames;

    // Checking if path is of file or folder.
    if (cv::utils::fs::isFile(InputImagePath)) 
    {
        // If path is of file.
        cv::Mat InputImage = cv::imread(InputImagePath);

        // Checking if image is read.
        if (InputImage.empty()) 
        {
            std::cerr << "Image not read. Provide a correct path" << std::endl;
            exit(EXIT_FAILURE);
        }

        Images.push_back(InputImage);
        ImageNames.push_back(cv::utils::fs::getFilename(InputImagePath));
    } 
    else 
    if (cv::utils::fs::isDirectory(InputImagePath)) 
    {
        // If path is of a folder containing images.
        std::vector<cv::String> filenames;
        cv::utils::fs::glob(InputImagePath, filenames);

        for (const auto& filename : filenames) 
        {
            cv::Mat InputImage = cv::imread(filename);
            Images.push_back(InputImage);
            ImageNames.push_back(cv::utils::fs::getFilename(filename));
        }
    } 
    else 
    {
        std::cerr << "Enter valid Image Path." << std::endl;
        exit(EXIT_FAILURE);
    }

    return Images;
}

std::vector<std::vector<double>> FilterLines(const std::vector<cv::Vec4i>& Lines) 
{
    std::vector<std::vector<double>> FinalLines;

    for (const auto& Line : Lines) 
    {
        int x1 = Line[0], y1 = Line[1], x2 = Line[2], y2 = Line[3];

        // Calculating equation of the line: y = mx + c
        double m, c;
        if (x1 != x2) 
        {
            m = static_cast<double>(y2 - y1) / (x2 - x1);
        } 
        else 
        {
            m = 1e9;
        }
        c = y2 - m * x2;
        // theta will contain values between -90 -> +90.
        double theta = std::atan(m) * 180.0 / CV_PI;

        // Rejecting lines of slope near to 0 degree or 90 degree and storing others
        if (REJECT_DEGREE_TH <= std::abs(theta) && std::abs(theta) <= (90 - REJECT_DEGREE_TH)) 
        {
            double l = std::sqrt(std::pow(y2 - y1, 2) + std::pow(x2 - x1, 2)); // length of the line
            FinalLines.push_back({static_cast<double>(x1), static_cast<double>(y1), static_cast<double>(x2), static_cast<double>(y2), m, c, l});
        }
    }

    // Removing extra lines
    // (we might get many lines, so we are going to take only longest 15 lines
    // for further computation because more than this number of lines will only
    // contribute towards slowing down of our algo.)
    if (FinalLines.size() > 15) 
    {
        std::sort(FinalLines.begin(), FinalLines.end(), [](const std::vector<double>& a, const std::vector<double>& b) 
        {
            return a[6] > b[6];
        });
        FinalLines.resize(15);
    }

    return FinalLines;
}

std::vector<std::vector<double>> GetLines(const cv::Mat& Image) 
{
    // Converting to grayscale
    cv::Mat GrayImage;
    cv::cvtColor(Image, GrayImage, cv::COLOR_BGR2GRAY);
    // Blurring image to reduce noise.
    cv::Mat BlurGrayImage;
    cv::GaussianBlur(GrayImage, BlurGrayImage, cv::Size(5, 5), 1);
    // Generating Edge image
    cv::Mat EdgeImage;
    cv::Canny(BlurGrayImage, EdgeImage, 40, 255);

    // Finding Lines in the image
    std::vector<cv::Vec4i> Lines;
    cv::HoughLinesP(EdgeImage, Lines, 1, CV_PI / 180, 50, 10, 15);

    // Check if lines found and exit if not.
    if (Lines.empty()) 
    {
        std::cerr << "Not enough lines found in the image for Vanishing Point detection." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Filtering Lines wrt angle
    return FilterLines(Lines);
}

std::vector<double> GetVanishingPoint(const std::vector<std::vector<double>>& Lines) 
{
    // We will apply RANSAC inspired algorithm for this. We will take combination
    // of 2 lines one by one, find their intersection point, and calculate the
    // total error(loss) of that point. Error of the point means root of sum of
    // squares of distance of that point from each line.
    std::vector<double> VanishingPoint(2, 0.0);
    double MinError = 1e9;

    for (size_t i = 0; i < Lines.size(); ++i) 
    {
        for (size_t j = i + 1; j < Lines.size(); ++j) 
        {
            double m1 = Lines[i][4], c1 = Lines[i][5];
            double m2 = Lines[j][4], c2 = Lines[j][5];

            if (m1 != m2) {
                double x0 = (c1 - c2) / (m2 - m1);
                double y0 = m1 * x0 + c1;

                double err = 0.0;
                for (const auto& Line : Lines) 
                {
                    double m = Line[4], c = Line[5];
                    double m_ = (-1.0 / m);
                    double c_ = y0 - m_ * x0;

                    double x_ = (c - c_) / (m_ - m);
                    double y_ = m_ * x_ + c_;

                    double l = std::sqrt(std::pow(y_ - y0, 2) + std::pow(x_ - x0, 2));

                    err += l * l;
                }

                err = std::sqrt(err);

                if (MinError > err) 
                {
                    MinError = err;
                    VanishingPoint = {x0, y0};
                }
            }
        }
    }

    return VanishingPoint;
}

int main() 
{
    std::vector<cv::Mat> Images = ReadImage("InputImages");

    for (const auto& Image : Images) 
    {
        // Getting the lines from the image
        std::vector<std::vector<double>> Lines = GetLines(Image);

        // Get vanishing point
        std::vector<double> VanishingPoint = GetVanishingPoint(Lines);

        // Checking if vanishing point found
        if (VanishingPoint.empty()) 
        {
            std::cerr << "Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point." << std::endl;
            continue;
        }

        // Drawing lines and vanishing point
        cv::Mat OutputImage = Image.clone();
        for (const auto& Line : Lines) 
        {
            cv::line(OutputImage, cv::Point(Line[0], Line[1]), cv::Point(Line[2], Line[3]), cv::Scalar(0, 255, 0), 2);
        }
        cv::circle(OutputImage, cv::Point(VanishingPoint[0], VanishingPoint[1]), 10, cv::Scalar(0, 0, 255), -1);

        // Showing the final image
        cv::imshow("OutputImage", OutputImage);
        cv::waitKey(0);
    }

    return 0;
}

