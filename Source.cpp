#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>
using namespace std;
using namespace cv;
void getStatData(Mat src, int* hist, float* ave, float* var, bool isPrint) {
	float average = 0, variance = 0;
	int channels = src.channels();
	int nRows = src.rows;
	int nCols = src.cols * channels;
	if (src.isContinuous()) {//是否在内存中连续存储，若连续则读一大长条内存就能读入整张图片
		nCols *= nRows;
		nRows = 1;
	}
	int i, j;
	uchar* p;
	for (i = 0; i < nRows; ++i) {
		p = src.ptr<uchar>(i);
		for (j = 0; j < nCols; ++j) {
			hist[p[j]]++;
		}
	}
	int max = *max_element(hist, hist + 256);
	Mat histgraph = Mat::zeros(300, 512, CV_8U);
	for (i = 0; i < 256; ++i) {
		if (isPrint) {
			for (j = histgraph.rows - ((double)hist[i] / max * (histgraph.rows - 1)); j < histgraph.rows; ++j) {
				histgraph.at<uchar>(j, 2 * i) = 255;
				histgraph.at<uchar>(j, 2 * i + 1) = 255;
			}
		}
		average += i * hist[i];
	}

	average /= src.rows * src.cols;
	for (i = 0; i < 256; ++i) {
		variance += hist[i] * (i - average) * (i - average);
	}
	variance /= src.rows * src.cols;
	*ave = average; *var = variance;
	if (isPrint) {
		imshow("Original Image", src);
		imshow("Histgraph", histgraph);
		cout << "图片的全局平均值为" << average << "，方差为" << variance << endl;
	}
}

Mat getlut(int* hist) {
	//积分直方图
	int addArray[256];
	addArray[0] = hist[0];
	for (int i = 1; i < 256; i++) {
		addArray[i] = addArray[i - 1] + hist[i];
	}
	//转换到调色盘
	Mat lut(1, 256, CV_8U);
	uchar* p = lut.ptr();
	for (int i = 0; i < 256; i++) {
		p[i] = addArray[i] / addArray[255] * 255;
	}
	return lut;
}

Mat equalization(Mat src) {
	float average, variance; int hist[256], addArray[256];
	getStatData(src, hist, &average, &variance, 0);
	Mat newLookUpTable = getlut(hist);
	Mat res(src.rows, src.cols, src.type());
	LUT(src, newLookUpTable, res);
	return res;
}

int main()
{
	cout << "////////////////////////计算全局直方图及统计值//////////////////////////" << endl;
	Mat image = imread("./test.tif");
	cvtColor(image, image, CV_BGR2GRAY);
	float average, variance;
	double t = (double)getTickCount();
	int hist[256];
	getStatData(image, hist, &average, &variance, 1);
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "此段程序用时为" << t << "s" << endl;
	waitKey();
	Mat result1 = equalization(image);
	imshow("Global equalization", result1);
}