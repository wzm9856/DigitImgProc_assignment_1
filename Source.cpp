#include <opencv2/opencv.hpp>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <iostream>

using namespace std;
using namespace cv;

void getStatData(int* hist, float* ave, float* var) {
	float average = 0, variance = 0;
	int size = 0;
	for (int i = 0; i < 256; i++) {
		average += i * hist[i];
		size += hist[i];
	}
	average /= size;
	for (int i = 0; i < 256; ++i) {
		variance += hist[i] * (i - average) * (i - average);
	}
	variance /= size;
	*ave = average; *var = variance;
}

void getGlobalHist(Mat src, int* hist) {
	int nRows = src.rows;
	int nCols = src.cols;
	if (src.isContinuous()) {//�Ƿ����ڴ��������洢�����������һ�����ڴ���ܶ�������ͼƬ
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
}

void printHist(Mat src, int* hist){
	int max = *max_element(hist, hist + 256);
	Mat histgraph = Mat::zeros(300, 512, CV_8U);
	for (int i = 0; i < 256; ++i) {
		for (int j = histgraph.rows - ((double)hist[i] / max * (histgraph.rows - 1)); j < histgraph.rows; ++j) {
			histgraph.at<uchar>(j, 2 * i) = 255;
			histgraph.at<uchar>(j, 2 * i + 1) = 255;
		}
	}
	imshow("Original Image", src);
	imshow("Histgraph", histgraph);
}

void interHist(int* hist, int* addedHist, int dst = 255) {
	//����ֱ��ͼ
	addedHist[0] = hist[0];
	for (int i = 1; i < dst + 1; i++) {
		addedHist[i] = addedHist[i - 1] + hist[i];
	}
}

Mat global_equalize(Mat src) {
	//������ȫ��ֱ��ͼ���⻯����
	int hist[256] = { 0 }, addedHist[256] = { 0 };
	getGlobalHist(src, hist);
	//printHist(src, hist);
	interHist(hist, addedHist);
	Mat newLookUpTable(1, 256, CV_8UC1);
	uchar* p = newLookUpTable.ptr();
	for (int i = 0; i < 256; i++) {
		p[i] = 255.0 * addedHist[i] / addedHist[255];
	}
	Mat res(src.rows, src.cols, src.type());
	LUT(src, newLookUpTable, res);
	return res;
}

Mat local_equalization(Mat src, int size) {
	int nRows = src.rows - size + 1;    //298...
	int nCols = src.cols - size + 1;    //252...
	int hist[256] = { 0 }, addedHist[256] = { 0 };
	float x, y;
	Mat equalizedMat(nRows, nCols, CV_8U);
	uchar *p, *q;
	for (int j = 0; j < nCols; j++) {
		memset(hist, 0, 256 * sizeof(int));
		memset(addedHist, 0, 256 * sizeof(int));
		for (int i = 0; i < nRows; i++) { //��ͼ���i��j��
			/////////////////////////////
			if (i == 0) {
				for (int a = 0; a < size; a++) {
					p = src.ptr<uchar>(a);
					for (int b = 0; b < size; b++) {
						hist[p[j + b]]++;
						//j��׼����������ص��У�bΪƫ������ֱ�����������
					}
				}
			}
			else {  //��������ƽ�Ƶķ���
				p = src.ptr<uchar>(i - 1);
				q = src.ptr<uchar>(i + size - 1);
				for (int a = 0; a < size; a++) {
					hist[p[j + a]]--;
					hist[q[j + a]]++;
				}
			}
			p = src.ptr<uchar>(i + (size - 1) / 2);
			q = equalizedMat.ptr<uchar>(i);
			interHist(hist, addedHist);
			q[j] = 255.0 * addedHist[p[j + (size - 1) / 2]] / size / size;
			///////////////////////////////����������ѡ������ȽϿ죩//������Ļ������memsetҲҪע��
			//memset(hist, 0, 256 * sizeof(int));
			//memset(addedHist, 0, 256 * sizeof(int));
			//Rect r(j, i, size, size);
			//Mat small = src(r);
			//getStatData(small, hist, &x, &y);
			//p = small.ptr<uchar>((size + 1) / 2);
			//q = equalizedMat.ptr<uchar>(i);
			//interHist(hist, addedHist, p[(size + 1) / 2]);
			//q[j] = 255.0 * addedHist[p[(size + 1) / 2]] / size / size;
			////////////////////////////////
		}
	}
	return equalizedMat;
}

Mat local_enhancement(Mat src, int size, float C, float K0, float K1, float K2, float K3) {
	float Gaverage, Gvariance, Laverage, Lvariance;
	int hist[256] = { 0 }, addedHist[256] = { 0 };
	getGlobalHist(src, hist);
	getStatData(hist, &Gaverage, &Gvariance);
	int nRows = src.rows - size + 1;    //298...
	int nCols = src.cols - size + 1;    //252...
	float x, y;
	Mat enhancedMat(nRows, nCols, CV_8U);
	uchar* p, * q;
	for (int j = 0; j < nCols; j++) {
		memset(hist, 0, 256 * sizeof(int));
		memset(addedHist, 0, 256 * sizeof(int));
		for (int i = 0; i < nRows; i++) { //��ͼ���i��j��
			/////////////////////////////
			if (i == 0) {
				for (int a = 0; a < size; a++) {
					p = src.ptr<uchar>(a);
					for (int b = 0; b < size; b++) {
						hist[p[j + b]]++;
						//j��׼����������ص��У�bΪƫ������ֱ�����������
					}
				}
			}
			else {  //��������ƽ�Ƶķ���
				p = src.ptr<uchar>(i - 1);
				q = src.ptr<uchar>(i + size - 1);
				for (int a = 0; a < size; a++) {
					hist[p[j + a]]--;
					hist[q[j + a]]++;
				}
			}
			p = src.ptr<uchar>(i + (size - 1) / 2);
			q = enhancedMat.ptr<uchar>(i);
			getStatData(hist, &Laverage, &Lvariance);
			if (Laverage > K0 * Gaverage && Laverage<K1 * Gaverage && Lvariance>K2 * Gvariance && Lvariance < K3 * Gvariance){
				q[j] = C * p[j + (size - 1) / 2];
			}
			else{
				q[j] =p[j + (size - 1) / 2];
			}
		}
	}
	return enhancedMat;
}
int main()
{
	cout << "////////////////////////����ȫ��ֱ��ͼ��ͳ��ֵ//////////////////////////" << endl;
	Mat image = imread("./test.tif");
	cvtColor(image, image, CV_BGR2GRAY);
	float average, variance;
	double t = (double)getTickCount();
	int hist[256] = { 0 };
	getGlobalHist(image, hist);
	getStatData(hist, &average, &variance);
	cout << "ͼƬ��ȫ��ƽ��ֵΪ" << average << "������Ϊ" << variance << endl;
	printHist(image, hist);
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "�˶γ�����ʱΪ" << t << "s" << endl;
	waitKey(0);
	cout << "///////////////////////////ȫ��ֱ��ͼ���⻯/////////////////////////////" << endl;
	t = (double)getTickCount();
	Mat result1 = global_equalize(image);
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "�˶γ�����ʱΪ" << t << "s" << endl;
	imshow("Global equalization", result1);
	waitKey(0);
	cout << "/////////////////////////////�ֲ�ͼ����ǿ1/////////////////////////////" << endl;
	t = (double)getTickCount();
	Mat result2 = local_equalization(image, 5);
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "�˶γ�����ʱΪ" << t << "s" << endl;
	imshow("Local equalization", result2);
	waitKey(0);
	cout << "/////////////////////////////�ֲ�ͼ����ǿ2/////////////////////////////" << endl;
	t = (double)getTickCount();
	Mat result3 = local_enhancement(image, 5, 3, 0, 0.3, 0, 0.3);
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "�˶γ�����ʱΪ" << t << "s" << endl;
	imshow("Local enhancement", result3);
	waitKey(0);
}