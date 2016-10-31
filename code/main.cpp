#include "opencv2/core/core.hpp"
#include "opencv2/core/optim.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv; 
using namespace std;

void onEdgeGradientWidthChanged(int value, void *userData);

class CellSegmenter;

struct ScoreFunction: public cv::MinProblemSolver::Function {
	CellSegmenter* cellSegmenter;
	
	ScoreFunction(CellSegmenter* cellSegmenter): cellSegmenter(cellSegmenter) {} 
	virtual double calc (const double *x) const;
	virtual int getDims () const;
};

class CellSegmenter {
	public:
		Mat background;
		Mat backgroundForScoring;
		vector<Point2f> centers;
		
		vector<Point2f>::iterator selected;
		int edge_gradient_width;
		
		Mat output;
		Mat preOutput;
		
		CellSegmenter(const char* filename):
			edge_gradient_width(40) {
			Mat img = imread(filename,CV_LOAD_IMAGE_UNCHANGED);
			if (img.empty())
				throw invalid_argument("Error: Image cannot be loaded!");
			
			Mat nucleus(img.rows, img.cols, CV_8UC1);
			extractChannel(img, nucleus, 0);
			
			SimpleBlobDetector::Params params;
			params.minThreshold = 30;
			params.maxThreshold = 256;
			params.filterByArea = true;
			params.minArea = 1000;
			params.maxArea = 1000*1000;
			params.filterByCircularity = false;
			params.filterByColor = false;
			params.filterByConvexity = false;
			params.filterByInertia = false;

			cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
			//SimpleBlobDetector detector(params);
			//Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params); 
			vector<KeyPoint> keypoints;
			detector->detect(nucleus, keypoints);
			for (size_t i = 0; i<keypoints.size(); ++i)
				centers.push_back(keypoints[i].pt);
			selected = centers.begin();
			extractChannel(img, background, 2);
			background.convertTo(backgroundForScoring, CV_32F);
		}
		
		Mat computeVoronoiImage(const vector<Point2f>& points) {
			Size size = background.size();
			Rect rect(0, 0, size.width, size.height);
			Subdiv2D subdiv(rect);
			
			for (auto it = points.begin(); it != points.end(); ++it)
				subdiv.insert(*it);
			
			vector<vector<Point2f>> facets;
			vector<Point2f> facetCenters;
			subdiv.getVoronoiFacetList(vector<int>(), facets, facetCenters);
			vector<Point> ifacet;
			vector<vector<Point>> ifacets(1);
			Mat voronoi = Mat::zeros(background.rows, background.cols, CV_8UC1);
			for (size_t i = 0; i < facets.size(); i++) {
				ifacet.resize(facets[i].size());
				for (size_t j = 0; j < facets[i].size(); j++)
					ifacet[j] = facets[i][j];
			 
				int color=0;

				fillConvexPoly(voronoi, ifacet, color, 8, 0);
			 
				ifacets[0] = ifacet;
				polylines(voronoi, ifacets, true, 255, 1, CV_AA, 0);
			}
			return voronoi;		
		}
		
		Mat computeEdgeGradiant(const Mat& voronoi) {
			Mat thicken, edgeGradientForScoring;
			Mat element = getStructuringElement(	
				MORPH_RECT, Size(2*edge_gradient_width + 1, 2*edge_gradient_width+1),
                Point(edge_gradient_width, edge_gradient_width)
            );
			
			dilate(voronoi, thicken, element);
			threshold(thicken, thicken, 0, 255, 0);
			distanceTransform(thicken, thicken, CV_DIST_L1, 5);
			normalize(thicken, edgeGradientForScoring, 0.0, 1.0, NORM_MINMAX);
			edgeGradientForScoring = edgeGradientForScoring * (1/norm(edgeGradientForScoring, NORM_L1));
			cout << computeScore(edgeGradientForScoring) << endl;
			return edgeGradientForScoring;
		}	
			
		Mat computeVoronoiToShow(const Mat& edgeGradientForScoring) {
			Mat thickenToShow;
			normalize(edgeGradientForScoring, thickenToShow, 0, 255, NORM_MINMAX, CV_8UC1);
			return thickenToShow;
		}
		
		void computeOutputImage() {
			vector<cv::Mat> images(3);	
			Mat black = Mat::zeros(background.rows, background.cols, CV_8UC1);
			Mat input = computeVoronoiToShow(computeEdgeGradiant(computeVoronoiImage(centers)));
			
			for (size_t i=0; i<centers.size(); i++)
				circle(input, centers[i], 5, 255, CV_FILLED, CV_AA, 0);
			
			images.at(0) = black;
			images.at(1) = input;
			images.at(2) = background;
			merge(images, preOutput);
		}
		
		double computeScore(const Mat& edgeGradientForScoring) {
			return backgroundForScoring.dot(edgeGradientForScoring);
		}
		
		vector<Point2f> buildSafePointVector(const double *data) {
			vector<Point2f> points;
			const Size size = background.size();
			for (size_t i=0; i<centers.size(); i++) {
				float x = *data++;
				float y = *data++;
				x = std::max<float>(x, 0);
				x = std::min<float>(x, size.width-1);
				y = std::max<float>(y, 0);
				y = std::min<float>(y, size.height-1);
				points.push_back({x, y});
			}
			return points;
		}
		
		void setCenters(const Mat& linearArray) {
			const Size size = background.size();
			for (size_t i=0; i<centers.size(); i++) {
				//centers[i].x = linearArray.at<double>(2*i);
				//centers[i].y = linearArray.at<double>(2*i + 1);
				centers[i].x = std::min<float>(size.width-1, std::max<float>(0, linearArray.at<double>(2*i)));
				centers[i].y = std::min<float>(size.height-1, std::max<float>(0, linearArray.at<double>(2*i + 1)));
			}	
		}
		
		Mat mat1DFromCenter() const {
			Mat_<double> value(centers.size() * 2, 1);
			for (int i=0; i<value.rows; i += 2) {
				value.at<double>(i) = centers[i/2].x; 
				value.at<double>(i+1) = centers[i/2].y;
			}
			return value;
		}
		
		// TODO: be smarter at handling values outside image bounds to have better optmization properties
		double computeScore(const double *data) {
			return computeScore(computeEdgeGradiant(computeVoronoiImage(buildSafePointVector(data))));
		}
		
		void displayImage() {
			Mat output;
			preOutput.copyTo(output);
			circle(output, *selected, 20, Scalar(255,255,255));
			imshow("My cells", output);
		}
		
		void run() {
			namedWindow("My cells", 1);
			createTrackbar("optimizer edge gradient width", "My cells", &edge_gradient_width, 300, onEdgeGradientWidthChanged,  this);
			
			int myKey;
			bool recompute(true);
			do {
				if (recompute) {
					computeOutputImage();
					recompute = false;
				}
				
				displayImage();
		
				myKey = waitKey();
				//cout << myKey << endl;
				switch (myKey) {
					case 1048585: { //tab
						if (selected == (--centers.end())) {
							selected = centers.begin();
						}
						else
							++selected;
						break;
					}
					case 1048695: { //up
						selected->y -= 2;
						recompute = true;
						break;
					}
					case 1048691: {
						selected->y += 2;
						recompute = true;
						break;
					}
					case 1048673: {
						selected->x -= 2;
						recompute = true;
						break;
					}
					case 1048676: {
						selected->x += 2;
						recompute = true;
						break;
					}
					case 1048690: { // r
						break;
					}
					case 1048687: { // o
						cout << "Optimizing" << endl;
						cv::Ptr<cv::DownhillSolver> solver = cv::DownhillSolver::create(
							new ScoreFunction(this),
							Mat_<double>::ones(centers.size() * 2, 1) * 5,
							TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 10000, 0.0001) 
						);
						Mat x(mat1DFromCenter());
						solver->minimize(x);
						setCenters(x);
						//cout << x << endl;
						cout << "done" << endl;
						recompute = true;
						break;
					}
					default:
					break;						
				}
			
			} while (myKey != 1048689);
		}	
};

void onEdgeGradientWidthChanged(int value, void *userData) {
	CellSegmenter* cellSegmenter(reinterpret_cast<CellSegmenter*>(userData));
	cellSegmenter->computeOutputImage();
	cellSegmenter->displayImage();
}

double ScoreFunction::calc (const double *x) const {
/*	const double* xp(x);
	for (int i = 0; i < getDims(); ++i) {
		cerr << *xp++ << " ";
		if (i % 2 == 1)
			cerr << endl;
	}
	cerr << endl; */
	const double score(cellSegmenter->computeScore(x));
	return -score;
}

int ScoreFunction::getDims () const {
	return cellSegmenter->centers.size() * 2;
}

int main(int argc, char** argv){
	//initialization
	if ( argc != 2 ){
        printf("usage: Please enter the image\n");
        return -1;
    }

	char* filename = argv[1];
	CellSegmenter mycells(filename);
	mycells.run();
	
	return 0;
}
