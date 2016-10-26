#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv; 
using namespace std;

static void voronoi( Mat& img, Mat& normalize_img, Subdiv2D& subdiv, dilation_size);
double computeCost(Mat& img, Mat& normalized_img);
static void draw_imgToShow(Mat& img,Mat& input1, Mat& input2);
vector<KeyPoint> cellCenters(Mat &img);
static void compute(vector<Point2f> points, Mat& output, Mat& img );






static void voronoi( Mat& img, Mat& normalize_img, Subdiv2D& subdiv, int dilation_size)
{
	vector<vector<Point2f> > facets;
	vector<Point2f> centers;
	subdiv.getVoronoiFacetList(vector<int>(), facets, centers);
		 
	vector<Point> ifacet;
	vector<vector<Point> > ifacets(1);
	 
	for( size_t i = 0; i < facets.size(); i++ )
	{
		ifacet.resize(facets[i].size());
		for( size_t j = 0; j < facets[i].size(); j++ )
		    ifacet[j] = facets[i][j];
	 
		int color=0;

		fillConvexPoly(img, ifacet, color, 8, 0);
	 
		ifacets[0] = ifacet;
		polylines(img, ifacets, true, 255, 1, CV_AA, 0);
		
	}


	Mat thicken_voronoi;

	Mat element= getStructuringElement(MORPH_RECT,Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
	dilate(img,thicken_voronoi,element);
	threshold(thicken_voronoi,thicken_voronoi,0,255,0);
	distanceTransform(thicken_voronoi,thicken_voronoi,CV_DIST_L1, 5);
	normalize(thicken_voronoi,normalize_img,0,255,NORM_MINMAX,CV_8UC1);
	normalize(thicken_voronoi, img, 0.0, 1.0, NORM_MINMAX);
	
	for (size_t i=0; i<centers.size();i++){
		circle(normalize_img, centers[i], 5, 255, CV_FILLED, CV_AA, 0);
	}
	
	
}

 

double computeCost(Mat& img, Mat& normalized_img){
	Mat result=img.mul(normalized_img);
	double score=sum(result).val[0];
	return score;
	
}

static void outputImg(Mat& img,Mat& input1, Mat& input2){
	vector<cv::Mat> images(3);	
	Mat black = Mat::zeros(input1.rows, input1.cols, CV_8UC1);
	images.at(0)=black;
	images.at(1)=input1;
	images.at(2)=input2;
	merge(images, img);

}

vector<KeyPoint> cellCenters(Mat &img){
	

	vector<KeyPoint> keypoints;
	SimpleBlobDetector::Params params;
	params.minThreshold=30;
	params.maxThreshold=256;
	params.filterByArea=true;
	params.minArea=1000;
	params.maxArea=1000*1000;
	params.filterByCircularity=false;
	params.filterByColor=false;
	params.filterByConvexity=false;
	params.filterByInertia=false;

	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
	//SimpleBlobDetector detector(params);
	//Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params); 
	detector->detect(img, keypoints);
	
	return keypoints;

}

static void compute(vector<Point2f> points, Mat& output, Mat& img, int dilation_size ){
	
	
	Size size=img.size();
	Rect rect(0,0,size.width,size.height);
	Subdiv2D subdiv(rect);



	for (auto it =points.begin(); it!=points.end(); ++it){
		subdiv.insert(*it);
	}
	
	
	Mat img_voronoi = Mat::zeros(img.rows, img.cols, CV_8UC1);
	Mat normalize_voronoi = Mat::zeros(img.rows,img.cols,CV_8UC1);
     
   	// Draw Voronoi diagram
    
    voronoi(img_voronoi,normalize_voronoi,subdiv,dilation_size);
	outputImg(output,normalize_voronoi,img);
	
}



int main(int argc, char** argv){
	//initialization
	if ( argc != 2 ){
        printf("usage: Please enter the image\n");
        return -1;
    	}

	char* filename = argv[1];
	Mat img=imread(filename,CV_LOAD_IMAGE_UNCHANGED);
	if(img.empty()){
		cout<<"Error: Image cannot be loaded!"<<endl;
		return -1;
	}
	
	//extract channel for nucleus and surface
	Mat nucleus(img.rows,img.cols,CV_8UC1);
	Mat surface(img.rows,img.cols,CV_8UC1);
	Mat green(img.rows,img.cols,CV_8UC1);
	extractChannel(img,nucleus,0);
	extractChannel(img,surface,2);
	extractChannel(img,green,1);
	Mat surface_float;
	surface.convertTo(surface_float, CV_32FC1);
	



	//detect the nucleus
	vector<KeyPoint> keypoints;
	keypoints=cellCenters(nucleus);
	int numPoints=keypoints.size();
	//cout << "number of cells = "<< endl << " "  << numPoints << endl;
	
	vector<Point2f> points;
	
	
	for (int i=0; i<keypoints.size(); ++i){
		points.push_back(keypoints[i].pt);
	}


	//double score=get_score(surface_float,img_voronoi);
	
	int edge_width = 10;
	
	auto selected = points.begin();
	Mat output;
	Mat background(img.rows,img.cols,CV_8UC3);
	bool recompute(true);
	int myKey;
	nameWindow("My cells")
	createTrackbar("optimizer edge width", "My cells", &edge_width, 100, on_edge_width_trackbar,  );
	do {
		if (recompute) {
			compute(points, background, surface,dilation_size);
			recompute = false;
		}
		background.copyTo(output);
		circle(output, *selected, 20, Scalar(255,255,255));
		imshow("Voronoi Diagram", output);
		
		
		myKey = waitKey();
		
		switch (myKey) {
			case 1048585: { //tab
				if (selected == (--points.end())) {
					selected = points.begin();
				}
				else
					++selected;
				break;
			}
			case 1113938: {//up
				selected->y -= 2;
				recompute = true;
				break;
			}
			case 1113940:{
				selected->y += 2;
				recompute = true;
				break;
			}
			case 1113937:{
				selected->x -= 2;
				recompute = true;
				break;
			}
			case 1113939:{
				selected->x += 2;
				recompute = true;
				break;
			}
			
			default:
			break;						
		}
			
	} while (myKey != 1048689);

	return 0;
}
