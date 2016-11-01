/*
 *  HandDetector.cpp
 *  TrainHandModels
 *
 *  Created by Kris Kitani on 4/1/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include "HandDetector.hpp"

void HandDetector::loadMaskFilenames(string msk_prefix)
{
	string cmd = "ls " + msk_prefix + " > maskfilename.txt";
	system(cmd.c_str());
	
	ifstream fs;
	fs.open("maskfilename.txt");
	string val;
	while(fs>>val) _filenames.push_back(val);
}

void HandDetector::clusterImages(string basename, string img_prefix, string msk_prefix, string model_prefix, string globfeat_prefix, string feature_set, int num_clusters, int width)
{
	cout << "In HandDetector::clusterImages" << endl;

	stringstream ss;
	string cmd = "ls " + img_prefix + " > trainImagefilename.txt";
	system(cmd.c_str());
	
	ifstream fs;
	fs.open("trainImagefilename.txt");
	string val;

	cout << "Extracting global HSV features" << endl;
	int f = 101; //used to be -1
	
	Mat p = Mat::zeros(0, 64, CV_32F);
	Mat labels;
	Mat centers;

	while(fs>>val){

		ss.str("");
		ss << img_prefix << setw(8) << setfill('0') << f << ".jpg";

		// Load the image
		Mat color_img = imread(ss.str(),1);
		if(!color_img.data) cout << "Missing: " << ss.str() << endl;
		if(!color_img.data) break;
		
		// resize the image
		_img_width = (float)width;
		_img_height = color_img.rows * (_img_width/color_img.cols);
		_img_size = Size(_img_width,_img_height);
		resize(color_img, color_img, _img_size);

		// Calculate the global features
		Mat globfeat;
		computeColorHist_HSV(color_img,globfeat);
		p.push_back(globfeat);

		// Store the global features in files. They'll be used during training
		ss.str("");
		ss << globfeat_prefix << "hsv_histogram_" << setw(8) << setfill('0') << f << ".xml";
		cout << "Writing global feature for: " << ss.str() << endl;
		
		FileStorage fs;
		fs.open(ss.str(),FileStorage::WRITE);
		fs << "globfeat" << globfeat;
		fs.release();

		f++;
	}
	fs.close();

	// cluster the images
	cout << "Starting clustering now" << endl;
	cv::kmeans(p, num_clusters, labels, cv::TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
	
	// here you save the knn to the cluster centers. At test time, the images will be compared to the top k cluster centers
	ss.str("");
	ss << "clusters.txt";
	cout << "Storing cluster centers" << ss.str() << endl;

	FileStorage file;
	file.open(ss.str(),FileStorage::WRITE);
	file << "clusters" << centers;
	file.release();

	// store images according to the clusters they belong to
	fs.open("trainImagefilename.txt");
	for (int i = 0; i< num_clusters; i++){
		ss.str("");
		ss << "mkdir -p " << globfeat_prefix << "cluster_" << i;
		system(ss.str().c_str());
	}

	int i = 0;
	std::map<string,int> imageToCluster;

	f = 101;
	while(fs>>val){
		// store the relavant cluster with the file
		int idx = labels.at<int>(i,0);
		imageToCluster[val] = idx;
		i++;

		// move the global feature files into appropriate clusters.
		stringstream cmd;
		cmd.str("");
		cmd << "mv " << globfeat_prefix << "hsv_histogram_" << setw(8) << setfill('0') << f << ".xml "<< globfeat_prefix << "cluster_" << idx << "/hsv_histogram_" << setw(8) << setfill('0') << f << ".xml"; 
		system(cmd.str().c_str());
		cout << cmd.str() << endl;
		f++;
	}

	//train one classifier per cluster
	for (int i = 0; i< num_clusters; i++){
		LcFeatureExtractor	_extractor;
		LcRandomTreesR		_classifier;
		_feature_set = feature_set;
		_extractor.set_extractor(feature_set);
		Mat clusterDesc;
		Mat clusterLab;

		// instead of this we can just cd into the directory and pick up those files
		for(map<string,int>::iterator it = imageToCluster.begin(); it != imageToCluster.end(); ++it) {
		  if (it->second == i){
		  		
		  		// load the mask image
		  		ss.str("");
				ss << msk_prefix << val; // here the file extensions may be different!
				Mat mask_img = imread(ss.str(),0);
				if(!mask_img.data) continue;
				if(countNonZero(mask_img)==0) cout << "Skipping: " << ss.str() << endl;
				if(countNonZero(mask_img)==0) continue;

				// load the color image
				ss.str("");
				ss << img_prefix << val;
				Mat color_img = imread(ss.str(),1);
				if(!color_img.data) cout << "Missing: " << ss.str() << endl;
				if(!color_img.data) break;
				
				// resize and convert images for proper format
				_img_height = color_img.rows * (_img_width/color_img.cols);
				_img_size = Size(_img_width,_img_height);
				resize(color_img,color_img,_img_size);
				vector<KeyPoint> kp;
				mask_img.convertTo(mask_img,CV_8UC1);

				// extract the features and labels for this image
				Mat desc;
				Mat lab;
				_extractor.work(color_img, desc, mask_img, lab, 1, &kp);
				clusterDesc.push_back(desc);
				clusterLab.push_back(lab);
				cout << desc.rows << endl;
				cout << desc.cols << endl;
		  }
		}
		cout << endl;
		cout << clusterDesc.rows << endl;
		cout << clusterDesc.cols << endl;

		// train and save cluster specific classifier
		_classifier.train(clusterDesc,clusterLab);
		ss.str("");
		ss << model_prefix << "model_" + basename + "_"+ feature_set + "_" << i;
		_classifier.save(ss.str());

	}
}

void HandDetector::trainModels(string basename, string img_prefix,string msk_prefix,string model_prefix,string globfeat_prefix, string feature_set, int max_models, int width)
{
	cout << "HandDetector::trainModels()" << endl;

	stringstream ss;
	
	ss.str("");
	ss << "mkdir -p " + model_prefix;
	system(ss.str().c_str());
	
	ss.str("");
	ss << "mkdir -p " + globfeat_prefix;
	system(ss.str().c_str());
	
	
	_img_width = (float)width;
	
	LcFeatureExtractor	_extractor;
	LcRandomTreesR		_classifier;
	// LcSVM _classifier;

	_feature_set = feature_set;
	_extractor.set_extractor(feature_set);

	//VideoCapture cap(vid_filename);
	//Mat color_img;
	
	int f = 100; //used to be -1
	int k = 0;
	
	while(k<max_models)
	{
		f++;
		
		//////////////////////////////////////////
		//										//
		//		 LOAD IMAGE AND MASK			//
		//										//
		//////////////////////////////////////////
		
		ss.str("");
		ss << msk_prefix << setw(8) << setfill('0') << f << ".jpg";
		
		Mat mask_img = imread(ss.str(),0);

		if(!mask_img.data) continue;
		if(countNonZero(mask_img)==0) cout << "Skipping: " << ss.str() << endl;
		if(countNonZero(mask_img)==0) continue;
		else cout << "\n  Loading: " << ss.str() << endl;
		
		
		ss.str("");
		ss << img_prefix << setw(8) << setfill('0') << f << ".jpg";
		Mat color_img = imread(ss.str(),1);
		if(!color_img.data) cout << "Missing: " << ss.str() << endl;
		if(!color_img.data) break;
		
		_img_height = color_img.rows * (_img_width/color_img.cols);
		_img_size = Size(_img_width,_img_height);
		
		resize(color_img,color_img,_img_size);
		
		int VISUALIZE = 0;
		if(VISUALIZE)
		{
			imshow("src",color_img);
			imshow("mask",mask_img);
		
			Mat dsp;
			cvtColor(mask_img,dsp,CV_GRAY2BGR);
			addWeighted(dsp,0.5,color_img,0.5,0,dsp);
			imshow("blend",dsp);
			waitKey(1);
		}
		
		
		//////////////////////////////////////////
		//										//
		//		 EXTRACT/SAVE Histogram			//
		//										//
		//////////////////////////////////////////
		
		Mat globfeat;
		computeColorHist_HSV(color_img,globfeat);
		
		ss.str("");
		ss << globfeat_prefix << "hsv_histogram_" << k << ".xml";
		cout << "Writing global feature: " << ss.str() << endl;
		
		FileStorage fs;
		fs.open(ss.str(),FileStorage::WRITE);
		fs << "globfeat" << globfeat;
		fs.release();
		
		
		//////////////////////////////////////////
		//										//
		//		  TRAIN/SAVE CLASSIFIER			//
		//										//
		//////////////////////////////////////////
		
		Mat desc;
		Mat lab;
		vector<KeyPoint> kp;

		mask_img.convertTo(mask_img,CV_8UC1);
		_extractor.work(color_img, desc, mask_img, lab,1, &kp);
		_classifier.train(desc,lab);
		
		ss.str("");
		ss << model_prefix << "model_" + basename + "_"+ feature_set + "_" << k;
		_classifier.save(ss.str());
		
		k++;
		cout << k << endl;
	}
	
}

void HandDetector::testInitialize(string model_prefix,string globfeat_prefix, string feature_set, int knn, int width)
{
	
	stringstream ss;
	
	_img_width = (float)width;
	
	
	//////////////////////////////////////////
	//										//
	//		     FEATURE EXTRACTOR			//
	//										//
	//////////////////////////////////////////
	
	cout << "set extractor" << endl;
	_feature_set = feature_set;
	_extractor.set_extractor(_feature_set);
	
	
	//////////////////////////////////////////
	//										//
	//		       LOAD CLASSIFIERS			//
	//										//
	//////////////////////////////////////////
	int num_models;
	{ // This will only work on linux systems
		string cmd;
		cmd = "find " + model_prefix + " -name *.xml -print > modelfilename.txt";
		cout << cmd << endl;
		system(cmd.c_str());
		
		ifstream fs;
		vector<string> filenames;
		fs.open("modelfilename.txt");
		filenames.clear();
		
		string val;
		while(fs>>val) filenames.push_back(val);
		
		num_models = (int)filenames.size();
		
		cout << "Loading classifiers " << num_models << endl;
		_classifier = vector<LcRandomTreesR>(num_models);
		// _classifier = vector<LcSVM>(num_models);
		
		for(int i=0;i<num_models;i++)
		{
			cout << "Loading model "<< filenames[i] << endl;
			_classifier[i].load_full(filenames[i]);
		}
	}
	
	
	//////////////////////////////////////////
	//										//
	//		       LOAD Histogram			//
	//										//
	//////////////////////////////////////////
	
	{
		// If no clustering. Then you use the global features to match, else you use the cluster centers
		/*string cmd;
		cmd = "find " + globfeat_prefix + " -name *.xml -print > globfeatfilename.txt";
		cout << cmd << endl;
		system(cmd.c_str());
		
		ifstream fs;
		vector<string> filenames;
		fs.open("globfeatfilename.txt");
		filenames.clear();
		
		string val;
		while(fs>>val) filenames.push_back(val);
		
		int num_models = (int)filenames.size();
		
		for(int i=0;i<num_models;i++)
		{
			Mat globalfeat;
			
			cout << filenames[i] << endl;
			FileStorage fs;
			fs.open(filenames[i],FileStorage::READ);
			fs["globfeat"] >> globalfeat;
			fs.release();
			
			_hist_all.push_back(globalfeat);
			
		}*/

		// if clustering, we've already set the _hist_all in the clusterImages function
		FileStorage fs;
		fs.open("clusters.txt", FileStorage::READ);
		fs["clusters"] >> _hist_all;
		fs.release();
		cout << _hist_all.size() << endl;

	}
	cout << _hist_all.size() << endl;

	if(_hist_all.rows != (int)_classifier.size()) cout << "ERROR: Number of classifers doesn't match number of global features.\n";
	
	//////////////////////////////////////////
	//										//
	//		       KNN CLASSIFIER			//
	//										//
	//////////////////////////////////////////
	
	cout << "Building FLANN search structure...";
	_indexParams = *new flann::KMeansIndexParams;	
	_searchtree  = *new flann::Index(_hist_all, _indexParams);
	_knn		= knn;						//number of nearest neighbors 
	_indices	= vector<int> (_knn); 
	_dists		= vector<float> (_knn);
	cout << "done." << endl;

}

void HandDetector::test(Mat &img)
{
	Mat tmp = Mat();
	test(img,tmp,1);
}

void HandDetector::test(Mat &img, Mat &dsp)
{
	int num_models = 1;
	int step_size = 1;
	test(img,dsp,num_models,step_size);
}

void HandDetector::test(Mat &img, int num_models)
{
	Mat tmp = Mat();
	int step_size = 1;
	test(img,tmp,num_models,step_size);
}

void HandDetector::test(Mat &img, Mat &dsp, int num_models)
{
	int step_size = 1;
	test(img,dsp,num_models,step_size);
}

void HandDetector::test(Mat &img, int num_models, int step_size)
{	
	Mat tmp = Mat();
	test(img,tmp,num_models,step_size);
	
}

void HandDetector::test(Mat &img, Mat &dsp, int num_models, int step_size)
{	
	if(num_models>_knn) return;
	
	cout << "In test" << endl;
	_img_height = img.rows * (_img_width/img.cols);
	_img_size   = Size(_img_width,_img_height);
		
	resize(img,img,_img_size);
	
	Mat hist;
	computeColorHist_HSV(img,hist);									// extract hist
	
	_searchtree.knnSearch(hist,
						   _indices, _dists, 
						   _knn, flann::SearchParams(4));			// probe search
	Mat lab;
	if (dsp.data){
		_extractor.work(img,_descriptors, dsp, lab, &_kp);						// every 3rd pixel
	}
	else{
		_extractor.work(img,_descriptors,step_size,&_kp);
	}

	if(!_response_avg.data) _response_avg = Mat::zeros(_descriptors.rows,1,CV_32FC1); 
	else _response_avg *= 0;

	float norm = 0;
	for(int i=0;i<num_models;i++)
	{
		int idx = _indices[i]; // changehere

		if (lab.data){
			cout << "Have labeled data" << endl;
			LcValidator model_output = _classifier[idx].predict(_descriptors,_response_vec, lab); // run classifier
		}
		else
			_classifier[idx].predict(_descriptors,_response_vec);
		
		_response_avg += _response_vec*float(pow(0.9f,(float)i));
		norm += float(pow(0.9f,(float)i));
	}
	
	_response_avg /= norm;
	
	LcValidator avgF1 = LcValidator(_response_avg, lab);
	avgF1.display();

	_sz = img.size();
	_bs = _extractor.bound_setting;	
	rasterizeResVec(_response_img,_response_avg,_kp,_sz,_bs);		// class one
	
	//colormap(_response_img,_raw,1);
	//vector<Point2f> pt;
	//_ppr = postprocess(_response_img,pt);
	//colormap(_ppr,_ppr,1);
	
}

Mat HandDetector::postprocess(Mat &img)
{
	vector<Point2f> pt;
	return postprocess(img,pt);
}

Mat HandDetector::postprocess(Mat &img,vector<Point2f> &pt)
{
	Mat tmp;
	
	GaussianBlur(img,tmp,cv::Size(11,11),0,0,BORDER_REFLECT);
	
	colormap(tmp,_blu,1);	// for visualization (blurred and colormap)
	
	tmp = tmp > 0.04;
	

	/////////////////////////////////////////////////////////////
	// GET CONNECTED COMPONENTS
	
	vector<vector<cv::Point> > co;
	vector<Vec4i> hi;
	
	findContours(tmp,co,hi,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	tmp *= 0;
	
	Moments m;
	//vector<Point2f> pt;
	for(int i=0;i<(int)co.size();i++)
	{
		if(contourArea(Mat(co[i])) < (tmp.rows*tmp.cols*0.01)) continue;
		drawContours(tmp, co,i, CV_RGB(255,255,255), CV_FILLED, CV_AA);
		m = moments(Mat(co[i]));
		pt.push_back(Point2f(m.m10/m.m00,m.m01/m.m00));
	}
	
	return tmp;

}

void HandDetector::rasterizeResVec(Mat &img, Mat&res,vector<KeyPoint> &keypts, cv::Size s, int bs)
{	
    if((img.rows!=s.height) || (img.cols!=s.width) || (img.type()!=CV_32FC1) ) img = Mat::zeros( s, CV_32FC1);
	
	for(int i = 0;i< (int)keypts.size();i++)
    {
		int r = floor(keypts[i].pt.y);
		int c = floor(keypts[i].pt.x);
		img.at<float>(r,c) = res.at<float>(i,0);
	}
}

void HandDetector::colormap(Mat &src, Mat &dst, int do_norm)
{
	
	double minVal,maxVal;
	minMaxLoc(src,&minVal,&maxVal,NULL,NULL);
	
	//cout << "colormap minmax: " << minVal << " " << maxVal << " Type:" <<  src.type() << endl;
	
	Mat im;
	src.copyTo(im);
	
	if(do_norm) im = (src-minVal)/(maxVal-minVal);		// normalization [0 to 1]
	
	Mat mask;	
	mask = Mat::ones(im.size(),CV_8UC1)*255.0;	
	
	compare(im,0.01,mask,CMP_GT);						// one color values greater than X	
	
	
	Mat U8;
	im.convertTo(U8,CV_8UC1,255,0);
	
	Mat I3[3],hsv;
	I3[0] = U8 * 0.85;
	I3[1] = mask;
	I3[2] = mask;
	merge(I3,3,hsv);
	cvtColor(hsv,dst,CV_HSV2RGB_FULL);
	
	
}

void HandDetector::computeColorHist_HSV(Mat &src, Mat &hist)
{
	
	int bins[] = {4,4,4};
    if(src.channels()!=3) exit(1);
    
	//Mat tmp;
    //src.copyTo(tmp);
    
	Mat hsv;
    cvtColor(src,hsv,CV_BGR2HSV_FULL);
    
	int histSize[] = {bins[0], bins[1], bins[2]};
    Mat his;
    his.create(3, histSize, CV_32F);
    his = Scalar(0);   
    CV_Assert(hsv.type() == CV_8UC3);
    MatConstIterator_<Vec3b> it = hsv.begin<Vec3b>();
    MatConstIterator_<Vec3b> it_end = hsv.end<Vec3b>();
    for( ; it != it_end; ++it )
    {
        const Vec3b& pix = *it;
        his.at<float>(pix[0]*bins[0]/256, pix[1]*bins[1]/256,pix[2]*bins[2]/256) += 1.f;
    }
	
    // ==== Remove small values ==== //
    float minProb = 0.01;
    minProb *= hsv.rows*hsv.cols;
    Mat plane;
    const Mat *_his = &his;
	
    NAryMatIterator itt = NAryMatIterator(&_his, &plane, 1);   
    threshold(itt.planes[0], itt.planes[0], minProb, 0, THRESH_TOZERO);
    double s = sum(itt.planes[0])[0];
	
    // ==== Normalize (L1) ==== //
    s = 1./s * 255.;
    itt.planes[0] *= s;
    itt.planes[0].copyTo(hist);
	
	
}
