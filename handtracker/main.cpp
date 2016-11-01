#include <opencv2/opencv.hpp>
#include "FeatureComputer.hpp"
#include "Classifier.h"
#include "LcBasic.h"
#include "HandDetector.hpp"

using namespace std;
using namespace cv;

int main (int argc, char * const argv[])
{
    bool TRAIN_MODEL = 1;           //1 if you are training the models, 0 if you are running the program to predict
    bool TEST_MODEL  = 0;           //0 if you are training the models, 1 if you are running the program to predict
    
    int target_width = 360;			// for resizing the input (small is faster)
    
    // maximum number of image masks that you will use
    // must have the masks prepared in advance
    // only used at training time
    int num_models_to_train = 16; // Here we train one model per image
    int num_clusters = 4;
    
    // number of models used to compute a single pixel response
    // must be less than the number of training models (number of clusters if using clustering or num_models_to_train if not)
    // only used at test time
    int num_models_to_average = 3;
    
    // runs detector on every 'step_size' pixels
    // only used at test time
    // bigger means faster but you lose resolution
    // you need post-processing to get contours
    int step_size = 3;
    
    // Assumes a certain file structure e.g., /root/img/basename/00000000.jpg
    string root = "/home/khushig/claireCVPR/handtrack/";       //replace with path to your Xcode project
    string basename = "GTEA";
    string img_prefix		= root + "img"		+ basename + "/";			// color images
    string msk_prefix		= root + "mask"     + basename + "/";			// binary masks
    string model_prefix		= root + "models"	+ basename + "/";			// output path for learned models
    string globfeat_prefix  = root + "globfeat" + basename + "/";			// output path for color histograms
    
    
    // types of features to use (you will over-fit if you do not have enough data)
    // r: RGB (5x5 patch)
    // v: HSV
    // l: LAB
    // b: BRIEF descriptor
    // o: ORB descriptor
    // s: SIFT descriptor
    // u: SURF descriptor
    // h: HOG descriptor
    // g: Gabor feature
    
    string feature_set = "rvl";
    
    
    if(TRAIN_MODEL)
    {
        msk_prefix = msk_prefix + "train/"
        img_prefix = img_prefix + "train/"

        cout << "Training..." << endl;
        HandDetector hd;
        hd.loadMaskFilenames(msk_prefix);
        hd.clusterImages(basename, img_prefix, msk_prefix, model_prefix, globfeat_prefix, feature_set, num_clusters, target_width);
        //hd.trainModels(basename, img_prefix, msk_prefix, model_prefix,globfeat_prefix,feature_set,num_models_to_train,target_width);
        cout << "Done Training..." << endl;
    }
    
    
    if(TEST_MODEL)
    {

        cout << "Testing..." << endl;
        
        // string root = "/home/khushig/claireCVPR/handtrack/";
        // string basename = "GTEA";
        // string vid_filename		= root + "vid/"		+ basename + ".avi";
        // string model_prefix		= root + "models/"	+ basename + "/";
        // string globfeat_prefix  = root + "globfeat/"+ basename + "/";
        // string feature_set = "rvl";
        
        int num_models_to_average = 2;
        
        msk_prefix = msk_prefix + "test/"
        img_prefix = img_prefix + "test/"
        
        ss.str("");
        ss << img_prefix << "00000101.jpg";
        Mat color_img = imread(ss.str(),1);
        if(!color_img.data) cout << "Missing: " << ss.str() << endl;
        
        stringstream ss;
        ss.str("");
        ss << msk_prefix << "00000101.jpg";

        Mat mask_img = imread(ss.str(),0);
        if(countNonZero(mask_img)==0) cout << "Skipping: " << ss.str() << endl;
        else cout << "\n  Loading: " << ss.str() << endl;

        Mat im = color_img;
        Mat ppr;
        resize(im,im,Size(640,360));

        HandDetector hd;
        hd.testInitialize(model_prefix,globfeat_prefix,feature_set,num_models_to_average,target_width);
        hd.test(im, mask_img, num_models_to_average);
        
        resize(hd._ppr,ppr,im.size(),0,0,INTER_LINEAR);
        addWeighted(im,0.7,ppr,0.3,0,ppr);
        imshow("result:contour",ppr);
        imshow("result:probability",hd._blu);
        waitKey(1);

        /*VideoCapture cap(vid_filename);
        Mat im;
        Mat ppr;
        
        while(1)
        {
            cap >> im; if(!im.data) break;
            cap >> im; if(!im.data) break;
            resize(im,im,Size(640,360));
            hd.test(im,num_models_to_average);
            resize(hd._ppr,ppr,im.size(),0,0,INTER_LINEAR);
            addWeighted(im,0.7,ppr,0.3,0,ppr);
            imshow("result:contour",ppr);
            imshow("result:probability",hd._blu);
            waitKey(1);
            
        }*/
    }
}
