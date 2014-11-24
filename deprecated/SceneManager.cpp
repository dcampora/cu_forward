/* OpenSceneGraph Scratch Project
 * Page 69, beginners guide
 * Create basic shapes with Geometry,
 * more efficient than ShapeDrawable
*/

#include <osg/Geometry>
#include <osg/Geode>
#include <osgViewer/Viewer>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osgDB/ReadFile>

#include "VELOModel.h"
#include "Tools.h"

#include <iostream>

/*
int* h_sensor_hitStarts;
int* h_sensor_hitNums;
int* h_hit_IDs;
*/

extern int* h_no_sensors;
extern int* h_sensor_Zs;
extern int* h_no_hits;
extern float* h_hit_Xs;
extern float* h_hit_Ys;
extern int* h_hit_Zs;

void updateMax(float& max, float value){
	if(value>max) max = value;
}

void updateMin(float& min, float value){
	if(value<min) min = value;
}

void addHits(VELOModel& v){
	float x1=0.0f,x2=0.0f,y1=0.0f,y2=0.0f;

	for(int i=0; i<h_no_hits[0]; ++i){
		v.addHit(osg::Vec3f(h_hit_Xs[i], h_hit_Ys[i], static_cast<float>(h_hit_Zs[i])));

		/*updateMax(x2,h_hit_Xs[i]);
		updateMax(y2,h_hit_Ys[i]);

		updateMin(x1,h_hit_Xs[i]);
		updateMin(y1,h_hit_Ys[i]);*/
	}

	// std::cout << x1 << ", " <<  x2 << ", " << y1 << ", " << y2 << std::endl;
}

void initializeModel(std::string& filename, VELOModel& v, osgViewer::Viewer& viewer)
{	
	// Read file
	char* input;
	int size;
	readFile(filename.c_str(), input, size);

	v = VELOModel(h_no_sensors[0], h_sensor_Zs);
	addHits(v);

	// v.addHit(osg::Vec3f(0.0f, 0.0f, 0.0f));
	// v.addTrack(osg::Vec3f(200.0f,0.0f,0.0f), osg::Vec3f(100.0f, 100.0f, 100.0f) );

	// osg::ref_ptr<osg::Node> model1 = osgDB::readNodeFile("C:\\Program Files\\OpenSceneGraph\\debug\\data\\axes.osgt");
	// v.root->addChild(model1);

	osgViewer::CompositeViewer composite_viewer;

	// osgViewer::Viewer viewer;
	viewer.getCamera()->setClearColor(osg::Vec4(0.9f,0.9f,0.9f,1.0f));
	viewer.setUpViewOnSingleScreen(0);

	osg::ref_ptr<osg::Light> light = new osg::Light;
	light->setDiffuse( osg::Vec4( 1, 1, 1, 1 ) );
	viewer.setLight(light);
	viewer.setLightingMode(osg::View::LightingMode::HEADLIGHT);

	viewer.addEventHandler(new osgViewer::WindowSizeHandler);
	viewer.addEventHandler(new osgViewer::StatsHandler);
	
	osg::ref_ptr<osg::Group> root_scene = new osg::Group;
	root_scene->addChild( v.root.get() );

	viewer.setSceneData( root_scene.get() );


	/* Creating an ortho camera :)
	osg::ref_ptr<osg::Camera> cam = new osg::Camera;
	cam->setClearMask( GL_DEPTH_BUFFER_BIT );
	cam->setRenderOrder( osg::Camera::POST_RENDER );
	cam->setReferenceFrame( osg::Camera::ABSOLUTE_RF);
	
	const osg::Vec3d eye = osg::Vec3d(-1000.0, 0.0, 0.0);
	const osg::Vec3d center = osg::Vec3d(0.0, 0.0, 0.0);
	const osg::Vec3d up = osg::Vec3d(-1000.0, 1.0, 0.0);
	cam->setViewMatrixAsLookAt(eye, center, up);
	cam->addChild( v.root.get() );

	root_scene->addChild( cam.get() );
	// cam->setProjectionMatrixAsOrtho2D(
	*/

	/*
	const osg::Vec3d eye = osg::Vec3d(-200.0, 0.0, 200.0);
	const osg::Vec3d center = osg::Vec3d(0.0, 0.0, 220.0);
	const osg::Vec3d up = osg::Vec3d(-200.0, 1.0, 200.0);

	const osg::Vec3d eye = osg::Vec3d(-200.0, 0.0, 200.0);
	const osg::Vec3d center = osg::Vec3d(0.0, 0.0, 200.0);
	const osg::Vec3d up = osg::Vec3d(-200.0, 1.0, 200.0);
	viewer.getCameraManipulator()->setHomePosition(eye, center, up);
	*/

	// return viewer.run();
}
