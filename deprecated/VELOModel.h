
#include <osg/Geometry>
#include <osg/Geode>
#include <osgViewer/Viewer>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>

#include "definitions.h"

/** Singleton class for HitGeometry */
class HitGeometry : public osg::Geometry {
public:
	HitGeometry(){
		// Vertices, they have an implicit order (as we push them)
		osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
		vertices->push_back( osg::Vec3(0.0f, 0.0f, 0.0f) );
		vertices->push_back( osg::Vec3(1.0f, 0.0f, 0.0f) );
		vertices->push_back( osg::Vec3(1.0f, 1.0f, 0.0f) );
		vertices->push_back( osg::Vec3(0.0f, 1.0f, 0.0f) );

		osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
		normals->push_back( osg::Vec3(0.0f, 0.0f, -1.0f) );

		// Colors
		osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
		colors->push_back( HIT_COLOR );

		// We used only one normal, thus we use "BIND_OVERALL" for it.
		this->setVertexArray( vertices.get() );
		this->setNormalArray( normals.get() );
		this->setNormalBinding( osg::Geometry::BIND_OVERALL );

		// On the other hand, we put a color for each of them, and therefore we use "BIND_PER_VERTEX"
		// FIFO (vector of four elements)
		this->setColorArray( colors.get() );
		this->setColorBinding( osg::Geometry::BIND_OVERALL );

		// Primitive Set: Used GL_QUADS to render the four vertices
		// as quad corners in a counter-clockwise order
		this->addPrimitiveSet( new osg::DrawArrays(GL_QUADS, 0, 4) );
	}
};


/** Singleton class for DetectorGeometry */
class SensorGeometry : public osg::Geometry {
public:
	SensorGeometry(){
		// Vertices, they have an implicit order (as we push them)
		osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
		vertices->push_back( osg::Vec3(0.0f, 0.0f, 0.0f) );
		vertices->push_back( osg::Vec3(SENSOR_SIZE, 0.0f, 0.0f) );
		vertices->push_back( osg::Vec3(SENSOR_SIZE, SENSOR_SIZE, 0.0f) );
		vertices->push_back( osg::Vec3(0.0f, SENSOR_SIZE, 0.0f) );

		osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
		normals->push_back( osg::Vec3(0.0f, 0.0f, -1.0f) );
		// normals->push_back( osg::Vec3(0.0f, 1.0f, 0.0f) );

		// Colors
		osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
		colors->push_back( DETECTOR_COLOR );

		// We used only one normal, thus we use "BIND_OVERALL" for it.
		this->setVertexArray( vertices.get() );
		this->setNormalArray( normals.get() );
		this->setNormalBinding( osg::Geometry::BIND_OVERALL );

		// On the other hand, we put a color for each of them, and therefore we use "BIND_PER_VERTEX"
		// FIFO (vector of four elements)
		this->setColorArray( colors.get() );
		this->setColorBinding( osg::Geometry::BIND_OVERALL );

		// Primitive Set: Used GL_QUADS to render the four vertices
		// as quad corners in a counter-clockwise order
		this->addPrimitiveSet( new osg::DrawArrays(GL_QUADS, 0, 4) );
	}
};

class VELOModel {
private:
	osg::ref_ptr<osg::Geometry> _hitGeometry;
	osg::ref_ptr<osg::Geometry> _sensorGeometry;

	osg::ref_ptr<osg::Geode> _hitGeode;
	osg::ref_ptr<osg::Geode> _sensorGeode;
	osg::ref_ptr<osg::Geode> _trackGeode;

	osg::ref_ptr<osg::ShapeDrawable> _hitShape;
	osg::ref_ptr<osg::ShapeDrawable> _trackShape;
	
	osg::ref_ptr<osg::Group> sensors;
	osg::ref_ptr<osg::Group> hits;
	osg::ref_ptr<osg::Group> tracks;

	std::vector<int> _sensorCoords;

public:
	osg::ref_ptr<osg::Group> root;
	std::map<int, osg::Node*> sensorByZ;

	void addHit(osg::Vec3&);
	void addTrack(osg::Vec3&, osg::Vec3&);
	void loadSensorCoords(int num_sensors, int*& sensor_Zs);

	VELOModel(){}
	VELOModel(int num_sensors, int*& sensor_Zs);
};
