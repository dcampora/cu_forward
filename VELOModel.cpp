
#include "VELOModel.h"

void VELOModel::addHit(osg::Vec3& coords){

	osg::ref_ptr<osg::MatrixTransform> t1 = new osg::MatrixTransform;
	t1->setMatrix( osg::Matrix::translate(
		coords.x() - HIT_RADIUS,
		coords.y(),
		coords.z()) );
	t1->addChild( _hitGeode.get() );

	hits->addChild( t1.get() );
}

void VELOModel::addTrack(osg::Vec3& v1, osg::Vec3& v2){
	float d = sqrt(pow(v2.x()-v1.x(), 2) + pow(v2.y()-v1.y(), 2) + pow(v2.z()-v1.z(), 2));
	
	osg::ref_ptr<osg::MatrixTransform> t2 = new osg::MatrixTransform;
	t2->setMatrix( osg::Matrix::scale(1.0f, 1.0f, d) );
	t2->addChild( _trackGeode.get() );
	
	osg::ref_ptr<osg::MatrixTransform> t3 = new osg::MatrixTransform;
	t3->setMatrix( osg::Matrix::rotate(osg::Vec3f(0.0f, 0.0f, 1.0f), v2 - v1) );
	t3->addChild( t2.get() );

	osg::ref_ptr<osg::MatrixTransform> t1 = new osg::MatrixTransform;
	t1->setMatrix( osg::Matrix::translate(osg::Vec3f(((v2.x() - v1.x()) / 2.0f) + v1.x(), ((v2.y() - v1.y()) / 2.0f) + v1.y(), ((v2.z() - v1.z()) / 2.0f) + v1.z())) );
	t1->addChild( t3.get() );

	tracks->addChild( t1.get() );
}

void VELOModel::loadSensorCoords(int num_sensors, int*& sensor_Zs){
	for(int i=0; i<num_sensors; ++i){
		_sensorCoords.push_back( static_cast<float>(sensor_Zs[i]) );
	}
}

VELOModel::VELOModel(int num_sensors, int*& sensor_Zs) : 
	_hitGeometry(new HitGeometry),
	_hitGeode(new osg::Geode),
	_sensorGeometry(new SensorGeometry),
	_sensorGeode(new osg::Geode),
	_hitShape(new osg::ShapeDrawable),
	_trackShape(new osg::ShapeDrawable),
	_trackGeode(new osg::Geode),
	sensors(new osg::Group),
	hits(new osg::Group),
	tracks(new osg::Group),
	root(new osg::Group)
	{

	root->addChild(sensors.get());
	root->addChild(hits.get());
	root->addChild(tracks.get());

	// Create hit Drawable
	_hitShape->setShape( new osg::Box(osg::Vec3f(2.0f, 0.0f, 0.0f), HIT_RADIUS) );
	_hitShape->setColor( osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) );

	// Create track Drawable
	_trackShape->setShape( new osg::Cylinder(osg::Vec3f(0.0f, 0.0f, 0.0f),
		TRACK_RADIUS, 1.0f) );
	_trackShape->setColor( osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f) );

	// Create hit and sensor Geometry and Geode
	_hitGeode->addDrawable( _hitShape.get() );
	_sensorGeode->addDrawable( _sensorGeometry.get() );
	_trackGeode->addDrawable( _trackShape.get() );

	float sensorYDisplacement = _sensorGeometry->getBound().yMax() / 2.0f + 10.0f;
	loadSensorCoords(num_sensors, sensor_Zs);

	// Attach a Transform to the _sensorGeometry for each sensor
	for(int i=0; i<_sensorCoords.size(); ++i){
		// Translate in Z
		osg::ref_ptr<osg::MatrixTransform> t1 = new osg::MatrixTransform;
		t1->setMatrix( osg::Matrix::translate( 
			(i%2==0 ? ((SENSOR_EVEN_MAX + SENSOR_EVEN_MIN) / 2.0f) - (SENSOR_SIZE / 2.0f) :
					  ((SENSOR_ODD_MAX + SENSOR_ODD_MIN) / 2.0f) - (SENSOR_SIZE / 2.0f)),
			-(SENSOR_SIZE / 2.0f),
			_sensorCoords[i]) );
		t1->addChild( _sensorGeode.get() );

		sensors->addChild( t1.get() );
		sensorByZ[_sensorCoords[i]] = t1.get();
	}
}

/*

	osg::ref_ptr<osg::Geode> g_quad = new osg::Geode;
	g_quad->addDrawable( quad.get() );

	osg::ref_ptr<osg::MatrixTransform> transform1 = new osg::MatrixTransform;
	transform1->setMatrix( osg::Matrix::translate(0.0f, 1.0f, 0.0f) );
	transform1->addChild( g_quad.get() );

	osg::ref_ptr<osg::Group> root = new osg::Group;
	root->addChild( g_quad.get() );
	root->addChild( transform1.get() );
	*/