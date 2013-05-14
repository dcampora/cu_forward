
#include "HelperFunctions.cuh"

float f_zBeam(track *tr) {
	return -( tr->m_x0 * tr->m_tx + tr->m_y0 * tr->m_ty ) / ( tr->m_tx * tr->m_tx + tr->m_ty * tr->m_ty );
}

float f_r2AtZ( float z , track *tr) {
    float xx = tr->m_x0 + z * tr->m_tx;
    float yy = tr->m_y0 + z * tr->m_ty;
    return xx*xx + yy * yy;
 }

void f_solve (track *tr) {
	float den = ( tr->m_sz2 * tr->m_s0 - tr->m_sz * tr->m_sz );
	if ( fabs(den) < 10e-10 ) den = 1.;
	tr->m_tx     = ( tr->m_sxz * tr->m_s0  - tr->m_sx  * tr->m_sz ) / den;
	tr->m_x0     = ( tr->m_sx  * tr->m_sz2 - tr->m_sxz * tr->m_sz ) / den;

	den = ( tr->m_uz2 * tr->m_u0 - tr->m_uz * tr->m_uz );
	if ( fabs(den) < 10e-10 ) den = 1.;
	tr->m_ty     = ( tr->m_uyz * tr->m_u0  - tr->m_uy  * tr->m_uz ) / den;
	tr->m_y0     = ( tr->m_uy  * tr->m_uz2 - tr->m_uyz * tr->m_uz ) / den;
}

void f_addHit ( track *tr, int offset, int Z ) {

	// track_ids[offset] = tr->internalId;
	tr->trackHitsNum++;

	float z = hit_Zs[offset];
	float x = hit_Xs[offset];
	// float f_w = hit_Ws[offset];

	float wz = f_w * z;

	tr->m_s0  += f_w;
	tr->m_sx  += f_w * x;
	tr->m_sz  += wz;
	tr->m_sxz += wz * x;
	tr->m_sz2 += wz * z;

	float y = hit_Ys[offset];

	tr->m_u0  += f_w;
	tr->m_uy  += f_w * y;
	tr->m_uz  += wz;
	tr->m_uyz += wz * y;
	tr->m_uz2 += wz * z;

	if( tr->trackHitsNum > 1 ) f_solve(tr);

	// tr->hits.push_back(offset);
	tr->hits[Z] = offset;
	
	tr->lastZ = Z;
	tr->activity = 2;
}

void f_setTrack(track *tr, int hit0_offset, int hit1_offset, int hit0_Z, int hit1_Z){

	tr->hits = (int*) ((void**) cudaMalloc(sens_num * sizeof(int)));

	// track_ids[hit0_offset] = tr->internalId;
	tr->trackHitsNum = 1;

	float z = hit_Zs[hit0_offset];
	float x = hit_Xs[hit0_offset];
	// float f_w = hit_Ws[hit0_offset];

	float wz = f_w * z;

	tr->m_s0  = f_w;
	tr->m_sx  = f_w * x;
	tr->m_sz  = wz;
	tr->m_sxz = wz * x;
	tr->m_sz2 = wz * z;

	float y = hit_Ys[hit0_offset];

	tr->m_u0  = f_w;
	tr->m_uy  = f_w * y;
	tr->m_uz  = wz;
	tr->m_uyz = wz * y;
	tr->m_uz2 = wz * z;

	// TODO: Remove when not needed
	// tr->hits.push_back(hit0_offset);
	tr->hits[hit0_Z] = hit0_offset;

	f_addHit (tr, hit1_offset, hit1_Z);
}

float f_chi2Hit( float x, float y, float hitX, float hitY, float hitW){
	float dx = x - hitX;
	float dy = y - hitY;
	return dx * dx * (hitW) + dy * dy * (hitW);
}
float f_xAtHit(track *tr, float z )
{
	return tr->m_x0 + tr->m_tx * z;
}
float f_yAtHit( track *tr, float z  )
{
	return tr->m_y0 + tr->m_ty * z;
}
float f_chi2Track(track *tr, int offset)
{
	float z = hit_Zs[offset];
	return f_chi2Hit( f_xAtHit( tr, z ), f_yAtHit(tr, z ), hit_Xs[offset], hit_Ys[offset], hit_Ws[offset]);
}
float f_chi2(track *t)
{
	float ch = 0.0;
	int nDoF  = -4;
	int hitNumber;
	for (int i=0; i<t->hits.size(); i++){
		hitNumber = t->hits[i];
		ch += f_chi2Track(t, hitNumber);
		nDoF += 2;
	}
	return ch/nDoF;
}

bool f_addHitsOnSensor( f_sensorInfo *sensor, float xTol, float maxChi2,
							 track *tr, int eventId ) {
	
	if (sensor->hitsNum == 0) return false;
	int offset = eventId * max_hits;

	float xGuess = f_xAtHit(tr, sensor->z) - xTol - 1;
	int lastHit = sensor->startPosition + sensor->hitsNum - 1;
	if(hit_Xs[offset + lastHit] < xGuess) return false;

	int hitStart = sensor->startPosition;
	unsigned int step = sensor->hitsNum;
	while ( step > 2 ) {
		step = step/2;
		if (hit_Xs[offset + hitStart + step] < xGuess) hitStart += step;
	}

	bool added = false;
	int tmpOffset = 0;
	float xPred;
	for(int iH=hitStart; iH<=lastHit; ++iH){
		tmpOffset = offset + iH;
		xPred = f_xAtHit(tr, hit_Zs[tmpOffset]);
		if ( hit_Xs[tmpOffset] + xTol < xPred ) continue;
		if ( hit_Xs[tmpOffset] - xTol > xPred ) break;
		if ( f_chi2Track(tr, tmpOffset) < maxChi2 ) {
			f_addHit(tr, tmpOffset, 0);
			// *usedHit = tmpOffset; - Used hits are tagged by the end of the algorithm, not before.
			added = true;
		}
	}
	return added;
}

void f_removeHit(track *tr, int worstHitOffset){
	tr->trackHitsNum--;

	float z = hit_Zs[worstHitOffset];
	// float f_w = hit_Ws[worstHitOffset];
	float x = hit_Xs[worstHitOffset];
	
	float wz = f_w * z;

	tr->m_s0  -= f_w;
	tr->m_sx  -= f_w * x;
	tr->m_sz  -= wz;
	tr->m_sxz -= wz * x;
	tr->m_sz2 -= wz * z;

	float y = hit_Ys[worstHitOffset];

	tr->m_u0  -= f_w;
	tr->m_uy  -= f_w * y;
	tr->m_uz  -= wz;
	tr->m_uyz -= wz * y;
	tr->m_uz2 -= wz * z;

	vector<int>::iterator it = find(tr->hits.begin(), tr->hits.end(), worstHitOffset);

	tr->hits.erase(it);

	if( tr->trackHitsNum > 1 ) f_solve(tr);
}

//== Remove the worst hit until all chi2 are good
void f_removeWorstHit(track* tr)
{
	float topChi2 = 1.e9;
	int worstHitOffset;

	while( topChi2 > f_m_maxChi2PerHit ) {
	    topChi2 = 0.0;


	    // This for loop gets the worst hit
		for (int i=0; i<tr->hits.size(); i++){

			float myChi2 = f_chi2Track(tr, tr->hits[i]);
			if (myChi2 > topChi2){
				topChi2 = myChi2;
				worstHitOffset = tr->hits[i];
			}
		}

	    // If it's bad, we remove it
	    if ( topChi2 > f_m_maxChi2PerHit ) {
	      // hit_isUseds[worstHitOffset] = 0;
		  // It has still not been added to isUseds, no need to do this :)

	      f_removeHit(tr, worstHitOffset);
		  // This changes the chi2 of the track, which is why 
	    }

	    // And the algorithm goes on... ?
	    // Every hit with chi2 > maxChi2 will be removed... is this the desired behaviour?
		// -> yes, read description above
	}
}

bool f_all3SensorsAreDifferent(track *t) {
    float s0 = hit_sensorNums[t->hits[0]];
    float s1 = hit_sensorNums[t->hits[1]];
    float s2 = hit_sensorNums[t->hits[2]];
	
    if ( s0 == s1 ) return false;
    if ( s0 == s2 ) return false;
    if ( s1 == s2 ) return false;
    return true;
}

int f_nbUnused(track *t) {
	int nn = 0;
	for (vector<int>::iterator it = t->hits.begin(); it != t->hits.end(); ++it){
		if (!hit_isUseds[(*it)])
			++nn;
	}
	return nn;
}
