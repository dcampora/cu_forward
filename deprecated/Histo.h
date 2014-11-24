
#include "TFile.h"
#include "TH1I.h"

#include <string>
#include "Definitions.cuh"

class Histo {
private:
	TFile* outfile;
	TH1I* h1D;

public:
	Histo(){}

	void plotChi2(std::string outfile, bool* track_holders, Track* tracks, int num_hits);
	void plotChi2(std::string outfile, int* track_indexes, Track* tracks, int num_tracks);
};

float h_trackChi2(Track& t);
float h_hitChi2(Track& t, Hit& h, int hit_z);
