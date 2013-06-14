
#include "Histo.h"

extern float* h_hit_Xs;
extern float* h_hit_Ys;
extern int* h_hit_Zs;

void Histo::plotChi2(std::string outfile, bool* track_holders, Track* tracks, int num_hits){
	TFile* histo = new TFile(outfile.c_str(), "CREATE");
	TH1I* histo1D = new TH1I("chi2", "chi2", 100, 0, 100);

	for(int i=0; i<num_hits; ++i){
		if(track_holders[i]){
			histo1D->Fill((double) h_trackChi2(tracks[i]));
		}
	}

	histo1D->Write();
	histo->Close();
}

void Histo::plotChi2(std::string outfile, int* track_indexes, Track* tracks, int num_tracks){
	TFile* histo = new TFile(outfile.c_str(), "CREATE");
	TH1I* histo1D = new TH1I("chi2", "chi2", 100, 0, 100);

	for(int i=0; i<num_tracks; ++i){
		histo1D->Fill((double) h_trackChi2(tracks[track_indexes[i]]));
	}

	histo1D->Write();
	histo->Close();
}


float h_trackChi2(Track& t){
	float ch = 0.0;
	int nDoF  = -4 + 2 * t.hitsNum;
	Hit h;
	for (int i=0; i<TRACK_SIZE; i++){
		// TODO: Maybe there's a better way to do this
		if(t.hits[i] != -1){
			h.x = h_hit_Xs[ t.hits[i] ];
			h.y = h_hit_Ys[ t.hits[i] ];

			ch += h_hitChi2(t, h, h_hit_Zs[ t.hits[i] ]);
		}
	}
	return ch/nDoF;
}

float h_hitChi2(Track& t, Hit& h, int hit_z){
	// chi2 of a hit
	float dx = (t.x0 + t.tx * hit_z) - h.x;
	float dy = (t.y0 + t.ty * hit_z) - h.y;
	return dx * dx * PARAM_W + dy * dy * PARAM_W;
}
