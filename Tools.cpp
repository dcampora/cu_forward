
#include "Tools.h"

int* h_no_sensors;
int* h_no_hits;
int* h_sensor_Zs;
int* h_sensor_hitStarts;
int* h_sensor_hitNums;
int* h_hit_IDs;
float* h_hit_Xs;
float* h_hit_Ys;
int* h_hit_Zs;

void readFile(std::string filename, char*& input, int& size){
	// Give me them datas!!11!
	std::ifstream infile (filename.c_str(), std::ifstream::binary);

	// get size of file
	infile.seekg(0, std::ifstream::end);
	size = infile.tellg();
	infile.seekg(0);

	// read content of infile with pointers
	input = (char*) malloc(size);
	infile.read (input, size);
	infile.close();

	h_no_sensors = (int*) &input[0];
	h_no_hits = (int*) (h_no_sensors + 1);
	h_sensor_Zs = (int*) (h_no_hits + 1);
	h_sensor_hitStarts = (int*) (h_sensor_Zs + h_no_sensors[0]);
	h_sensor_hitNums = (int*) (h_sensor_hitStarts + h_no_sensors[0]);
	h_hit_IDs = (int*) (h_sensor_hitNums + h_no_sensors[0]);
	h_hit_Xs = (float*) (h_hit_IDs + h_no_hits[0]);
	h_hit_Ys = (float*) (h_hit_Xs + h_no_hits[0]);
	h_hit_Zs = (int*) (h_hit_Ys + h_no_hits[0]);
}

void quickSortInput(char*& input){
	for(int i=0; i<h_no_sensors[0]; i++)
        quickSort(h_hit_Xs, h_hit_Ys, h_hit_IDs, h_hit_Zs,
		    h_sensor_hitStarts[i], h_sensor_hitStarts[i] + h_sensor_hitNums[i]);
}

void quickSort(float*& hit_Xs, float*& hit_Ys, int*& hit_IDs, int*& hit_Zs, int _beginning, int _end)
{
	const int max_levels = 300;
	int beg[max_levels], end[max_levels], i=0, L, R, swap;

	double piv, d1;
	int i1, i2;

	beg[0]=_beginning; end[0]=_end;
	while (i>=0) {
		L=beg[i]; R=end[i]-1;
		if (L<R) {

			piv = hit_Xs[L];
			d1  = hit_Ys[L];
			i1  = hit_IDs[L];
			i2  = hit_Zs[L];

			while (L<R) {
				while (hit_Xs[R] >= piv && L < R) R--;
				if (L<R){
					hit_Xs[L] = hit_Xs[R];
					hit_Ys[L] = hit_Ys[R];
					hit_Zs[L] = hit_Zs[R];
					hit_IDs[L] = hit_IDs[R];
					L++;
				}

				while (hit_Xs[L] <= piv && L < R) L++;
				if (L<R){
					hit_Xs[R] = hit_Xs[L];
					hit_Ys[R] = hit_Ys[L];
					hit_Zs[R] = hit_Zs[L];
					hit_IDs[R] = hit_IDs[L];
					R--;
				}
			}
			hit_Xs[L] = piv;
			hit_Ys[L] = d1;
			hit_IDs[L] = i1;
			hit_Zs[L] = i2;

			beg[i+1]=L+1; end[i+1]=end[i]; end[i++]=L;
			if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
				swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
				swap=end[i]; end[i]=end[i-1]; end[i-1]=swap;
			}
		}
		else {
			i--;
		}
	}
}
