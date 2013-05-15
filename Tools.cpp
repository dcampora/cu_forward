
#include "Tools.h"

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
}

void quickSortInput(char*& input){
	int* l_no_sensors = (int*) &input[0];
    int* l_no_hits = (int*) (l_no_sensors + 1);
    int* l_sensor_Zs = (int*) (l_no_hits + 1);
    int* l_sensor_hitStarts = (int*) (l_sensor_Zs + l_no_sensors[0]);
    int* l_sensor_hitNums = (int*) (l_sensor_hitStarts + l_no_sensors[0]);
    int* l_hit_IDs = (int*) (l_sensor_hitNums + l_no_sensors[0]);
    double* l_hit_Xs = (double*) (l_hit_IDs + l_no_hits[0]);
	double* l_hit_Ys = (double*) (l_hit_Xs + l_no_hits[0]);
	int* l_hit_Zs = (int*) (l_hit_Ys + l_no_hits[0]);

	for(int i=0; i<l_no_sensors[0]; i++)
        quickSort(l_hit_Xs, l_hit_Ys, l_hit_IDs, l_hit_Zs,
		    l_sensor_hitStarts[i], l_sensor_hitStarts[i] + l_sensor_hitNums[i]);
}

void quickSort(double*& hit_Xs, double*& hit_Ys, int*& hit_IDs, int*& hit_Zs, int _beginning, int _end)
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
