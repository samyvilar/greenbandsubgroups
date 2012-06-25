void set_min_max(int *, unsigned int, unsigned int, float *, unsigned int, unsigned int);
void lookuptable(int *, unsigned int, unsigned int, float *, float *, unsigned int);
void predict_double(int *, unsigned int, unsigned int, double *, unsigned int, double *);
void flatten_lookuptable(double *, unsigned int, double *, unsigned int, int);

void update_min_max_lut(float *, float *, unsigned int, unsigned int);
float ***map_1d_array_to_3d(float *, unsigned int);
void free_3d_array(float ***, unsigned int);

