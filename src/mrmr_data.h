#ifndef DATATABLE_H
#define DATATABLE_H
class DataTable
{
public:
	float* data;
	float** data2d;
	long nsample;
	long nvar;
	int* classLabel;
	int* sampleNo;
	char** variableName;
	int b_zscore;
	int b_discetize;

	DataTable ();
	~DataTable();

	int buildData2d ();
	void destroyData2d();
	void printData (long nsample_start, long nsample_end, long nvar_start, long nvar_end);
	void printData ();

	void zscore (long indExcludeColumn, int b_discretize);
	void discretize (double threshold, int b_discretize);
};


#endif
