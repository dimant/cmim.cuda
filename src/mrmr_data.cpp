#include "mrmr_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

template < class T > T putInRange (T val, T minval, T maxval)
{
	T myval = val;

	if (minval > maxval)
	{
		fprintf (stderr,
		         "The input parameters of putInRange() have error. The result is wrong!\n");
		return myval;
	}

	if (myval < minval)
		myval = minval;

	else
	{
		if (myval > maxval)
			myval = maxval;
	}

	return myval;
}



DataTable::DataTable () :
	data(NULL),
	data2d(NULL),
	nsample(0),
	nvar(0),
	classLabel(NULL),
	sampleNo(NULL),
	variableName(NULL),
	b_zscore(0),		// initialze the data as not being z-scored
	b_discetize(0)	// initialze the data as continous
{}

DataTable::~DataTable ()
{
	if (data)
	{
		delete[]data;
		data = NULL;
	}

	if (classLabel)
	{
		delete[]classLabel;
		classLabel = NULL;
	}

	if (sampleNo)
	{
		delete[]sampleNo;
		sampleNo = NULL;
	}

	if (variableName)
	{
		for (long i = 0; i < nvar; i++)
			if (variableName[i])
			{
				delete[]variableName[i];
			}

		delete[]variableName;
		variableName = NULL;
	}

	if (data2d)
	{
		destroyData2d ();
	}

	nsample = nvar = 0;
	return;
}

int DataTable::buildData2d ()
{
	if (!data)
	{
		fprintf (stderr, "The data is not ready yet: data point is NULL");
	}

	if (data2d)
		destroyData2d ();

	if (nsample <= 0 || nvar <= 0)

	{
		fprintf (stderr, "The data is not ready yet: nsample=%ld nvar=%ld",
		         nsample, nvar);
		return 0;
	}

	data2d = new float *[nsample];

	if (!data2d)
	{
		fprintf (stderr, "Line %d: Fail to allocate memory.\n", __LINE__);
		return 0;
	}

	else
	{
		for (long i = 0; i < nsample; i++)
		{
			data2d[i] = data + i * nvar;
		}
	}

	return 1;
}

void DataTable::destroyData2d ()
{
	if (data2d)
	{
		delete[]data2d;
		data2d = NULL;
	}
}

void DataTable::printData (long nsample_start, long nsample_end, long nvar_start,
                           long nvar_end)
{
	long ns0 = putInRange (nsample_start - 1, long (0), nsample - 1);
	long ns1 = putInRange (nsample_end - 1, long (0), nsample - 1);
	long nv0 = putInRange (nvar_start - 1, long (0), nvar - 1);
	long nv1 = putInRange (nvar_end - 1, long (0), nvar - 1);

	printf ("%ld %ld %ld %ld\n", ns0, ns1, nv0, nv1);

	long i, j;

	if (variableName)
	{
		if (classLabel)
			printf ("<label>\t");

		for (i = nv0; i <= nv1; i++)
		{
			printf ("[%s]", variableName[i]);
		}

		printf ("\n");
	}

	for (i = ns0; i <= ns1; i++)
	{
		if (classLabel)
			printf ("<%d>\t", classLabel[i]);

		for (j = nv0; j <= nv1; j++)
			printf ("%5.3f\t", data[i * nvar + j]);

		printf ("\n");
	}
}

void DataTable::printData ()
{
	printData (1, nsample, 1, nvar);
}

void DataTable::zscore (long indExcludeColumn, int b_discretize)
{
	if (!data2d)
		buildData2d ();

	if (!b_discretize)
		return; // in this case, just generate the 2D data array

	long i, j;

	for (j = 0; j < nvar; j++)
	{
		if (j == indExcludeColumn)
		{
			continue; //this is useful to exclude the first column, which will be the target classification variable
		}

		double cursum = 0;
		double curmean = 0;
		double curstd = 0;

		for (i = 0; i < nsample; i++)
			cursum += data2d[i][j];

		curmean = cursum / nsample;
		cursum = 0;
		register double tmpf;

		for (i = 0; i < nsample; i++)
		{
			tmpf = data2d[i][j] - curmean;
			cursum += tmpf * tmpf;
		}

		curstd = (nsample == 1) ? 0 : sqrt (cursum / (nsample - 1));	//nsample -1 is an unbiased version for Gaussian

		for (i = 0; i < nsample; i++)
		{
			data2d[i][j] = (data2d[i][j] - curmean) / curstd;
		}
	}

	b_zscore = 1;
}

void DataTable::discretize (double threshold, int b_discretize)
{
	long indExcludeColumn = 0; //exclude the first column

	if (b_zscore == 0)
	{
		zscore (indExcludeColumn, b_discretize); //exclude the first column
	}

	if (!b_discretize)
		return; // in this case, just generate the 2D array

	long i, j;

	for (j = 0; j < nvar; j++)
	{
		if (j == indExcludeColumn)
		{
			continue;
		}

		register double tmpf;

		for (i = 0; i < nsample; i++)
		{
			tmpf = data2d[i][j];

			if (tmpf > threshold)
				tmpf = 1;

			else
			{
				if (tmpf < -threshold)
					tmpf = -1;

				else
					tmpf = 0;
			}

			data2d[i][j] = tmpf;
		}
	}

	b_discetize = 1;
}

