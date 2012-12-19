//=========================================================
//
//A C++ program to implement the mRMR selection using mutual information
// written by Hanchuan Peng.
//
//Disclaimer: The author of program is Hanchuan Peng
//      at <penghanchuan@yahoo.com>.
//
//The CopyRight is reserved by the author.
//
//Last modification: Oct/24/2005
//
//
// make -f mrmr.makefile
//
// by Hanchuan Peng
// 2005-08-01
// 2005-10-17
// 2005-10-20
// 2005-10-22
// 2005-10-24: finalize the computing parts of the whole program
// 2005-10-25: add non-discretization for the classification variable. Also slightly change some output info for the web application
// 2005-11-01: add control to the user-defined max variable number and sample number
// 2006-01-26: add gnu_getline.c to convert the code to be compilable under Max OS.
// 2007-01-25: change the address info

//#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "sort2.h"
#include "mrmr.h"

double compute_mutualinfo (double* pab, long pabhei, long pabwid);

template < class T > void copyvecdata (T* srcdata, long len, int* desdata,
                                       int &nstate);
template < class T > double* compute_jointprob (T* img1, T* img2, long len,
    long maxstatenum,
    int &nstate1, int &nstate2);

double calMutualInfo (DataTable* myData, long v1, long v2);

template < class T > void copyvecdata (T* srcdata, long len, int* desdata, int &nstate)
{
	if (!srcdata || !desdata)
	{
		printf ("NULL points in copyvecdata()!\n");
		return;
	}

	long i;

	//note: originally I added 0.5 before rounding, however seems the negative numbers and
	//      positive numbers are all rounded towarded 0; hence int(-1+0.5)=0 and int(1+0.5)=1;
	//      This is unwanted because I need the above to be -1 and 1.
	// for this reason I just round with 0.5 adjustment for positive and negative differently

	//copy data
	int minn, maxx;

	if (srcdata[0] > 0)
	{
		maxx = minn = int (srcdata[0] + 0.5);

	}

	else
	{
		maxx = minn = int (srcdata[0] - 0.5);
	}

	int tmp;
	double tmp1;

	for (i = 0; i < len; i++)
	{
		tmp1 = double (srcdata[i]);
		tmp = (tmp1 > 0) ? (int) (tmp1 + 0.5) : (int) (tmp1 - 0.5);	//round to integers
		minn = (minn < tmp) ? minn : tmp;
		maxx = (maxx > tmp) ? maxx : tmp;
		desdata[i] = tmp;
		//    printf("%i ",desdata[i]);
	}

	//printf("\n");

	//make the vector data begin from 0 (i.e. 1st state)
	for (i = 0; i < len; i++)
	{
		desdata[i] -= minn;
	}

	//return the #state
	nstate = (maxx - minn + 1);

	return;
}


template < class T > double*
compute_jointprob (T* img1, T* img2, long len, long maxstatenum,
                   int &nstate1, int &nstate2)
{
	//get and check size information

	long i, j;

	if (!img1 || !img2 || len < 0)
	{
		printf ("Line %d: At least one of the input vectors is invalid.\n",
		        __LINE__);
		exit (1);
	}

	int b_findstatenum = 1;
	//  int nstate1 = 0, nstate2 = 0;

	int b_returnprob = 1;

	//copy data into new INT type array (hence quantization) and then reange them begin from 0 (i.e. state1)

	int* vec1 = new int[len];
	int* vec2 = new int[len];

	if (!vec1 || !vec2)
	{
		printf ("Line %d: Fail to allocate memory.\n", __LINE__);
		exit (0);
	}

	int nrealstate1 = 0, nrealstate2 = 0;

	copyvecdata (img1, len, vec1, nrealstate1);
	copyvecdata (img2, len, vec2, nrealstate2);

	//update the #state when necessary
	nstate1 = (nstate1 < nrealstate1) ? nrealstate1 : nstate1;
	//printf("First vector #state = %i\n",nrealstate1);
	nstate2 = (nstate2 < nrealstate2) ? nrealstate2 : nstate2;
	//printf("Second vector #state = %i\n",nrealstate2);

	//  if (nstate1!=maxstatenum || nstate2!=maxstatenum)
	//    printf("find nstate1=%d nstate2=%d\n", nstate1, nstate2);

	//generate the joint-distribution table

	double* hab = new double[nstate1 * nstate2];
	double** hab2d = new double *[nstate2];

	if (!hab || !hab2d)
	{
		printf ("Line %d: Fail to allocate memory.\n", __LINE__);
		exit (0);
	}

	for (j = 0; j < nstate2; j++)
		hab2d[j] = hab + (long) j * nstate1;

	for (i = 0; i < nstate1; i++)
		for (j = 0; j < nstate2; j++)
		{
			hab2d[j][i] = 0;
		}

	for (i = 0; i < len; i++)
	{
		//old method -- slow
		//     indx = (long)(vec2[i]) * nstate1 + vec1[i];
		//     hab[indx] += 1;

		//new method -- fast
		hab2d[vec2[i]][vec1[i]] += 1;
		//printf("vec2[%d]=%d vec1[%d]=%d\n", i, vec2[i], i, vec1[i]);
	}

	//return the probabilities, otherwise return count numbers
	if (b_returnprob)
	{
		for (i = 0; i < nstate1; i++)
			for (j = 0; j < nstate2; j++)
			{
				hab2d[j][i] /= len;
			}
	}

	//finish
	if (hab2d)
	{
		delete[]hab2d;
		hab2d = 0;
	}

	if (vec1)
	{
		delete[]vec1;
		vec1 = 0;
	}

	if (vec2)
	{
		delete[]vec2;
		vec2 = 0;
	}

	return hab;
}


double
compute_mutualinfo (double* pab, long pabhei, long pabwid)
{
	//check if parameters are correct

	if (!pab)
	{
		fprintf (stderr, "Got illeagal parameter in compute_mutualinfo().\n");
		exit (1);
	}

	long i, j;

	double** pab2d = new double *[pabwid];

	for (j = 0; j < pabwid; j++)
		pab2d[j] = pab + (long) j * pabhei;


	double* p1 = 0, *p2 = 0;
	long p1len = 0, p2len = 0;
	int b_findmarginalprob = 1;

	//generate marginal probability arrays
	if (b_findmarginalprob != 0)
	{
		p1len = pabhei;
		p2len = pabwid;
		p1 = new double[p1len];
		p2 = new double[p2len];

		for (i = 0; i < p1len; i++)
			p1[i] = 0;

		for (j = 0; j < p2len; j++)
			p2[j] = 0;

		for (i = 0; i < p1len; i++)	//p1len = pabhei
			for (j = 0; j < p2len; j++)  	//p2len = panwid
			{
				//          printf("%f ",pab2d[j][i]);
				p1[i] += pab2d[j][i];
				p2[j] += pab2d[j][i];
			}
	}



	//calculate the mutual information

	double muInf = 0;

	muInf = 0.0;

	for (j = 0; j < pabwid; j++)  	// should for p2
	{
		for (i = 0; i < pabhei; i++)  	// should for p1
		{
			if (pab2d[j][i] != 0 && p1[i] != 0 && p2[j] != 0)
			{
				muInf += pab2d[j][i] * log (pab2d[j][i] / p1[i] / p2[j]);
				//printf("%f %fab %fa %fb\n",muInf,pab2d[j][i]/p1[i]/p2[j],p1[i],p2[j]);
			}
		}
	}

	muInf /= log (2.0);

	//free memory
	if (pab2d)
	{
		delete[]pab2d;
	}

	if (b_findmarginalprob != 0)
	{
		if (p1)
		{
			delete[]p1;
		}

		if (p2)
		{
			delete[]p2;
		}
	}

	return muInf;
}


double
calMutualInfo (DataTable* myData, long v1, long v2)
{
	double mi = -1;		//initialized as an illegal value

	if (!myData || !myData->data || !myData->data2d)
	{
		fprintf (stderr, "Line %d: The input data is NULL.\n", __LINE__);
		exit (1);
		return mi;
	}

	long nsample = myData->nsample;
	long nvar = myData->nvar;

	if (v1 >= nvar || v2 >= nvar || v1 < 0 || v2 < 0)
	{
		fprintf (stderr,
		         "Line %d: The input variable indexes are invalid (out of range).\n",
		         __LINE__);
		exit (1);
		return mi;
	}

	long i;

	//copy data

	int* v1data = new int[nsample];
	int* v2data = new int[nsample];

	if (!v1data || !v2data)
	{
		fprintf (stderr, "Line %d: Fail to allocate memory.\n", __LINE__);
		exit (1);
		return mi;
	}

	for (i = 0; i < nsample; i++)
	{
		v1data[i] = int (myData->data2d[i][v1]);	//the float already been discretized, should be safe now
		v2data[i] = int (myData->data2d[i][v2]);
	}

	//compute mutual info

	long nstate = 3;		//always true for DataTable, which was discretized as three states

	int nstate1 = 0, nstate2 = 0;
	double* pab =
	  compute_jointprob (v1data, v2data, nsample, nstate, nstate1, nstate2);
	mi = compute_mutualinfo (pab, nstate1, nstate2);
	//printf("pab=%d nstate1=%d nstate2=%d mi=%5.3f\n", long(pab), nstate1, nstate2, mi);

	//free memory and return
	if (v1data)
	{
		delete[]v1data;
		v1data = 0;
	}

	if (v2data)
	{
		delete[]v2data;
		v2data = 0;
	}

	if (pab)
	{
		delete[]pab;
		pab = 0;
	}

	return mi;
}

long* mRMR_selection (DataTable* myData, long nfea, FeaSelectionMethod mymethod)
{
	long* feaInd = 0;

	if (!myData || !myData->data || !myData->data2d)
	{
		fprintf (stderr, "Line %d: The input data is NULL.\n", __LINE__);
		exit (1);
		return feaInd;
	}

	if (nfea < 0)
	{
		fprintf (stderr, "Line %d: The input data nfea is negative.\n",
		         __LINE__);
		exit (1);
		return feaInd;
	}

	//long poolUseFeaLen = myData->nvar - 1; //500;
	long poolUseFeaLen = 500;

	if (poolUseFeaLen > myData->nvar - 1)	// there is a target variable (the first one), that is why must remove one
		poolUseFeaLen = myData->nvar - 1;

	if (nfea > poolUseFeaLen)
		nfea = poolUseFeaLen;

	feaInd = new long[nfea];

	if (!feaInd)
	{
		fprintf (stderr, "Line %d: Fail to allocate memory.\n", __LINE__);
		exit (1);
		return feaInd;
	}

	int i, j;

	//determine the pool

	float* mival = new float[myData->nvar];
	float* poolInd = new float[myData->nvar];
	char* poolIndMask = new char[myData->nvar];

	if (!mival || !poolInd || !poolIndMask)
	{
		fprintf (stderr, "Line %d: Fail to allocate memory.\n", __LINE__);
		exit (1);
		return feaInd;
	}

	for (i = 0; i < myData->nvar; i++)  	//the mival[0] is the entropy of target classification variable
	{
		mival[i] = (float) -calMutualInfo (myData, 0, i);	//set as negative for sorting purpose
		poolInd[i] = (float) i;
		poolIndMask[i] = 1;	//initialized to be everything in pool would be considered
		//      if (i < nfea)   printf ("poolInd[%d]=%d\t%5.3f\n", i, int (poolInd[i]), mival[i]);
	}

	//  printf ("\n{%d}\n", myData->nvar - 1);

	float* NR_mival = mival;	//vector_phc(1,myData->nvar-1);
	float* NR_poolInd = poolInd;	//vector_phc(1,myData->nvar-1);

	sort2 (myData->nvar - 1, NR_mival, NR_poolInd);	// note that poolIndMask is not needed to be sorted, as everything in it is 1 up to this point

	mival[0] = -mival[0];
	printf
	("\nTarget classification variable (#%d column in the input data) has name=%s \tentropy score=%5.3f\n",
	 0 + 1, myData->variableName[0], mival[0]);

	printf ("\n*** MaxRel features ***\n");
	printf ("Order \t Fea \t Name \t Score\n");

	for (i = 1; i < myData->nvar - 1; i++)
	{
		mival[i] = -mival[i];

		if (i <= nfea)
			printf ("%ld \t %d \t %s \t %5.3f\n", i, int (poolInd[i]),
			        myData->variableName[int (poolInd[i])], mival[i]);
	}

	//mRMR selection

	long poolFeaIndMin = 1;
	long poolFeaIndMax = poolFeaIndMin + poolUseFeaLen - 1;

	feaInd[0] = long (poolInd[1]);
	poolIndMask[feaInd[0]] = 0;	//after selection, no longer consider this feature

	poolIndMask[0] = 0;		// of course the first one, which corresponds to the classification variable, should not be considered. Just set the mask as 0 for safety.

	printf ("\n*** mRMR features *** \n");
	printf ("Order \t Fea \t Name \t Score\n");
	printf ("%d \t %ld \t %s \t %5.3f\n", 0 + 1, feaInd[0],
	        myData->variableName[feaInd[0]], mival[1]);

	long k;

	for (k = 1; k < nfea; k++)  	//the first one, feaInd[0] has been determined already
	{
		double relevanceVal, redundancyVal, tmpscore, selectscore;
		long selectind;
		int b_firstSelected = 0;

		for (i = poolFeaIndMin; i <= poolFeaIndMax; i++)
		{
			if (poolIndMask[long (poolInd[i])] == 0)
				continue;		//skip this feature as it was selected already

			relevanceVal = calMutualInfo (myData, 0, long (poolInd[i]));	//actually no necessary to re-compute it, this value can be retrieved from mival vector
			redundancyVal = 0;

			for (j = 0; j < k; j++)
				redundancyVal +=
				  calMutualInfo (myData, feaInd[j], long (poolInd[i]));

			redundancyVal /= k;

			switch (mymethod)
			{
			case MID:
				tmpscore = relevanceVal - redundancyVal;
				break;

			case MIQ:
				tmpscore = relevanceVal / (redundancyVal + 0.0001);
				break;

			default:
				fprintf (stderr,
				         "Undefined selection method=%d. Use MID instead.\n",
				         mymethod);
				tmpscore = relevanceVal - redundancyVal;
			}

			if (b_firstSelected == 0)
			{
				selectscore = tmpscore;
				selectind = long (poolInd[i]);
				b_firstSelected = 1;

			}

			else
			{
				if (tmpscore > selectscore)
				{
					//update the best feature found and the score
					selectscore = tmpscore;
					selectind = long (poolInd[i]);
				}
			}
		}

		feaInd[k] = selectind;
		poolIndMask[selectind] = 0;
		printf ("%ld \t %ld \t %s \t %5.3f\n", k + 1, feaInd[k],
		        myData->variableName[feaInd[k]], selectscore);
	}

	//return
	if (mival)
	{
		delete[]mival;
		mival = 0;
	}

	if (poolInd)
	{
		delete[]poolInd;
		poolInd = 0;
	}

	if (poolIndMask)
	{
		delete[]poolIndMask;
		poolIndMask = 0;
	}

	return feaInd;
}


