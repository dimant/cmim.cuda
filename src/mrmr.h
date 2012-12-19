#ifndef MRMR_H
#define MRMR_H

#include "mrmr_data.h"

enum FeaSelectionMethod { 
	MID, 
	MIQ 
};

long* mRMR_selection (DataTable* myData, long nfea, FeaSelectionMethod mymethod);

#endif
