#ifndef _PRECISIONTIMER_H_
#define _PRECISIONTIMER_H_

#include <windows.h>

class CPrecisionTimer
{
	LARGE_INTEGER lFreq, lStart;

	public:
	CPrecisionTimer()
	{
		QueryPerformanceFrequency(&lFreq);
	}

	inline void start()
	{
		QueryPerformanceCounter(&lStart);
	}

	inline double stop()
	{
		// Return duration in seconds...
		LARGE_INTEGER lEnd;
		QueryPerformanceCounter(&lEnd);
		return (double(lEnd.QuadPart - lStart.QuadPart) / lFreq.QuadPart);
	}
};

#endif