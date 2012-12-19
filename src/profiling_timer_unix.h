#ifndef _PRECISIONTIMER_H_
#define _PRECISIONTIMER_H_

#include <sys/time.h>

class CPrecisionTimer
{
    timeval tv_start, tv_end;

	public:
	CPrecisionTimer()
	{
		
	}

	inline void start()
	{
		gettimeofday(&tv_start, 0);
	}

	inline double stop()
	{
		// Return duration in seconds...
		gettimeofday(&tv_end, 0);

		return (double(tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + double(tv_end.tv_usec - tv_start.tv_usec));
	}
};

#endif

