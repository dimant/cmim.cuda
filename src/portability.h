#ifdef _MSC_VER

#define PORTABLE_RANDOM_GEN (double(rand()) / RAND_MAX)
#define strncpy strncpy_s

#define strdup _strdup
#define strtok_r strtok_s

#else

#define PORTABLE_RANDOM_GEN drand48()

#endif
