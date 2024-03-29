
#ifndef __ST_BENCHMARK_H__
#define __ST_BENCHMARK_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <stdint.h>

#ifdef _WIN32

#include <windows.h>

#define st_gettimeofday(tp, tzp) { \
	LARGE_INTEGER counter, frequency; \
	QueryPerformanceCounter(&counter); \
	QueryPerformanceFrequency(&frequency); \
	tp.tv_sec = counter.QuadPart / frequency.QuadPart; \
	tp.tv_usec = (counter.QuadPart % frequency.QuadPart) * 1000000 / frequency.QuadPart; \
}

#else
	#include <unistd.h>
	#include <sys/time.h>
	#define st_gettimeofday(tp, tzp) gettimeofday(&tp, tzp)
#endif

#define ST_COLOR_RESET "\033[0m"
#define ST_COLOR_RED "\033[0;31m"
#define ST_COLOR_YELLOW "\033[0;33m"

#define ST_BENCHMARK_BETWEEN(ST_BENCH_buffer, ST_BENCH_f1, ST_BENCH_f2, ST_BENCH_f1_name, ST_BENCH_f2_name, ST_BENCH_testing_time) { \
	long ST_BENCH_countF1 = 0; \
	long ST_BENCH_countF2 = 0; \
	time_t ST_BENCH_end = time(NULL) + 1; \
	while (time(NULL) < ST_BENCH_end) {} \
	ST_BENCH_end = time(NULL) + ST_BENCH_testing_time; \
	while (time(NULL) < ST_BENCH_end) { \
		ST_BENCH_f1; \
		ST_BENCH_countF1 += 1; \
	} \
	ST_BENCH_end = time(NULL) + ST_BENCH_testing_time; \
	while (time(NULL) < ST_BENCH_end) { \
		ST_BENCH_f2; \
		ST_BENCH_countF2 += 1; \
	} \
	sprintf(ST_BENCH_buffer, ST_COLOR_YELLOW "[BENCHMARK] " ST_COLOR_RED "'%s" ST_COLOR_RED "' %s" ST_COLOR_RED " than '%s'" ST_COLOR_RED " by " ST_COLOR_YELLOW "%f" ST_COLOR_RED " times with" ST_COLOR_YELLOW "\n[BENCHMARK] " ST_COLOR_RED ST_BENCH_f1_name " executed " ST_COLOR_YELLOW "%ld" ST_COLOR_RED " time%s and " ST_BENCH_f2_name " executed " ST_COLOR_YELLOW "%ld" ST_COLOR_RED " time%s\n" ST_COLOR_RESET, ST_BENCH_f1_name, (ST_BENCH_countF1 > ST_BENCH_countF2) ? "faster" : "slower", ST_BENCH_f2_name, (double)ST_BENCH_countF1 / (double)ST_BENCH_countF2, ST_BENCH_countF1, (ST_BENCH_countF1 == 1 ? "" : "s"), ST_BENCH_countF2, (ST_BENCH_countF2 == 1 ? "" : "s")); \
}


#define ST_BENCHMARK_SOLO_COUNT(ST_BENCH_buffer, ST_BENCH_f, ST_BENCH_f_name, ST_BENCH_count, ST_RETURN_TIME_ONLY) { \
	struct timeval ST_BENCH_timeval, ST_BENCH_timeval2; \
	st_gettimeofday(ST_BENCH_timeval, NULL); \
	unsigned long ST_BENCH_time = 1000000 * ST_BENCH_timeval.tv_sec + ST_BENCH_timeval.tv_usec; \
	long ST_BENCH_i = 0; \
	for (ST_BENCH_i = 0; ST_BENCH_i < ST_BENCH_count; ST_BENCH_i++) { \
		ST_BENCH_f; \
	} \
	st_gettimeofday(ST_BENCH_timeval2, NULL); \
	ST_BENCH_time = 1000000 * ST_BENCH_timeval2.tv_sec + ST_BENCH_timeval2.tv_usec - ST_BENCH_time; \
	if (ST_RETURN_TIME_ONLY) \
		sprintf(ST_BENCH_buffer, "%lf", (double)ST_BENCH_time / 1000000.0); \
	else if (ST_BENCH_count != 1) \
		sprintf(ST_BENCH_buffer, ST_COLOR_YELLOW "[BENCHMARK] " ST_COLOR_RED "'%s" ST_COLOR_RED "' executed " ST_COLOR_YELLOW "%d" ST_COLOR_RED " time%s in " ST_COLOR_YELLOW "%lf" ST_COLOR_RED "s\n" ST_COLOR_RESET, ST_BENCH_f_name, ST_BENCH_count, (ST_BENCH_count == 1 ? "" : "s"), (double)ST_BENCH_time / 1000000.0); \
	else \
		sprintf(ST_BENCH_buffer, ST_COLOR_YELLOW "[BENCHMARK] " ST_COLOR_RED "'%s" ST_COLOR_RED "' executed in " ST_COLOR_YELLOW "%lf" ST_COLOR_RED "s\n" ST_COLOR_RESET, ST_BENCH_f_name, (double)ST_BENCH_time / 1000000.0); \
}


#define ST_BENCHMARK_SOLO_TIME(ST_BENCH_buffer, ST_BENCH_f, ST_BENCH_f_name, ST_BENCH_testing_time) { \
	long ST_BENCH_count = 0; \
	time_t ST_BENCH_end = time(NULL) + 1; \
	while (time(NULL) < ST_BENCH_end) {} \
	ST_BENCH_end = time(NULL) + (long)ST_BENCH_testing_time; \
	while (time(NULL) < ST_BENCH_end) { \
		ST_BENCH_f; \
		ST_BENCH_count += 1; \
	} \
	sprintf(ST_BENCH_buffer, ST_COLOR_YELLOW "[BENCHMARK] " ST_COLOR_RED "'%s" ST_COLOR_RED "' executed " ST_COLOR_YELLOW "%ld" ST_COLOR_RED " time%s in " ST_COLOR_YELLOW "%ld" ST_COLOR_RED "s\n" ST_COLOR_RESET, ST_BENCH_f_name, ST_BENCH_count, (ST_BENCH_count == 1 ? "" : "s"), (long)ST_BENCH_testing_time); \
}

#endif

