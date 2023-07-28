
#ifndef __UNIVERSAL_PTHREAD_H__
#define __UNIVERSAL_PTHREAD_H__

// This file is used to make the code compatible with Windows and Linux
// It defines the pthread functions for Windows, and includes pthread.h for Linux
// to allow the use of threads in the same way on both OS

#ifdef _WIN32
	#include <windows.h>
	#define thread_return DWORD WINAPI
	#define thread_param LPVOID
	#define pthread_t HANDLE
	#define pthread_create(thread, attr, start_routine, arg) ((*thread = CreateThread(NULL, 0, start_routine, arg, 0, NULL)) == NULL) ? -1 : 0
	#define pthread_join(thread, value_ptr) (WaitForSingleObject(thread, INFINITE) == WAIT_FAILED) ? -1 : 0
	#define pthread_exit(value_ptr) ExitThread(value_ptr)
	#define pthread_mutex_t CRITICAL_SECTION
	#define pthread_mutex_init(mutex, attr) InitializeCriticalSection(mutex)
	#define pthread_mutex_lock(mutex) EnterCriticalSection(mutex)
	#define pthread_mutex_trylock(mutex) TryEnterCriticalSection(mutex)
	#define pthread_mutex_unlock(mutex) LeaveCriticalSection(mutex)
	#define pthread_cond_t CONDITION_VARIABLE
	#define pthread_cond_init(cond, attr) InitializeConditionVariable(cond)
	#define pthread_cond_wait(cond, mutex) SleepConditionVariableCS(cond, mutex, INFINITE)
	#define pthread_cond_timedwait(cond, mutex, abstime) SleepConditionVariableCS(cond, mutex, abstime)
	#define pthread_cond_signal(cond) WakeConditionVariable(cond)
	#define pthread_cond_broadcast(cond) WakeAllConditionVariable(cond)
#else
	#include <pthread.h>
	#define thread_return void *
	#define thread_param void *
#endif


#endif

