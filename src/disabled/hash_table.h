
#ifndef __HASH_TABLE_H__
#define __HASH_TABLE_H__

#include "tree.h"

/**
 * @brief Hash table data structure
 * 
 * @param size Size of the hash table (number of buckets)
 * @param count Current number of elements in the hash table (including tree nodes)
 * @param data Array of tree nodes
*/
struct hash_table_t {
	int size;
	int count;
	struct tree_t* data;
};

// Hash functions
int hash_int(int key, int size);
int hash_float(float key, int size);
int hash_double(double key, int size);
int hash_pointer(void* key, int size);
int hash_string(char* key, int size);

// Hash table functions
struct hash_table_t hash_table_init(int size);



#endif

