
#include "hash_table.h"

// /**
//  * @brief Hash function for integers
//  * 
//  * @param key		The key
//  * 
//  * @return int		The hash value between 0 and HASH_TABLE_SIZE
//  */
// int hash_int(int key, int size) { return key % size; }
// int hash_float(float key, int size) { return (int)key % size; }
// int hash_double(double key, int size) { return (int)key % size; }
// int hash_pointer(void* key, int size) { return (int)key % size; }
// int hash_string(char* key, int size) {
// 	int hash = 0;
// 	int i = 0;
// 	int multiplier = 1;
// 	while (key[i] != '\0') {
// 		hash += (key[i] * multiplier) % size;
// 		multiplier *= 256;
// 		i++;
// 	}
// 	return hash % size;
// }


// /**
//  * @brief Initializes the hash table
//  * 
//  * @param size The size of the hash table
//  * 
//  * @return struct hash_table_t	The hash table
//  */
// struct hash_table_t hash_table_init(int size) {

// 	// Allocate memory for the hash table
// 	struct hash_table_t hash_table;
// 	hash_table.size = size;
// 	hash_table.data = malloc(sizeof(struct tree_t) * size);

// 	// Initialize the tree nodes
// 	int i;
// 	for (i = 0; i < size; i++) {
// 		// TODO: Initialize the tree nodes
// 	}

// 	// Return the hash table
// 	return hash_table;
// }

