
#include "token_dictionary.h"
#include "../universal_utils.h"

/**
 * @brief Initialize a token Linked List.
 * 
 * @param list_ptr		The pointer to the token Linked List to initialize.
 */
void token_list_init(token_list_t *list_ptr) {
	list_ptr->head = NULL;
	list_ptr->size = 0;
}

/**
 * @brief Add a token to a token Linked List.
 * 
 * @param list_ptr		The pointer to the token Linked List.
 * @param token			The token to add.
 */
void token_list_add(token_list_t *list_ptr, token_t token) {
	token_t *new_token = mallocBlocking(sizeof(token_t), "token_list_add()");
	memcpy(new_token, &token, sizeof(token_t));
	new_token->next = list_ptr->head;
	list_ptr->head = new_token;
	list_ptr->size++;
}

/**
 * @brief Search a token in a token Linked List.
 * 
 * @param list_ptr		The pointer to the token Linked List.
 * @param token			The token to search (only the size and the str are used)
 * 
 * @return token_t*		The pointer to the token found or NULL if not found.
 */
token_t* token_list_search(token_list_t *list_ptr, token_t token) {
	token_t *current_token = list_ptr->head;
	while (current_token != NULL) {
		if (current_token->size == token.size && strcmp(current_token->str, token.str) == 0)
			return current_token;
		current_token = current_token->next;
	}
	return NULL;
}

/**
 * @brief Search a token in a token Linked List.
 * 
 * @param list_ptr		The pointer to the token Linked List.
 * @param token_id		The id of the token to search.
 * 
 * @return token_t*		The pointer to the token found or NULL if not found.
 */
token_t* token_list_search_id(token_list_t *list_ptr, int token_id) {
	token_t *current_token = list_ptr->head;
	while (current_token != NULL) {
		if (current_token->token_id == token_id)
			return current_token;
		current_token = current_token->next;
	}
	return NULL;
}

/**
 * @brief Delete a token in a token Linked List.
 * 
 * @param list_ptr		The pointer to the token Linked List.
 * @param token_ptr		The pointer to the token to delete.
 * 
 * @return int			0 if the token was found and deleted, -1 if not found.
 */
int token_list_delete(token_list_t *list_ptr, token_t *token_ptr) {

	// Search the token
	token_t *current_token = list_ptr->head;
	token_t *previous_token = NULL;
	while (current_token != NULL) {

		// If found, delete it
		if (current_token == token_ptr) {

			// Remove the token from the list
			if (previous_token == NULL)
				list_ptr->head = current_token->next;
			else
				previous_token->next = current_token->next;
			
			// Free the token, decrease the size and return
			free(current_token->str);
			free(current_token);
			list_ptr->size--;
			return 0;
		}

		// Else, continue to search
		previous_token = current_token;
		current_token = current_token->next;
	}

	// Not found
	return -1;
}

/**
 * @brief Free a token Linked List.
 * 
 * @param list_ptr		The pointer to the token Linked List to free.
 */
void token_list_free(token_list_t *list_ptr) {

	// Free all the tokens
	token_t *current_token = list_ptr->head;
	while (current_token != NULL) {
		token_t *next_token = current_token->next;
		free(current_token->str);
		free(current_token);
		current_token = next_token;
	}

	// Reset the list
	list_ptr->head = NULL;
	list_ptr->size = 0;
}



/**
 * @brief Initialize a token dictionary.
 * 
 * @param dict_ptr			The pointer to the token dictionary to initialize.
 * @param hash_table_size	The size of the hash table.
 */
void token_dict_init(token_dictionary_t *dict_ptr, int hash_table_size) {
	dict_ptr->size = 0;
	dict_ptr->hash_table_size = hash_table_size;
	dict_ptr->table = mallocBlocking(hash_table_size * sizeof(token_list_t), "token_dict_init()");
	for (int i = 0; i < hash_table_size; i++)
		token_list_init(&dict_ptr->table[i]);
}

/**
 * @brief Add a token to a token dictionary.
 * 
 * @param dict_ptr			The pointer to the token dictionary.
 * @param token				The token to add (only the size and the str are used, the token_id is set by the function).
 */
void token_dict_add(token_dictionary_t *dict_ptr, token_t token) {

	// Set the token number
	token.token_id = dict_ptr->size + 1;

	// Add the token to the list
	token_list_add(&dict_ptr->table[token.token_id % dict_ptr->hash_table_size], token);

	// Increase the size
	dict_ptr->size++;
}

/**
 * @brief Search a token in a token dictionary (unefficient, use token_dict_search_id() if possible)
 * 
 * @param dict				The token dictionary.
 * @param token				The token to search (only the size and the str are used)
 * 
 * @return token_t*			The pointer to the token found or NULL if not found.
 */
token_t* token_dict_search(token_dictionary_t dict, token_t token) {
	for (int i = 0; i < dict.hash_table_size; i++) {
		token_t *found_token = token_list_search(&dict.table[i], token);
		if (found_token != NULL)
			return found_token;
	}
	return NULL;
}

/**
 * @brief Search a token in a token dictionary.
 * 
 * @param dict				The token dictionary.
 * @param token_id			The id of the token to search.
 * 
 * @return token_t*			The pointer to the token found or NULL if not found.
 */
token_t* token_dict_search_id(token_dictionary_t dict, int token_id) {
	return token_list_search_id(&dict.table[token_id % dict.hash_table_size], token_id);
}


/**
 * @brief Delete a token in a token dictionary.
 * 
 * @param dict_ptr			The pointer to the token dictionary.
 * @param token_ptr			The pointer to the token to delete.
 * 
 * @return int				0 if the token was found and deleted, -1 if not found.
 */
int token_dict_delete(token_dictionary_t *dict_ptr, token_t *token_ptr) {
	return token_list_delete(&dict_ptr->table[token_ptr->token_id % dict_ptr->hash_table_size], token_ptr);
}

/**
 * @brief Free a token dictionary.
 * 
 * @param dict_ptr			The pointer to the token dictionary to free.
 */
void token_dict_free(token_dictionary_t *dict_ptr) {

	// Free all the lists
	for (int i = 0; i < dict_ptr->hash_table_size; i++)
		token_list_free(&dict_ptr->table[i]);

	// Free the table
	free(dict_ptr->table);

	// Reset the dictionary
	dict_ptr->size = 0;
	dict_ptr->hash_table_size = 0;
	dict_ptr->table = NULL;
}

/**
 * @brief Print a token dictionary.
 * 
 * @param dict		The token dictionary to print.
 * @param nb_tokens	The number of tokens to print (0 to print all)
 */
void token_dict_print(token_dictionary_t dict, int nb_tokens) {

	// Print the size and the hash table size
	INFO_PRINT("token_dict_print(): size: %d, hash_table_size: %d\n", dict.size, dict.hash_table_size);

	// Print all the tokens
	int nb_printed_tokens = 0;
	for (int i = 0; i < dict.hash_table_size; i++) {
		token_t *current_token = dict.table[i].head;
		while (current_token != NULL) {
			printf("%d\t - '%s'\n", current_token->token_id, current_token->str);
			nb_printed_tokens++;
			if (nb_tokens != 0 && nb_printed_tokens >= nb_tokens) return;
			current_token = current_token->next;
		}
	}

	// Print the end
	INFO_PRINT("token_dict_print(): End of token dictionary\n");
}



/**
 * @brief Save a token dictionary to a file.
 * 
 * @param dict_ptr	The pointer to the token dictionary to save.
 * @param filename	The name of the file to save to.
 * 
 * @return int		0 if the token dictionary was saved, -1 else.
*/
int token_dict_save(token_dictionary_t *dict_ptr, char *filename) {

	// Open the file in write mode
	FILE *file = fopen(filename, "wb");
	ERROR_HANDLE_PTR_RETURN_INT(file, "token_dict_save(): Error while opening the file\n");

	// Write size and hash table size
	fwrite(&dict_ptr->size, sizeof(int), 1, file);
	fwrite(&dict_ptr->hash_table_size, sizeof(int), 1, file);

	// Write all the tokens
	for (int i = 0; i < dict_ptr->hash_table_size; i++) {
		token_t *current_token = dict_ptr->table[i].head;
		while (current_token != NULL) {

			// Write the token
			fwrite(&current_token->token_id, sizeof(int), 1, file);
			fwrite(&current_token->size, sizeof(int), 1, file);
			fwrite(current_token->str, sizeof(char), current_token->size, file);

			// Next token
			current_token = current_token->next;
		}
	}

	// Close the file and return
	return fclose(file);
}

/**
 * @brief Load a token dictionary from a file.
 * 
 * @param dict_ptr	The pointer to the token dictionary to load to.
 * @param filename	The name of the file to load from.
 * 
 * @return int		0 if the token dictionary was loaded, -1 else.
 */
int token_dict_load(token_dictionary_t *dict_ptr, char *filename) {

	// Open the file in read mode
	FILE *file = fopen(filename, "rb");
	ERROR_HANDLE_PTR_RETURN_INT(file, "token_dict_load(): Error while opening the file\n");

	// Read size and hash table size
	fread(&dict_ptr->size, sizeof(int), 1, file);
	fread(&dict_ptr->hash_table_size, sizeof(int), 1, file);

	// Allocate the table
	dict_ptr->table = mallocBlocking(dict_ptr->hash_table_size * sizeof(token_list_t), "token_dict_load()");

	// Read all the tokens
	for (int i = 0; i < dict_ptr->hash_table_size; i++)
		token_list_init(&dict_ptr->table[i]);
	for (int i = 0; i < dict_ptr->size; i++) {

		// Read the token
		token_t token;
		fread(&token.token_id, sizeof(int), 1, file);
		fread(&token.size, sizeof(int), 1, file);
		token.str = mallocBlocking((token.size + 1) * sizeof(char), "token_dict_load()");
		fread(token.str, sizeof(char), token.size, file);
		token.str[token.size] = '\0';

		// Add the token directly to the table
		token_list_add(&dict_ptr->table[token.token_id % dict_ptr->hash_table_size], token);
	}

	// Close the file and return
	return fclose(file);
}


/**
 * @brief Convert a sentence to a sequence of tokens.
 * 
 * @param token_dictionary		Pointer to the token dictionary.
 * @param sentence				The sentence to convert.
 * @param sentence_tokens		The array to store the tokens.
 * @param nb_sentence_tokens	Pointer to the number of tokens.
*/
void convertSentenceToTokensArray(token_dictionary_t *token_dictionary, char *sentence, int *sentence_tokens, int *nb_sentence_tokens) {
	
	// Initialize the number of tokens
	*nb_sentence_tokens = 0;

	// Convert the sentence to a sequence of tokens
	char* word = strtok(strdup(sentence), " ");
	while (word != NULL) {

		// Get the token id of the word
		token_t token = {0};
		token.size = strlen(word);
		token.str = word;
		token_t *token_ptr = token_dict_search(*token_dictionary, token);
		int token_id = token_ptr == NULL ? 0 : token_ptr->token_id;
		if (token_id == 0) {
			token_dict_add(token_dictionary, token);
			token.token_id = token_dictionary->size;
			WARNING_PRINT("convertSentenceToTokensArray(): Unknown word: '%s', added to the dictionary with id %d\n", word, token.token_id);
		}

		// Add the token to the sentence tokens and get the next word
		sentence_tokens[(*nb_sentence_tokens)++] = token_id;
		word = strtok(NULL, " ");
	}
}



