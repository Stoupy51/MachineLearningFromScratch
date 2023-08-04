
#ifndef __TOKEN_DICTIONARY_H__
#define __TOKEN_DICTIONARY_H__

/**
 * @brief Structure representing a token in a token Dictionary and Linked List.
 * 
 * @param str			The string representing the token. ex: "hello"
 * @param size			The size of the token == strlen. ex: 5
 * @param token_id		The ID of the token in the Dictionary. ex: 1
 * @param next			The next token in a Linked List of tokens.
 */
typedef struct token_t {
	char *str;
	int size;
	int token_id;
	struct token_t *next;
} token_t;

/**
 * @brief Structure representing a token Linked List.
 * 
 * @param head		The first token in the Linked List.
 * @param size		The number of tokens in the Linked List.
 */
typedef struct token_list_t {
	token_t *head;
	int size;
} token_list_t;

/**
 * @brief Structure representing a token Dictionary using a Hash Table.
 * 
 * @param hash_table_size		The size of the Hash Table.
 * @param size			The number of tokens in the Dictionary.
 * @param table			The Hash Table.
 */
typedef struct token_dictionary_t {
	int hash_table_size;
	int size;
	token_list_t *table;
} token_dictionary_t;


// Functions declarations

void token_list_init(token_list_t *list_ptr);
void token_list_add(token_list_t *list_ptr, token_t token);
token_t* token_list_search(token_list_t *list_ptr, token_t token);
token_t* token_list_search_id(token_list_t *list_ptr, int token_id);
int token_list_delete(token_list_t *list_ptr, token_t *token_ptr);
void token_list_free(token_list_t *list_ptr);

int token_hash(token_t token, int hash_table_size);
void token_dict_init(token_dictionary_t *dict_ptr, int hash_table_size);
void token_dict_add(token_dictionary_t *dict_ptr, token_t token);
token_t* token_dict_search(token_dictionary_t dict, token_t token);
token_t* token_dict_search_id(token_dictionary_t dict, int token_id);
int token_dict_delete(token_dictionary_t *dict_ptr, token_t *token_ptr);
void token_dict_free(token_dictionary_t *dict_ptr);
void token_dict_print(token_dictionary_t dict, int nb_tokens);

int token_dict_save(token_dictionary_t *dict_ptr, char *filename);
int token_dict_load(token_dictionary_t *dict_ptr, char *filename);

#endif

