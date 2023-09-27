
#ifndef __TEXT_FILE_UTILS_H__
#define __TEXT_FILE_UTILS_H__

#include <stdio.h>

// Utility functions for reading text files and returning content(s) in different formats
char *readTextFromFile(FILE *file);
char *readTextFromTextFile(char *filename);
char *readTextFromFolder(char *folder);

char *generateCharVocabularyFromText(const char *text, int *vocabulary_size);

int generateSentencesFromFileForGPT(FILE *file, char ***sentences, int max_sentences, int max_words_per_sentence, int *total_sentences);
int generateSentencesFromTextFileForGPT(char *filename, char ***sentences, int max_sentences, int max_words_per_sentence, int *total_sentences);

int generateSentencesFromFolderForGPT(char *folder, char ***sentences, int max_sentences, int max_words_per_sentence, int *total_sentences);



#endif

